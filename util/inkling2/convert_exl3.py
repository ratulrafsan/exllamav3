"""
EXL3 conversion of Inkling-2-Mini's routed experts.

The stock converter (convert.py) walks the modules once, in order, and cannot drive a looped model: the loop layers
see three different input distributions and the layers after the loop need the third pass's output. This script
does the calibration itself and hands the weights and Hessians to ExLlamaV3's quantizer:

  capture    run the calibration rows through the unquantized model (experts in the checkpoint's packed form) and
             accumulate, for every expert of every layer, the Hessians of its gate/up input and of its down input,
             over all loop passes. No error propagation between layers: every layer is calibrated on the
             unquantized model's activations.
  quantize   per layer: dequantize the experts (NVFP4 / MXFP8 -> fp32), quantize gate_proj / up_proj / down_proj of
             each expert to EXL3 (gate and up share a Hessian and are quantized together; down projections in
             batches)
  assemble   write the output model directory: the non-expert tensors as stored (bf16 / fp32), the EXL3 expert
             tensors, config and tokenizer files

Only the experts are quantized (93% of the parameters); everything else stays in the checkpoint's precision.

    python util/inkling2/convert_exl3.py <model dir> <out dir> --work <work dir> \
        [--stage capture quantize assemble] [--bits 6] [--bits-mx 8] [--rows 100] [--cols 2048] \
        [--layers 0 1 ...] [--device cuda:0] [--h-device cuda:1]

Stages can be run separately; `quantize --layers ... --device ...` lets two processes share the layers over two GPUs.
"""
import argparse, json, os, shutil, sys, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file

TEXT_FILES = [
    "chat_template.jinja", "generation_config.json", "processor_config.json", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json", "README.md", "mtp.safetensors",
]


def log(msg):
    print(msg, flush = True)


@torch.inference_mode()
def capture(args):
    from exllamav3 import Config, Model, Tokenizer
    from exllamav3.modules import inkling2 as I2
    from exllamav3.modules.quant.exl3_lib.hessian_packed import packed_numel, accumulate_
    from exllamav3.conversion.calibration_data import get_default_calibration

    config = Config.from_directory(args.model)
    cfg = config.i2
    model = Model.from_config(config)
    model.load(device = args.device, progressbar = True)
    tokenizer = Tokenizer.from_config(config)
    rows = get_default_calibration({"cal_rows": args.rows, "cal_cols": args.cols}, tokenizer)
    log(f" -- Calibration: {len(rows)} rows of {args.cols} tokens")

    hdev = torch.device(args.h_device)
    todo = [i for i in (args.layers or range(cfg.layers))
            if not os.path.exists(os.path.join(args.work, f"H_{i:02d}.safetensors"))]
    D, I, E = cfg.hidden, cfg.inter, cfg.experts
    per_layer = E * (packed_numel(D) + packed_numel(I)) * 4
    free = torch.cuda.mem_get_info(hdev)[0] - (2 << 30)
    group = max(1, min(args.group or 99, free // per_layer))
    log(f" -- {len(todo)} layers to capture, {per_layer / 2**30:.2f} GiB of Hessians per layer, {group} layers per pass on {hdev}")

    for g0 in range(0, len(todo), group):
        layers = todo[g0 : g0 + group]
        H13 = {i: torch.zeros((E, packed_numel(D)), dtype = torch.float32, device = hdev) for i in layers}
        H2 = {i: torch.zeros((E, packed_numel(I)), dtype = torch.float32, device = hdev) for i in layers}
        count = {i: torch.zeros(E, dtype = torch.long) for i in layers}
        bad = {i: 0 for i in layers}

        def hook(W, e, x, a):
            i = W.i
            if i not in H13:
                return
            ok = torch.isfinite(x).all(1) & torch.isfinite(a).all(1)
            if not bool(ok.all()):
                bad[i] += int((~ok).sum())
                x, a = x[ok], a[ok]
            if x.shape[0] == 0:
                return
            accumulate_(H13[i][e], x.to(hdev).contiguous())
            accumulate_(H2[i][e], a.to(hdev).contiguous())
            count[i][e] += x.shape[0]

        I2.MOE_HOOK = hook
        t0 = time.time()
        for r, ids in enumerate(rows):
            model.forward(ids, {"last_tokens_only": 1})
            if (r + 1) % 10 == 0 or r + 1 == len(rows):
                el = time.time() - t0
                log(f"    layers {layers[0]}..{layers[-1]}: row {r + 1}/{len(rows)}, {el:.0f} s, "
                    f"eta {el / (r + 1) * (len(rows) - r - 1):.0f} s")
        I2.MOE_HOOK = None
        for i in layers:
            save_file({"H13": H13[i].cpu(), "H2": H2[i].cpu(), "count": count[i]},
                      os.path.join(args.work, f"H_{i:02d}.safetensors"))
            c = count[i]
            log(f" -- Layer {i}: tokens per expert min {int(c.min())}, median {int(c.median())}, max {int(c.max())}"
                + (f"; {bad[i]} non-finite rows dropped" if bad[i] else ""))
        del H13, H2
        torch.cuda.empty_cache()
    model.unload()


class Source:
    def __init__(self, directory, device):
        with open(os.path.join(directory, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self.dir, self.device, self.handles = directory, device, {}

    def has(self, key):
        return key in self.map

    def get(self, key, device = None):
        fn = self.map[key]
        if fn not in self.handles:
            self.handles[fn] = safe_open(os.path.join(self.dir, fn), "pt")
        return self.handles[fn].get_tensor(key).to(self.device if device is None else device)


@torch.inference_mode()
def quantize(args):
    from exllamav3.modules import inkling2 as I2
    from exllamav3.modules.quant import LinearEXL3
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3_batch

    cfg = I2.Inkling2Cfg.from_dir(args.model)
    dev = torch.device(args.device)
    src = Source(args.model, dev)
    D, I, E = cfg.hidden, cfg.inter, cfg.experts
    debug_dir = os.path.join(args.work, "debug")
    os.makedirs(debug_dir, exist_ok = True)

    for i in (args.layers or range(cfg.layers)):
        out_path = os.path.join(args.work, f"q_{i:02d}.safetensors")
        if os.path.exists(out_path):
            continue
        h_path = os.path.join(args.work, f"H_{i:02d}.safetensors")
        assert os.path.exists(h_path), f"no Hessians for layer {i}: run the capture stage first"
        ex = I2.PackedExperts(src.get, src.has, cfg, i)
        K = args.bits if ex.nvfp4 else args.bits_mx
        Hs = load_file(h_path)
        count = Hs["count"]
        key = f"model.llm.layers.{i}.mlp.experts"
        out, errs = {}, {"gate_proj": [], "up_proj": [], "down_proj": []}
        t0 = time.time()

        def h_data(packed, n, e, first_key):
            return {
                "H": packed.to(dev), "packed": True, "n": n, "first_key": first_key, "count": int(count[e]),
                "finalized": False, "num_total": int(count[e]) * n,
                "inf_nan": torch.zeros(2, dtype = torch.long, device = dev), "device": dev,
            }

        def q_args():
            return {"seed": i, "K": K, "devices": [dev.index], "device_ratios": None, "apply_out_scales": None,
                    "debug_dir": debug_dir, "mcg": True}

        def store(e, name, fi, fo, proxy_err, tensors):
            lin = LinearEXL3(
                None, fi, fo, tensors.get("scale"), tensors.get("su"), tensors.get("sv"), tensors.get("suh"),
                tensors.get("svh"), tensors.get("trellis"), tensors.get("mcg"), tensors.get("mul1"), None,
                torch.half, key = f"{key}.{e}.{name}")
            out.update({k: v.cpu() for k, v in lin.get_tensors(f"{key}.{e}.{name}").items()})
            errs[name].append(float(proxy_err))

        # gate and up of one expert share the input Hessian: one batched pass. Weights go in as (in, out)
        for e in range(E):
            w13 = ex.w13_f32(e)                                    # [2 * inter, D], gate rows first
            hd = h_data(Hs["H13"][e], D, e, f"{key}.{e}.gate_proj")
            res = quantize_exl3_batch([w13[:I].t().contiguous(), w13[I:].t().contiguous()], [hd, hd], [q_args(), q_args()])
            store(e, "gate_proj", D, I, *res[0])
            store(e, "up_proj", D, I, *res[1])
            del w13, hd, res
            if (e + 1) % 16 == 0:
                log(f"    layer {i}: gate/up {e + 1}/{E}, {time.time() - t0:.0f} s")

        # down projections: same shape, one Hessian each, in stacks
        for e0 in range(0, E, args.stack):
            es = list(range(e0, min(E, e0 + args.stack)))
            ws = [ex.w2_f32(e).t().contiguous() for e in es]       # (inter, D)
            hds = [h_data(Hs["H2"][e], I, e, f"{key}.{e}.down_proj") for e in es]
            res = quantize_exl3_batch(ws, hds, [q_args() for _ in es])
            for e, r in zip(es, res):
                store(e, "down_proj", I, D, *r)
            del ws, hds, res
        save_file(out, out_path)
        m = lambda v: sum(v) / len(v)
        log(f" -- Layer {i} ({'NVFP4' if ex.nvfp4 else 'MXFP8'} -> {K} bpw): proxy err gate {m(errs['gate_proj']):.5f} "
            f"(max {max(errs['gate_proj']):.5f}), up {m(errs['up_proj']):.5f} (max {max(errs['up_proj']):.5f}), "
            f"down {m(errs['down_proj']):.5f} (max {max(errs['down_proj']):.5f}); "
            f"{sum(t.numel() * t.element_size() for t in out.values()) / 2**20:.0f} MiB, {time.time() - t0:.0f} s")
        del ex, Hs, out
        torch.cuda.empty_cache()


def assemble(args):
    from exllamav3.modules import inkling2 as I2
    from exllamav3.version import __version__ as exl3_version
    cfg = I2.Inkling2Cfg.from_dir(args.model)
    os.makedirs(args.out, exist_ok = True)
    missing = [i for i in range(cfg.layers) if not os.path.exists(os.path.join(args.work, f"q_{i:02d}.safetensors"))]
    assert not missing, f"layers not quantized yet: {missing}"

    src = Source(args.model, "cpu")
    rest = {k: src.get(k) for k in src.map if ".mlp.experts." not in k}
    save_file(rest, os.path.join(args.out, "model-base.safetensors"))
    log(f" -- model-base.safetensors: {len(rest)} tensors as stored, "
        f"{sum(t.numel() * t.element_size() for t in rest.values()) / 2**30:.2f} GiB")
    total = 0
    for i in range(cfg.layers):
        dst = os.path.join(args.out, f"model-experts-{i:02d}.safetensors")
        shutil.copyfile(os.path.join(args.work, f"q_{i:02d}.safetensors"), dst)
        total += os.path.getsize(dst)
    log(f" -- 24 expert files: {total / 2**30:.2f} GiB")

    with open(os.path.join(args.model, "config.json")) as f:
        config = json.load(f)
    n_exp = cfg.experts * 3 * cfg.hidden * cfg.inter
    nv = len([i for i in range(cfg.layers) if src.has(f"model.llm.layers.{i}.mlp.experts.w13_weight_amax")])
    bits = (nv * args.bits + (cfg.layers - nv) * args.bits_mx) / cfg.layers
    config["quantization_config"] = {
        "quant_method": "exl3", "version": exl3_version, "bits": round(bits, 2), "head_bits": 16,
        "calibration": {"rows": args.rows, "cols": args.cols}, "codebook": "mcg",
        "note": "routed experts only; all other tensors as in the source checkpoint",
        "expert_bits": {"nvfp4_layers": args.bits, "mxfp8_layers": args.bits_mx},
        "original_quantization_config": config.get("quantization_config"),
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent = 2)
    for name in TEXT_FILES:
        p = os.path.join(args.model, name)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(args.out, name))
    log(f" -- Wrote {args.out}: experts {bits:.2f} bpw on average ({n_exp * cfg.layers / 1e9:.1f}B expert weights)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--work", required = True)
    ap.add_argument("--stage", nargs = "*", default = ["capture", "quantize", "assemble"])
    ap.add_argument("--bits", type = float, default = 6, help = "bits per weight for the experts stored as NVFP4")
    ap.add_argument("--bits-mx", type = float, default = 8, help = "bits per weight for the experts stored as MXFP8")
    ap.add_argument("--rows", type = int, default = 100)
    ap.add_argument("--cols", type = int, default = 2048)
    ap.add_argument("--layers", type = int, nargs = "*")
    ap.add_argument("--device", default = "cuda:0", help = "model (capture) / quantizer device")
    ap.add_argument("--h-device", default = "cuda:1", help = "device the Hessians accumulate on during capture")
    ap.add_argument("--group", type = int, help = "layers captured per calibration pass (default: what fits)")
    ap.add_argument("--stack", type = int, default = 16, help = "down projections quantized per batch")
    args = ap.parse_args()
    args.bits = int(args.bits) if float(args.bits).is_integer() else args.bits
    args.bits_mx = int(args.bits_mx) if float(args.bits_mx).is_integer() else args.bits_mx
    os.makedirs(args.work, exist_ok = True)
    for stage in args.stage:
        t0 = time.time()
        {"capture": capture, "quantize": quantize, "assemble": assemble}[stage](args)
        log(f" -- {stage}: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
