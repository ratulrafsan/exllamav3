"""
Step 2a check: the plain-PyTorch Inkling-2 math in exllamav3/modules/inkling2.py against the transformers
ground truth (the *.npz files written by ref_dump.py). Reads the source checkpoint directly, experts kept packed
(NVFP4 / MXFP8) on one GPU and dequantized per routed expert.

    python util/inkling2/port_check.py <model dir> --ref <ground truth dir> [--only name ...] [--device cuda:1]

Per prompt it prints, for every layer application, the relative RMS error of the layer output and the share of
routed expert ids that agree, then the error of the final hidden state and of the logits and the argmax agreement.
"""
import argparse, glob, importlib.util, json, os, sys, time

import numpy as np
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "inkling2", os.path.join(HERE, "..", "..", "exllamav3", "modules", "inkling2.py"))
I2 = importlib.util.module_from_spec(spec)
sys.modules["inkling2"] = I2
spec.loader.exec_module(I2)


class Source:
    def __init__(self, directory, device):
        with open(os.path.join(directory, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self.dir, self.device, self.handles = directory, device, {}

    def has(self, key):
        return key in self.map

    def get(self, key):
        fn = self.map[key]
        if fn not in self.handles:
            self.handles[fn] = safe_open(os.path.join(self.dir, fn), "pt")
        return self.handles[fn].get_tensor(key).to(self.device)


def rel(a, b):
    a, b = a.float(), b.float()
    return (torch.sqrt((a - b).square().mean()) / torch.sqrt(b.square().mean()).clamp_min(1e-12)).item()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ref", default = os.path.join(HERE, "ref"))
    ap.add_argument("--only", nargs = "*")
    ap.add_argument("--device", default = "cuda:1")
    ap.add_argument("--verbose", action = "store_true")
    ap.add_argument("--ref-exact", action = "store_true",
                    help = "reproduce the reference's bf16 rounding of attention logits and of the NVFP4 amax")
    ap.add_argument("--teacher", action = "store_true",
                    help = "feed every layer application the reference's input (single-layer error, no accumulation)")
    args = ap.parse_args()
    dev = torch.device(args.device)
    adt = torch.bfloat16
    I2.REF_EXACT = args.ref_exact

    cfg = I2.Inkling2Cfg.from_dir(args.model)
    src = Source(args.model, dev)
    t0 = time.time()
    embed, embed_norm = src.get("model.llm.embed.weight"), src.get("model.llm.embed_norm.weight")
    W = [I2.LayerWeights(src.get, cfg, i) for i in range(cfg.layers)]
    EX = [I2.PackedExperts(src.get, src.has, cfg, i) for i in range(cfg.layers)]
    Hd = I2.HeadWeights(src.get, cfg)
    torch.cuda.synchronize(dev)
    print(f"loaded in {time.time() - t0:.0f} s, {torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB on {dev}; "
          f"{len(cfg.applications)} layer applications, MXFP8 layers: {[i for i, e in enumerate(EX) if not e.nvfp4]}")

    files = sorted(f for f in glob.glob(os.path.join(args.ref, "*.npz")) if "grouped_mm" not in f and not os.path.basename(f).startswith("sub_"))
    for f in files:
        name = os.path.basename(f)[:-4]
        if args.only and name not in args.only:
            continue
        d = np.load(f)
        ids = torch.from_numpy(d["ids"]).to(dev)
        pos0 = int(d["pos0"]) if "pos0" in d else 0
        ref_apps = {int(a): n for n, a in enumerate(d["layer_apps"])}
        t0 = time.time()
        hidden = I2.embed_tokens(embed, embed_norm, ids, cfg.eps, adt)
        e_embed = rel(hidden, torch.from_numpy(d["embed"]).to(dev))
        blocks = None
        state = I2.LoopState()
        errs, agree = [], []
        for n, (i, p) in enumerate(cfg.applications):
            if args.teacher and n > 0 and (n - 1) in ref_apps:
                hidden = torch.from_numpy(d["layer_out"][ref_apps[n - 1]]).to(dev, adt)
            msgs = state.enter(cfg, W[i], hidden, blocks, p)
            trace = {}
            hidden, blocks = I2.layer_forward(W[i], EX[i], state.hidden, state.blocks, adt, None, pos0, msgs, trace)
            state.leave(cfg, W[i], p)
            if n in ref_apps:
                errs.append(rel(hidden, torch.from_numpy(d["layer_out"][ref_apps[n]]).to(dev)))
            r = torch.from_numpy(d["route"][n].astype(np.int64)).to(dev)
            agree.append((trace["route"].sort(-1).values == r).float().mean().item())
        fin = I2.final_hidden(Hd, hidden, blocks, adt)
        logits = I2.logits_from_hidden(Hd, fin)
        ref_tail = torch.from_numpy(d["logits_tail"]).to(dev)
        tail = logits[-ref_tail.shape[0]:, : ref_tail.shape[1]]
        am = (logits.argmax(-1).cpu().numpy() == d["argmax"]).mean()
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(tail.float(), -1), torch.log_softmax(ref_tail.float(), -1),
            log_target = True, reduction = "batchmean").item()
        e = np.array(errs)
        print(f"\n{name}: {len(ids)} tokens, pos0 {pos0}, {time.time() - t0:.1f} s")
        print(f"  embed err {e_embed:.2e}; layer output rel RMS err: first {e[0]:.4f}, max {e.max():.4f} "
              f"(application {int(e.argmax())}), last {e[-1]:.4f}")
        print(f"  routed ids agree: min {min(agree):.4f}, mean {np.mean(agree):.4f}")
        print(f"  final hidden err {rel(fin, torch.from_numpy(d['final']).to(dev)):.4f}; logits (last "
              f"{ref_tail.shape[0]}) err {rel(tail, ref_tail):.4f}, KL {kl:.2e}; argmax agree {am:.4f}; "
              f"logit shape ref {tuple(ref_tail.shape)} / port {tuple(logits.shape)}")
        if args.verbose:
            print("  per application: " + " ".join(f"{x:.3f}" for x in e))
            print("  route agree:     " + " ".join(f"{x:.3f}" for x in agree))


if __name__ == "__main__":
    main()
