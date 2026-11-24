"""
Assemble an Inkling-2 EXL3 model directory from quantized expert files of several bit rates, without copying:
every file is a hard link (the work directories and the output must be on one file system). convert_exl3.py's
quantize stage leaves one q_NN.safetensors per layer in a work directory per bit rate; this picks a work
directory per layer.

    python util/inkling2/assemble_variant.py <base model dir> <out dir> --work 4=<dir> 5=<dir> 8=<dir> \
        --bits 4 --layers 6-17=5 20-23=8

--bits is the default rate; --layers overrides it for layer ranges. The base model directory (an assembled
conversion) supplies everything that is not an expert file.
"""
import argparse, json, os, struct


def expert_bits(path):
    """(bits, weights) stored in a safetensors file of EXL3 experts, from its header"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    bits = sum(v["data_offsets"][1] - v["data_offsets"][0] for k, v in h.items() if k != "__metadata__") * 8
    return bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("out")
    ap.add_argument("--work", nargs = "+", required = True, help = "rate=work dir")
    ap.add_argument("--bits", required = True)
    ap.add_argument("--layers", nargs = "*", default = [], help = "a-b=rate")
    args = ap.parse_args()
    work = dict(w.split("=", 1) for w in args.work)
    cfg = json.load(open(os.path.join(args.base, "config.json")))
    tc = cfg["text_config"]
    n_layers = tc["num_hidden_layers"]
    rate = {i: args.bits for i in range(n_layers)}
    for spec in args.layers:
        rng, r = spec.split("=")
        a, b = (rng.split("-") + [rng])[:2]
        for i in range(int(a), int(b) + 1):
            rate[i] = r
    os.makedirs(args.out, exist_ok = True)
    for name in sorted(os.listdir(args.base)):
        if name.startswith("model-experts-") or name in ("config.json", "README.md"):
            continue
        dst = os.path.join(args.out, name)
        if not os.path.exists(dst):
            os.link(os.path.join(args.base, name), dst)
    total_bits = 0
    for i in range(n_layers):
        src = os.path.join(work[rate[i]], f"q_{i:02d}.safetensors")
        dst = os.path.join(args.out, f"model-experts-{i:02d}.safetensors")
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)
        total_bits += expert_bits(src)
    weights = n_layers * tc["n_routed_experts"] * 3 * tc["hidden_size"] * tc["intermediate_size"]
    avg = sum(float(r) for r in rate.values()) / n_layers
    q = dict(cfg.get("quantization_config") or {})
    q.update({"bits": round(avg, 2), "expert_bits": {str(i): float(rate[i]) for i in range(n_layers)}})
    q.pop("note", None)
    q["note"] = "routed experts only; all other tensors as in the source checkpoint; per-layer rates in expert_bits"
    cfg["quantization_config"] = q
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent = 2)
    size = sum(os.path.getsize(os.path.join(args.out, n)) for n in os.listdir(args.out) if n.endswith(".safetensors") and n != "mtp.safetensors")
    print(f"{args.out}: experts {avg:.2f} bpw nominal ({total_bits / weights:.2f} stored), "
          f"weight files {size / 2**30:.2f} GiB; rates " + " ".join(f"{i}:{rate[i]}" for i in range(n_layers)))


if __name__ == "__main__":
    main()
