"""
Where a SenseNova Loop decode step and a prefill chunk spend their time: per-section wall time with a device
sync after each section, next to the unsynced total.

    python util/sensenova_loop/profile.py <model dir> [--context 2000] [--steps 40]
        [--chunk 2048] [--device cuda:0]
"""
import argparse, os, time
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--context", type = int, default = 2000)
    ap.add_argument("--steps", type = int, default = 40)
    ap.add_argument("--chunk", type = int, default = 2048)
    ap.add_argument("--device", default = "cuda:0")
    ap.add_argument("--fp32-state", action = "store_true")
    args = ap.parse_args()
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.architecture import sensenova_loop as SL
    from exllamav3.modules.gated_delta_net import GatedDeltaNet
    from exllamav3.modules.attn import Attention
    from exllamav3.modules.rmsnorm import RMSNorm
    from exllamav3.modules.loop_router import LoopRouter
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    size = (args.context + 2 * args.steps + args.chunk + 767) // 256 * 256
    cache = Cache(model, max_num_tokens = size, max_batch_size = 1,
                  recurrent_dtype = torch.float if args.fp32_state else torch.half)
    model.load(device = torch.device(args.device))
    tok = Tokenizer.from_config(config)
    with open(os.path.join(HERE, "..", "..", "eval", "eval_texts", "variable_man_mod.txt"), encoding = "utf8") as f:
        text = f.read()
    ids = tok.encode(text[: size * 8], add_bos = False)[:, :size]

    def step(t, past, st, last = True):
        p = {"attn_mode": "flash_attn", "cache": cache, "past_len": past, "batch_shape": (1, size)}
        if last:
            p["last_tokens_only"] = 1
        if st is not None:
            p["recurrent_states"] = st
        return model.forward(t, p), p["recurrent_states"]

    sync = lambda: torch.cuda.synchronize()
    st, pos = None, 0
    while pos < args.context:
        n = min(2048, args.context - pos)
        _, st = step(ids[:, pos: pos + n], pos, st)
        pos += n
    for _ in range(8):
        _, st = step(ids[:, pos: pos + 1], pos, st); pos += 1

    # Unsynced decode
    sync(); t0 = time.perf_counter()
    for _ in range(args.steps):
        _, st = step(ids[:, pos: pos + 1], pos, st); pos += 1
    sync(); plain = (time.perf_counter() - t0) / args.steps

    times, calls = defaultdict(float), defaultdict(int)
    depth = [0]

    def timed(name, fn):
        def wrapper(*a, **k):
            if depth[0]:
                return fn(*a, **k)
            depth[0] += 1
            sync(); t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                sync(); times[name] += time.perf_counter() - t0; calls[name] += 1
                depth[0] -= 1
        return wrapper

    mlp_cls = {type(m.mlp) for m in model.modules if getattr(m, "mlp", None) is not None}
    patches = [(GatedDeltaNet, "forward", "delta-net"), (Attention, "forward", "attention"),
               (RMSNorm, "forward", "norm"), (LoopRouter, "forward", "router")] + \
              [(c, "forward", "mlp") for c in mlp_cls]
    saved = [(c, a, getattr(c, a)) for c, a, _ in patches]
    for c, a, name in patches:
        setattr(c, a, timed(name, getattr(c, a)))
    saved.append((SL, "merge", SL.merge))
    SL.merge = timed("merge", SL.merge)
    head, embed = model.modules[-1], model.modules[0]
    head.forward, embed.forward = timed("head", head.forward), timed("embedding", embed.forward)

    def report(label, total, n, unsynced):
        print(f"\n{label}: {unsynced * 1e3:.2f} ms unsynced ({1 / unsynced:.1f} per second), "
              f"{total / n * 1e3:.2f} ms with syncs")
        acc = 0.
        for k, v in sorted(times.items(), key = lambda kv: -kv[1]):
            acc += v
            print(f"  {k:10s} {v / n * 1e3:7.3f} ms  {calls[k] // n:4d} calls  {v / calls[k] * 1e6:8.1f} us each  {v / total * 100:5.1f}%")
        print(f"  {'other':10s} {(total - acc) / n * 1e3:7.3f} ms  {(total - acc) / total * 100:5.1f}%")

    sync(); t0 = time.perf_counter()
    for _ in range(args.steps):
        _, st = step(ids[:, pos: pos + 1], pos, st); pos += 1
    sync(); total = time.perf_counter() - t0
    report(f"decode step at context {pos}", total, args.steps, plain)

    # Prefill chunk: once with section timers, once without
    times.clear(); calls.clear()
    sync(); t0 = time.perf_counter()
    _, st = step(ids[:, pos: pos + args.chunk // 2], pos, st); pos += args.chunk // 2
    sync(); total = time.perf_counter() - t0
    n_timed = args.chunk // 2
    for c, a, f in saved:
        setattr(c, a, f)
    del head.forward, embed.forward
    sync(); t0 = time.perf_counter()
    _, st = step(ids[:, pos: pos + args.chunk // 2], pos, st)
    sync(); un = time.perf_counter() - t0
    print(f"\n(prefill chunks of {n_timed} tokens: {n_timed / un:.0f} T/s unsynced)")
    report(f"prefill chunk of {n_timed} tokens", total, 1, un)


if __name__ == "__main__":
    main()
