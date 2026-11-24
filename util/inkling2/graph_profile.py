"""
Which kernels a CUDA-graph decode step consists of, and how long each runs on the GPU: the step is run once more
op by op with the same code path (the graph's capture function) under the PyTorch profiler, and the kernel
durations are summed by kernel name and device. Also the replay time of each GPU segment alone.

    python util/inkling2/graph_profile.py <EXL3 model dir> [--context 2000] [--width 1]
        [--cache 163840 --chunk 4096 --reserve 3.0 1.0]
"""
import argparse, os, time
from collections import defaultdict
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--context", type = int, default = 2000)
    ap.add_argument("--width", type = int, default = 1); ap.add_argument("--top", type = int, default = 28)
    ap.add_argument("--cache", type = int, help = "cache size in tokens (default: just above the context); with "
                    "tabbyAPI's size the GPU split is tabbyAPI's")
    ap.add_argument("--reserve", type = float, nargs = 2, default = [3.0, 1.0], help = "GiB kept free per GPU")
    ap.add_argument("--chunk", type = int, default = 2048, help = "max_chunk_size of the load (tabbyAPI: chunk_size)")
    args = ap.parse_args()
    from exllamav3 import Config, Model, Cache, Tokenizer
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    size = (args.context + 1400) // 256 * 256
    cache = Cache(model, max_num_tokens = max(size, args.cache or 0), max_batch_size = 1)
    model.load(reserve_per_device = args.reserve, max_chunk_size = args.chunk)
    devs = [str(m.device) for m in model.modules]
    print("modules per device: " + ", ".join(f"{d} {devs.count(d)}" for d in sorted(set(devs))) + "; allocated GiB: " +
          ", ".join(f"{torch.cuda.memory_allocated(d) / 2**30:.2f}" for d in range(torch.cuda.device_count())))
    tok = Tokenizer.from_config(config)
    with open(os.path.join(HERE, "..", "..", "eval", "eval_texts", "variable_man_mod.txt"), encoding = "utf8") as f:
        ids = tok.encode(f.read())
    while ids.shape[1] < size:
        ids = torch.cat([ids, ids], dim = 1)

    def step(t, past, st):
        p = {"attn_mode": "flash_attn", "cache": cache, "past_len": past, "batch_shape": (1, size), "last_tokens_only": 1}
        if st is not None:
            p["recurrent_states"] = st
        model.forward(t, p)
        return p["recurrent_states"]

    st, pos, W = None, 0, args.width
    while pos < args.context:
        n = min(2048, args.context - pos)
        st = step(ids[:, pos : pos + n], pos, st); pos += n
    for _ in range(20):
        st = step(ids[:, pos : pos + W], pos, st); pos += W
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(50):
        st = step(ids[:, pos : pos + W], pos, st); pos += W
    torch.cuda.synchronize()
    print(f"graph step of {W} token(s) at context {args.context}: {(time.perf_counter() - t0) / 50 * 1e3:.2f} ms")
    g = model._graphs[W]
    for seg in g.segments:
        dev = seg["device"]
        with torch.cuda.device(dev):
            seg["graph"].replay(); torch.cuda.synchronize(dev); t0 = time.perf_counter()
            for _ in range(50):
                seg["graph"].replay()
            torch.cuda.synchronize(dev)
        print(f"  segment on {dev}: {len(seg['apps'])} modules, {(time.perf_counter() - t0) / 50 * 1e3:.2f} ms per replay")

    from torch.profiler import profile, ProfilerActivity
    g._run(False)
    with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        g._run(False)
        torch.cuda.synchronize()
    tot = defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            k = (e.name[:70], e.device_index)
            tot[k][0] += 1; tot[k][1] += e.device_time
    rows = sorted(tot.items(), key = lambda kv: -kv[1][1])
    n_all = sum(v[0] for v in tot.values()); t_all = sum(v[1] for v in tot.values())
    print(f"kernels and copies in one step: {n_all}, GPU time {t_all / 1e3:.2f} ms (run op by op: cold caches, so longer than a replay)")
    for (name, dev), (n, t) in rows[: args.top]:
        print(f"  cuda:{dev} {n:5d} x {t / n:7.1f} us = {t / 1e3:6.2f} ms  {name}")


if __name__ == "__main__":
    main()
