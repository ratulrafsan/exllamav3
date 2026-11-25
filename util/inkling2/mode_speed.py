"""
Prefill and decode speed of the model as the environment configures it (LoopCD: EXL3_I2_LOOPCD, control vector:
EXL3_I2_CVEC), to compare a setting against the plain model: one process per setting, same text, same cache.

Prefill: chunks of 2,048 tokens up to --context, timed from the second chunk on. Decode: single-token steps as
CUDA graph replays after the context, the median of --reps runs of --steps tokens.

    python util/inkling2/mode_speed.py <EXL3 model dir> [--context 8192]
        [--cache 16384] [--steps 100] [--reps 5]
"""
import argparse, os, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--context", type = int, default = 8192)
    ap.add_argument("--cache", type = int, default = 16384)
    ap.add_argument("--steps", type = int, default = 100)
    ap.add_argument("--reps", type = int, default = 5)
    ap.add_argument("--device")
    args = ap.parse_args()
    from exllamav3 import Config, Model, Cache, Tokenizer
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = args.cache, max_batch_size = 1)
    if args.device:
        model.load(device = args.device)
    else:
        model.load(reserve_per_device = [3.0, 1.0])
    tok = Tokenizer.from_config(config)
    with open(os.path.join(HERE, "..", "..", "eval", "eval_texts", "variable_man_mod.txt"), encoding = "utf8") as f:
        ids = tok.encode(f.read())
    while ids.shape[1] < args.cache:
        ids = torch.cat([ids, ids], dim = 1)
    sync = lambda: [torch.cuda.synchronize(d) for d in model.active_devices_cuda()]

    def step(t, past, st, last = True):
        p = {"attn_mode": "flash_attn", "cache": cache, "past_len": past, "batch_shape": (1, args.cache)}
        if last:
            p["last_tokens_only"] = 1
        if st is not None:
            p["recurrent_states"] = st
        model.forward(t, p)
        return p["recurrent_states"]

    st, pos, chunks = None, 0, []
    while pos < args.context:
        n = min(2048, args.context - pos)
        sync(); t0 = time.perf_counter()
        st = step(ids[:, pos : pos + n], pos, st)
        sync()
        chunks.append((n, time.perf_counter() - t0))
        pos += n
    timed = chunks[1:] or chunks
    prefill = sum(n for n, _ in timed) / sum(t for _, t in timed)

    for _ in range(12):                                     # graph build and warm-up
        st = step(ids[:, pos : pos + 1], pos, st); pos += 1
    runs = []
    for _ in range(args.reps):
        sync(); t0 = time.perf_counter()
        for _ in range(args.steps):
            st = step(ids[:, pos : pos + 1], pos, st); pos += 1
        sync()
        runs.append((time.perf_counter() - t0) / args.steps * 1e3)
    ms = sorted(runs)[len(runs) // 2]
    devs = sorted({str(m.device) for m in model.modules})
    mem = [f"{torch.cuda.max_memory_allocated(d) / 2 ** 30:.2f}" for d in model.active_devices_cuda()]
    print(f"LoopCD {config.loopcd}; control vector {'on, scale %g' % config.cvec.scale if config.cvec else 'off'}; "
          f"{len(model.fwd_modules) - 2} layer applications on {devs}", flush = True)
    print(f"  prefill {prefill:7.0f} T/s (chunks of 2048 to {args.context});  decode {ms:6.2f} ms/token = {1e3 / ms:5.1f} T/s "
          f"at {pos} tokens (runs: {' '.join(f'{r:.2f}' for r in runs)});  graph steps {model.graph_steps}, "
          f"peak GiB {' / '.join(mem)}", flush = True)


if __name__ == "__main__":
    main()
