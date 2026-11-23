"""
Multi-token CUDA-graph steps (a decode token plus drafted tokens to verify) against the op-by-op forward.

Per trial, from the same state: [one token] [a pass of W tokens, as a draft verification] [rewind R of them]
[one token], once op by op and once through the graphs. Widths 2..8, many positions, so that passes straddle page
boundaries and the sliding buffers' page drops.

    python util/inkling2/graph_spec_check.py <EXL3 model dir> [--context 3000] [--trials 60]
"""
import argparse, os, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).square().mean().sqrt() / b.square().mean().sqrt()).item()

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--context", type = int, default = 3000); ap.add_argument("--trials", type = int, default = 60); ap.add_argument("--verbose", action = "store_true")
    args = ap.parse_args()
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.architecture import inkling2_loop as A
    config = Config.from_directory(args.model); model = Model.from_config(config)
    size = (args.context + args.trials * 12 + 1024) // 256 * 256
    cache = Cache(model, max_num_tokens = size, max_batch_size = 1, max_history = 64)
    model.load(reserve_per_device = [3.0, 1.0])
    tok = Tokenizer.from_config(config)
    with open(os.path.join(HERE, "..", "..", "eval", "eval_texts", "variable_man_mod.txt"), encoding = "utf8") as f:
        ids = tok.encode(f.read())
    while ids.shape[1] < size: ids = torch.cat([ids, ids], 1)

    def step(t, past, st, hist = False):
        p = {"attn_mode": "flash_attn", "cache": cache, "past_len": past, "batch_shape": (1, size)}
        if st is not None: p["recurrent_states"] = st
        if hist: p["recurrent_history"] = True
        return model.forward(t, p)[0].float().cpu(), p["recurrent_states"]

    A._GRAPH = False
    st, P = None, 0
    while P < args.context:
        n = min(2048, args.context - P)
        _, st = step(ids[:, P : P + n], P, st); P += n
    stats = {}
    widths = [2, 3, 4, 8, 5, 8, 2, 8]
    t_e = t_g = 0.0
    for trial in range(args.trials):
        W = widths[trial % len(widths)]
        R = (trial * 3) % W                      # drafted tokens rejected
        keep = W - R
        seqs = {}
        for mode in (False, True):
            A._GRAPH = mode
            a0, st = step(ids[:, P : P + 1], P, st)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            a1, st = step(ids[:, P + 1 : P + 1 + W], P + 1, st, True)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            if mode: t_g += dt
            else: t_e += dt
            st[0].rewind(R)
            a2, st = step(ids[:, P + 1 + keep : P + 2 + keep], P + 1 + keep, st)
            seqs[mode] = (a0, a1, a2)
            if not mode:
                st[0].rewind(2 + keep)
        e = seqs[False]; g = seqs[True]
        s = stats.setdefault(W, dict(n = 0, err = [], agree = 0, total = 0, after = []))
        s["n"] += 1
        s["err"].append(rel(g[1], e[1]))
        s["agree"] += int((g[1].argmax(-1) == e[1].argmax(-1)).sum()); s["total"] += W
        s["after"].append(rel(g[2], e[2]))
        if args.verbose or (g[1].argmax(-1) != e[1].argmax(-1)).any():
            print(f"    trial {trial}: width {W} at {P + 1}, reject {R}: pass err {rel(g[1], e[1]):.2e}, per token " +
                  " ".join(f"{rel(g[1][i], e[1][i]):.1e}" for i in range(W)) + f", window_beg {st[0].window_beg}, row {P + 1 - st[0].window_beg}", flush = True)
        P += 2 + keep
    print(f"context {args.context}, {args.trials} trials; graph forwards {model.graph_steps}; state at {st[0].position}, window_beg {st[0].window_beg}")
    for W in sorted(stats):
        s = stats[W]
        print(f"  width {W}: {s['n']} passes; pass logits err median {sorted(s['err'])[len(s['err']) // 2]:.2e}, max {max(s['err']):.2e}; "
              f"argmax agree {s['agree']}/{s['total']}; the token after the rewind: err max {max(s['after']):.2e}")
    print(f"  verification passes: op by op {t_e / args.trials * 1e3:.1f} ms, graph {t_g / args.trials * 1e3:.1f} ms on average")

    # Steady-state cost of a pass per width (the same pass repeated: rewind, run again)
    print("  steady state, ms per pass (op by op / graph) and ms per token through the graph:")
    for W in (1, 2, 4, 8):
        res = {}
        for mode in (False, True):
            A._GRAPH = mode
            for rep in range(14):
                if rep == 4:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                _, st = step(ids[:, P : P + W], P, st, W > 1)
                st[0].rewind(W)
            torch.cuda.synchronize(); res[mode] = (time.perf_counter() - t0) / 10 * 1e3
        print(f"    width {W}: {res[False]:6.1f} / {res[True]:6.1f} ms  ({res[True] / W:.1f} ms per token)", flush = True)


if __name__ == "__main__":
    main()
