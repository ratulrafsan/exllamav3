"""
CUDA-graph decode step against the op-by-op step, from the same state: each test position is decoded op by op, the
state is rewound one token, and the same token is decoded through the graph. Then speed over a run of steps.

    python util/inkling2/graph_check.py <EXL3 model dir> [--context 2000]
"""
import argparse, os, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def rel(a, b):
    a, b = a.float(), b.float()
    return (torch.sqrt((a - b).square().mean()) / torch.sqrt(b.square().mean()).clamp_min(1e-12)).item()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--context", type = int, nargs = "*", default = [2000])
    ap.add_argument("--steps", type = int, default = 60)
    ap.add_argument("--device")
    args = ap.parse_args()
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.architecture import inkling2_loop as A
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    size = (max(args.context) + args.steps + 1100 + 255) // 256 * 256
    cache = Cache(model, max_num_tokens = size, max_batch_size = 1)
    if args.device:
        model.load(device = args.device)
    else:
        model.load(reserve_per_device = [3.0, 1.0])
    print("devices:", sorted({str(m.device) for m in model.modules}), flush = True)
    tok = Tokenizer.from_config(config)
    with open(os.path.join(HERE, "..", "..", "eval", "eval_texts", "variable_man_mod.txt"), encoding = "utf8") as f:
        text = f.read()
    ids = tok.encode(text)
    while ids.shape[1] < size:
        ids = torch.cat([ids, ids], dim = 1)

    def step(t, past, st):
        p = {"attn_mode": "flash_attn", "cache": cache, "past_len": past, "batch_shape": (1, size), "last_tokens_only": 1}
        if st is not None:
            p["recurrent_states"] = st
        return model.forward(t, p)[0].float(), p["recurrent_states"]

    st, pos = None, 0
    for ctx in sorted(args.context):
        while pos < ctx:
            n = min(2048, ctx - pos)
            A._GRAPH = False
            _, st = step(ids[:, pos : pos + n], pos, st)
            pos += n
        # agreement, step by step from the same state
        errs, agree, kls = [], 0, []
        n_cmp = 12
        for j in range(n_cmp):
            A._GRAPH = False
            a, st = step(ids[:, pos : pos + 1], pos, st)
            st[0].rewind(1)
            A._GRAPH = True
            b, st = step(ids[:, pos : pos + 1], pos, st)
            errs.append(rel(b, a))
            agree += int(a.argmax(-1).item() == b.argmax(-1).item())
            kls.append(torch.nn.functional.kl_div(torch.log_softmax(b, -1), torch.log_softmax(a, -1),
                                                  log_target = True, reduction = "batchmean").item())
            pos += 1
        used = bool(model._graphs) and not model._graph_failed
        print(f"context {ctx}: graph {'used' if used else 'NOT used'}; graph vs op-by-op over {n_cmp} steps: "
              f"logits err median {sorted(errs)[n_cmp // 2]:.2e}, max {max(errs):.2e}; KL max {max(kls):.1e}; "
              f"argmax agree {agree}/{n_cmp}", flush = True)
        # speed
        res = {}
        for mode in (False, True, False, True):
            A._GRAPH = mode
            _, st = step(ids[:, pos : pos + 1], pos, st); pos += 1
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for j in range(args.steps // 4):
                _, st = step(ids[:, pos : pos + 1], pos, st); pos += 1
            torch.cuda.synchronize()
            res.setdefault(mode, []).append((time.perf_counter() - t0) / (args.steps // 4) * 1e3)
        f = lambda v: sum(v) / len(v)
        print(f"  op by op {f(res[False]):.1f} ms/token ({1e3 / f(res[False]):.1f} T/s), "
              f"graph {f(res[True]):.1f} ms/token ({1e3 / f(res[True]):.1f} T/s)", flush = True)


if __name__ == "__main__":
    main()
