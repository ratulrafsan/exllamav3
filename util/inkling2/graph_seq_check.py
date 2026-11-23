"""
CUDA-graph decode against op-by-op decode over a run of steps (teacher-forced, same tokens): decode N tokens op
by op keeping the logits, rewind to the start in checkpoints, decode the same tokens through the graph. Unlike
graph_check.py this also tests the state each graph step leaves behind for the next ones.

    python util/inkling2/graph_seq_check.py <EXL3 model dir> [--context 2000] [--steps 700]
"""
import argparse, os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--context", type = int, default = 2000); ap.add_argument("--steps", type = int, default = 700)
    args = ap.parse_args()
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.architecture import inkling2_loop as A
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    size = (args.context + args.steps + 600) // 256 * 256
    cache = Cache(model, max_num_tokens = size, max_batch_size = 2)
    model.load(reserve_per_device = [3.0, 1.0])
    tok = Tokenizer.from_config(config)
    with open(os.path.join(HERE, "..", "..", "eval", "eval_texts", "variable_man_mod.txt"), encoding = "utf8") as f:
        ids = tok.encode(f.read())
    while ids.shape[1] < size:
        ids = torch.cat([ids, ids], 1)

    def step(t, past, st):
        p = {"attn_mode": "flash_attn", "cache": cache, "past_len": past, "batch_shape": (1, size), "last_tokens_only": 1}
        p["recurrent_states"] = st
        return model.forward(t, p)[0, -1].float().cpu(), p["recurrent_states"]

    A._GRAPH = False
    p = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, size), "last_tokens_only": 1}
    pos = 0
    st = None
    while pos < args.context:
        n = min(2048, args.context - pos)
        pp = dict(p, past_len = pos)
        if st is not None: pp["recurrent_states"] = st
        model.forward(ids[:, pos : pos + n], pp); st = pp["recurrent_states"]; pos += n
    start = st[0].stash()                       # the recurrent state at the start (paged K/V stays in the cache)
    wb = st[0].window_beg
    ref = []
    for j in range(args.steps):
        l, st = step(ids[:, pos + j : pos + j + 1], pos + j, st)
        ref.append(l)
    # back to the start: restore the stashed state into the other slot
    st[0].free()
    st = [cache.new_from_stashed(start, args.context)]
    A._GRAPH = True
    errs, agree = [], []
    for j in range(args.steps):
        l, st = step(ids[:, pos + j : pos + j + 1], pos + j, st)
        errs.append(((l - ref[j]).square().mean().sqrt() / ref[j].square().mean().sqrt()).item())
        agree.append(int(l.argmax().item() == ref[j].argmax().item()))
    print(f"context {args.context}, {args.steps} steps, graph steps used {model.graph_steps}, window_beg {wb} -> {st[0].window_beg}")
    for a in range(0, args.steps, 100):
        e = errs[a : a + 100]; g = agree[a : a + 100]
        print(f"  steps {a:4d}..{a + len(e):4d}: logits err median {sorted(e)[len(e) // 2]:.2e}, max {max(e):.2e}; argmax agree {sum(g)}/{len(g)}")

if __name__ == "__main__":
    main()
