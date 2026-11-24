"""
Step 3a check: incremental decode through ExLlamaV3's cache (paged K/V for the global layers, sliding buffers and
convolution rings in the recurrent state) against an uncached forward over the same tokens, in the same process.

    python util/inkling2/cache_check.py <model dir> [--tokens 3000] [--device cuda:0]

Tests
  chunks      prefill in uneven chunks, then token by token: logits against the uncached forward
  rollback    rewind the state in place and decode the same tokens again
  checkpoint  stash the state, restore it into another slot, continue from there
"""
import argparse, os, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LONG_TEXT = os.path.join(HERE, "..", "..", "eval", "eval_texts", "variable_man_mod.txt")


def rel(a, b):
    a, b = a.float(), b.float()
    return (torch.sqrt((a - b).square().mean()) / torch.sqrt(b.square().mean()).clamp_min(1e-12)).item()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", type = int, default = 3000)
    ap.add_argument("--device")
    ap.add_argument("--cache", type = int, default = 8192)
    ap.add_argument("--replay-routes", action = "store_true")
    args = ap.parse_args()

    from exllamav3 import Config, Model, Cache, Tokenizer
    if os.environ.get("EXL3_I2_CONV_FP32"):
        # Exact mode (with EXL3_I2_ADT=fp32): K/V rounded to fp16 in both passes, convolution ring in fp32
        from exllamav3.modules import inkling2 as I2
        I2.ROUND_KV = True
        print("exact mode: K/V rounded to fp16 in both passes, convolution ring in fp32")
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = args.cache, max_batch_size = 2)
    t0 = time.time()
    if args.device:
        model.load(device = args.device, progressbar = True)
    else:
        # Room for the logits of the whole sequence (the reference forward) next to the weights
        model.load(progressbar = True, reserve_per_device = [6.0, 1.0])
    print(f"loaded in {time.time() - t0:.0f} s; " + ", ".join(
        f"cuda:{d} {torch.cuda.memory_allocated(d) / 2**30:.1f} GiB" for d in range(torch.cuda.device_count())))
    rl = cache.get_all_recurrent_layers()
    print(f"cache: {len(cache.layers)} paged layers ({sum(l.storage_size() for l in cache.layers.values()) / 2**20:.0f} MiB "
          f"for {args.cache} tokens), {len(rl)} recurrent layers "
          f"({sum(l.storage_size() for l in rl.values()) / 2**20:.0f} MiB for 2 slots), "
          f"checkpoint {sum(l.get_checkpoint_size() for l in rl.values()) / 2**20:.0f} MiB")

    tokenizer = Tokenizer.from_config(config)
    with open(LONG_TEXT, encoding = "utf8") as f:
        ids = tokenizer.encode(f.read()[:args.tokens * 8])[:, :args.tokens]
    N = ids.shape[1]

    # Record the routing of the uncached pass and replay it in the cached ones (--replay-routes): top-k routing is
    # discontinuous, so passes that differ only by rounding noise otherwise drift apart by a few percent
    from exllamav3.modules import inkling2 as I2
    routes, call = [], [0]

    def record(W, pos0, sel):
        routes.append(sel)
        return sel

    def replay(W, pos0, sel):
        forced = routes[call[0]][pos0 : pos0 + sel.shape[0]].to(sel.device)
        call[0] += 1
        return forced

    if args.replay_routes:
        I2.ROUTE_HOOK = record
    t0 = time.time()
    full = model.forward(ids)[0].float().cpu()
    if args.replay_routes:
        I2.ROUTE_HOOK = replay
        print(f"replaying the routing of the uncached pass ({len(routes)} layer applications)")
    print(f"\nuncached forward over {N} tokens: {time.time() - t0:.1f} s")

    def step(tokens, past, states):
        params = {"attn_mode": "flash_attn", "cache": cache, "past_len": past, "batch_shape": (1, args.cache)}
        if states is not None:
            params["recurrent_states"] = states
        call[0] = 0
        out = model.forward(tokens, params)[0].float().cpu()
        return out, params["recurrent_states"]

    def report(name, got, a, b):
        ref = full[a:b]
        am = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
        kl = torch.nn.functional.kl_div(torch.log_softmax(got, -1), torch.log_softmax(ref, -1),
                                        log_target = True, reduction = "batchmean").item()
        print(f"  {name:44s} positions {a:5d}..{b:5d}: logits err {rel(got, ref):.2e}, KL {kl:.1e}, argmax agree {am:.4f}")

    # --- chunks: uneven prefill chunks, then single tokens. Sizes chosen to cross the sliding window, to fill the
    # sliding buffer exactly, to drop pages from it, and to exceed it in one chunk
    print("\nchunks")
    tail = 40
    sizes = [5, 300, 719, 256, 1, 1, 1400]
    plan, pos = [], 0
    for sz in sizes:
        if pos + sz > N - tail:
            break
        plan.append(sz)
        pos += sz
    if N - tail - pos > 0:
        plan.append(N - tail - pos)
    plan += [1] * tail
    states, pos, t0 = None, 0, time.time()
    singles = []
    for sz in plan:
        out, states = step(ids[:, pos : pos + sz], pos, states)
        if sz > 1:
            report(f"chunk of {sz} (window_beg {states[0].window_beg})", out, pos, pos + sz)
        else:
            singles.append((pos, out))
        pos += sz
    for a in sorted({p for p, _ in singles[:2]} | {singles[-tail][0]}):
        seq = [o for p, o in singles if p >= a]
        seq = seq[: tail if a == singles[-tail][0] else 1]
        report(f"{len(seq)} single token step(s)", torch.cat(seq), a, a + len(seq))
    print(f"  cached pass: {time.time() - t0:.1f} s; state at {states[0].position}, window_beg {states[0].window_beg}, "
          f"rollback capacity {states[0].rollback_capacity()}")

    # --- rollback: rewind in place, decode the same tokens again
    print("\nrollback")
    rs = states[0]
    for back in (1, 7, 200):
        back = min(back, rs.rollback_capacity())
        rs.rewind(back)
        a = rs.position
        out, states = step(ids[:, a : a + back], a, states)
        report(f"rewind {back}, one chunk", out, a, a + back)
    rs.rewind(20)
    a = rs.position
    outs = []
    for j in range(20):
        out, states = step(ids[:, a + j : a + j + 1], a + j, states)
        outs.append(out)
    report("rewind 20, token by token", torch.cat(outs), a, a + 20)

    # --- checkpoint: stash at a page boundary, restore into another slot, continue. The paged K/V of the global
    # layers stays where it is (same pages); only the recurrent part moves
    print("\ncheckpoint")
    boundary = rs.position // 256 * 256
    if rs.can_stash_at(boundary):
        rs.rewind(rs.position - boundary)
        stashed = rs.stash()
        size = sum(t.numel() * t.element_size() for v in stashed.values() if isinstance(v, tuple) for t in v)
        rs2 = cache.new_from_stashed(stashed, boundary)
        print(f"  stashed at {boundary}: {size / 2**20:.0f} MiB, restored into slot {rs2.slot} (was {rs.slot})")
        out, st2 = step(ids[:, boundary:N], boundary, [rs2])
        report("continue from the restored state", out, boundary, N)
        outs, st2[0] = [], st2[0]
        st2[0].rewind(min(8, st2[0].rollback_capacity()))
        a = st2[0].position
        for j in range(N - a):
            out, st2 = step(ids[:, a + j : a + j + 1], a + j, st2)
            outs.append(out)
        report("restored state, rewind, token by token", torch.cat(outs), a, N)
    else:
        print(f"  cannot stash at {boundary} (position {rs.position}, capacity {rs.rollback_capacity()})")


if __name__ == "__main__":
    main()
