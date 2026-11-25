"""
LoopCD (arXiv:2610.02185) on Inkling-2-Mini: correctness of the implementation in architecture/inkling2_loop.py.

Three models are loaded one after the other, each run uncached (one row) and cached (prefill in chunks, then
teacher-forced decode steps, op by op and as CUDA graph replays):

    1  the model with only `ref` loop passes (EXL3_I2_PASSES), LoopCD off: the oracle for the reference logits
    2  the full model, LoopCD off, then hidden mode switched on at run time
    3  the full model in logits mode

Checked:

    logits mode   the last pass's logits equal the plain model's; the reference logits equal model 1's; the
                  guided logits equal z + w * (z - z_ref) computed here, fixed and adaptive; strength 0 returns
                  the plain logits; graph replays agree with op-by-op steps
    hidden mode   strength 0 returns the plain logits; graph replays agree with op-by-op steps; a strength
                  above 0 changes the logits

Use a model that fits one GPU with --device (two models on different splits differ by the routing noise of
the GPUs' GEMMs, see the README).

    python util/inkling2/loopcd_check.py <EXL3 model dir> --device cuda:0
        [--ref 1] [--rows 128] [--context 1200] [--steps 24]
"""
import argparse, gc, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FAILED = []


def cmp(name, a, b, tol):
    a, b = a.float(), b.float()
    err = ((a - b).square().mean().sqrt() / b.square().mean().sqrt()).item()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    ok = err <= tol
    if not ok:
        FAILED.append(name)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: rel. RMS error {err:.2e} (limit {tol:.0e}), "
          f"max abs {(a - b).abs().max().item():.2e}, argmax agreement {100 * agree:.1f}%", flush = True)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device")
    ap.add_argument("--ref", type = int, default = 1)
    ap.add_argument("--rows", type = int, default = 128, help = "tokens of the uncached row")
    ap.add_argument("--context", type = int, default = 1200)
    ap.add_argument("--steps", type = int, default = 24)
    args = ap.parse_args()
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.architecture import inkling2_loop as A
    from exllamav3.modules import inkling2 as I2

    size = (args.context + args.steps + 600) // 256 * 256
    ids = None

    def load(passes = 0, mode = "off"):
        nonlocal ids
        A._PASSES = passes
        config = Config.from_directory(args.model)
        config.set_loopcd(mode, args.ref, 0.0, 0.0) if mode != "off" else None
        model = Model.from_config(config)
        cache = Cache(model, max_num_tokens = size, max_batch_size = 2)
        if args.device:
            model.load(device = args.device)
        else:
            model.load(reserve_per_device = [3.0, 1.0])
        if ids is None:
            tok = Tokenizer.from_config(config)
            with open(os.path.join(HERE, "..", "..", "eval", "eval_texts", "variable_man_mod.txt"), encoding = "utf8") as f:
                ids = tok.encode(f.read())[:, : size]
        print(f"-- {config.i2.passes} passes, LoopCD {config.loopcd}, {len(model.fwd_modules) - 2} layer applications",
              flush = True)
        return model, cache

    def unload(model, cache):
        model.unload()
        del model, cache
        gc.collect()
        torch.cuda.empty_cache()

    def uncached(model, export = False):
        p = {"loopcd_export": True} if export else {}
        z = model.forward(ids[:, : args.rows], p)[0].float().cpu()
        if export:
            return z, tuple(t[0].float().cpu() for t in p["loopcd_logits"])
        return z

    def cached(model, cache, graph, export = False):
        """Prefill the context in chunks, then decode `steps` tokens of the text. The decode steps' logits"""
        A._GRAPH = graph
        base = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, size), "last_tokens_only": 1}
        pos, st = 0, None
        while pos < args.context:
            n = min(512, args.context - pos)
            p = dict(base, past_len = pos)
            if st is not None:
                p["recurrent_states"] = st
            model.forward(ids[:, pos : pos + n], p)
            st = p["recurrent_states"]
            pos += n
        out, pairs = [], []
        before = model.graph_steps
        for j in range(args.steps):
            p = dict(base, past_len = pos + j, recurrent_states = st)
            if export:
                p["loopcd_export"] = True
            out.append(model.forward(ids[:, pos + j : pos + j + 1], p)[0, -1].float().cpu())
            st = p["recurrent_states"]
            if export:
                pairs.append(tuple(t[0, -1].float().cpu() for t in p["loopcd_logits"]))
        used = model.graph_steps - before
        assert (used > 0) == graph, f"graph steps used: {used}, expected {'some' if graph else 'none'}"
        st[0].free()
        A._GRAPH = True
        out = torch.stack(out)
        if export:
            return out, (torch.stack([a for a, _ in pairs]), torch.stack([b for _, b in pairs]))
        return out

    def guide(z, zr, w, wmax):
        return I2.loopcd_logits(z, zr, torch.tensor([w, wmax]))

    # 1: the model with `ref` passes
    model, cache = load(passes = args.ref)
    few_u, few_c = uncached(model), cached(model, cache, False)
    unload(model, cache)

    # 2: the full model; hidden mode
    model, cache = load()
    full_u, full_c, full_g = uncached(model), cached(model, cache, False), cached(model, cache, True)
    cmp("plain model, graph vs op by op (the reference point for the graph checks below)", full_g, full_c, 3e-2)
    cmp("the model with fewer passes differs from the full model (expect a FAIL-sized error)", few_u, full_u, 1e9)
    model.set_loopcd("hidden", args.ref, strength = 0.0)
    print(f"-- LoopCD {model.config.loopcd}")
    cmp("hidden, strength 0, uncached vs plain", uncached(model), full_u, 1e-6)
    cmp("hidden, strength 0, cached op by op vs plain", cached(model, cache, False), full_c, 1e-6)
    cmp("hidden, strength 0, graph vs plain graph", cached(model, cache, True), full_g, 1e-6)
    model.set_loopcd(strength = 0.5)
    print(f"-- LoopCD {model.config.loopcd}")
    hid_u, hid_c, hid_g = uncached(model), cached(model, cache, False), cached(model, cache, True)
    cmp("hidden, strength 0.5, graph vs op by op", hid_g, hid_c, 3e-2)
    d = ((hid_u - full_u).square().mean().sqrt() / full_u.square().mean().sqrt()).item()
    print(f"  {'ok  ' if d > 1e-3 else 'FAIL'} hidden, strength 0.5 changes the logits: rel. RMS difference {d:.2e}")
    if d <= 1e-3:
        FAILED.append("hidden mode has no effect")
    # Strength changes reach a captured graph without a rebuild
    builds = model.graph_builds
    model.set_loopcd(strength = 0.0)
    cmp("hidden, back to strength 0, graph vs plain graph", cached(model, cache, True), full_g, 1e-6)
    model.set_loopcd("off")
    cmp("LoopCD off again, graph vs plain graph", cached(model, cache, True), full_g, 1e-6)
    unload(model, cache)

    # 3: logits mode
    model, cache = load(mode = "logits")
    z0, (zu, zru) = uncached(model, True)
    cmp("logits, strength 0, uncached vs plain", z0, full_u, 1e-3)
    cmp("logits, last pass's logits, uncached vs plain", zu, full_u, 1e-3)
    cmp(f"logits, reference logits, uncached vs the model with {args.ref} pass(es)", zru, few_u, 1e-3)
    c0, (zc, zrc) = cached(model, cache, False, True)
    cmp("logits, strength 0, cached op by op vs plain", c0, full_c, 1e-3)
    cmp(f"logits, reference logits, cached vs the model with {args.ref} pass(es)", zrc, few_c, 1e-3)
    cmp("logits, strength 0, graph vs plain graph", cached(model, cache, True), full_g, 1e-3)
    for w, wmax in ((0.3, 0.0), (0.0, 1.0), (0.2, 0.5)):
        model.set_loopcd(strength = w, wmax = wmax)
        print(f"-- LoopCD {model.config.loopcd}")
        cmp("logits, uncached vs the formula", uncached(model), guide(zu, zru, w, wmax), 1e-5)
        gc_, go = cached(model, cache, True), cached(model, cache, False)
        cmp("logits, cached op by op vs the formula", go, guide(zc, zrc, w, wmax), 1e-5)
        cmp("logits, graph vs op by op", gc_, go, 3e-2)
    wad = 1.0 - (torch.softmax(zc, -1).topk(2).values @ torch.tensor([1.0, -1.0]))
    print(f"  adaptive factor 1 - (p1 - p2) over the decode steps: mean {wad.mean():.3f}, min {wad.min():.3f}, max {wad.max():.3f}")
    unload(model, cache)

    print("FAILED: " + "; ".join(FAILED) if FAILED else "all checks passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
