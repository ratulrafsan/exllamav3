"""
Control vector on Inkling-2-Mini (EXL3_I2_CVEC, architecture/inkling2_loop.py): correctness of the hook.

    1  scale 0 returns the plain model's logits, bit for bit (uncached, cached op by op, CUDA graph)
    2  at scale 1 (projection) every steered layer application's output has no component left along its
       direction, and at the file's scale the component shrinks by exactly that factor (the reader's trace)
    3  the state the next layer application receives is the steered one: the hidden state tapped at the input
       of application n + 1 has no component along direction n at scale 1, and has one in the plain model
    4  with the vector on, CUDA graph replays agree with op-by-op steps, and the logits differ from the plain
       model's

    python util/inkling2/cvec_check.py <EXL3 model dir> <vector.safetensors>
        [--device cuda:0] [--rows 128] [--context 1200] [--steps 24]
"""
import argparse, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FAILED = []


def check(name, ok, detail):
    if not ok:
        FAILED.append(name)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {detail}", flush = True)


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).square().mean().sqrt() / b.square().mean().sqrt()).item()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("vector")
    ap.add_argument("--device")
    ap.add_argument("--rows", type = int, default = 128)
    ap.add_argument("--context", type = int, default = 1200)
    ap.add_argument("--steps", type = int, default = 24)
    args = ap.parse_args()
    os.environ["EXL3_I2_CVEC"] = args.vector
    os.environ.pop("EXL3_I2_CVEC_SCALE", None)
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.architecture import inkling2_loop as A

    size = (args.context + args.steps + 600) // 256 * 256
    config = Config.from_directory(args.model)
    cv = config.cvec
    file_scale = cv.scale
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = size, max_batch_size = 2)
    if args.device:
        model.load(device = args.device)
    else:
        model.load(reserve_per_device = [3.0, 1.0])
    tok = Tokenizer.from_config(config)
    with open(os.path.join(HERE, "..", "..", "eval", "eval_texts", "variable_man_mod.txt"), encoding = "utf8") as f:
        ids = tok.encode(f.read())[:, : size]
    n_apps = len(config.i2.applications)

    def uncached(taps = False):
        p = {"export_state_layers": set(range(n_apps))} if taps else {}
        z = model.forward(ids[:, : args.rows], p)[0].float().cpu()
        return (z, [t[0].float().cpu() for t in p["export_states"]]) if taps else z

    def cached(graph):
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
        out = []
        before = model.graph_steps
        for j in range(args.steps):
            p = dict(base, past_len = pos + j, recurrent_states = st)
            out.append(model.forward(ids[:, pos + j : pos + j + 1], p)[0, -1].float().cpu())
            st = p["recurrent_states"]
        assert (model.graph_steps > before) == graph
        st[0].free()
        A._GRAPH = True
        return torch.stack(out)

    # The plain model: the vector unhooked
    config.cvec = None
    plain_u, plain_taps = uncached(True)
    plain_c, plain_g = cached(False), cached(True)
    config.cvec = cv
    for d in model.active_devices_cuda():
        torch.cuda.synchronize(d)
    model._graphs.clear()

    print(f"-- {cv}")
    cv.set_scale(0.0)
    check("scale 0, uncached vs plain", rel(uncached(), plain_u) == 0, "identical" )
    check("scale 0, cached op by op vs plain", rel(cached(False), plain_c) == 0, "identical")
    check("scale 0, graph vs plain graph", rel(cached(True), plain_g) == 0, "identical")

    def trace_run(scale):
        cv.set_scale(scale)
        A.ControlVector.TRACE = []
        out = uncached(True)
        tr, A.ControlVector.TRACE = A.ControlVector.TRACE, None
        return out, tr

    (z1, taps1), tr = trace_run(1.0)
    check("every layer application is steered once, in order", [a for a, _, _ in tr] == sorted(cv.vecs),
          f"{len(tr)} applications")
    before = torch.stack([b.abs().mean() for _, b, _ in tr])
    after = torch.stack([a.abs().mean() for _, _, a in tr])
    check("scale 1 removes the component along the direction", bool((after < 2e-2 * before).all()),
          f"mean |h.d| {before.mean():.3f} -> {after.mean():.5f} (worst ratio {(after / before).max():.1e}; fp16 rounding of h)")
    dirs = cv.tensors("cpu")[0]
    carried = [a for a in sorted(cv.vecs) if a + 1 < n_apps and config.i2.applications[a + 1][0] != config.i2.loop_start]
    lin = lambda taps: torch.stack([(taps[a + 1] @ dirs[a]).abs().mean() for a in carried])
    p_in, s_in = lin(plain_taps), lin(taps1)
    check("the next application's input is the steered state", bool((s_in < 2e-2 * p_in).all()),
          f"{len(carried)} applications whose output is the next one's input: mean |h.d| plain {p_in.mean():.3f}, "
          f"steered {s_in.mean():.5f}")

    (zf, _), tr = trace_run(file_scale)
    ratio = torch.stack([(a / b)[b.abs() > 0.05 * b.abs().mean()].mean() for _, b, a in tr])
    check(f"scale {file_scale:g} leaves {1 - file_scale:g} of the component", bool(((ratio - (1 - file_scale)).abs() < 2e-2).all()),
          f"h.d after / before: {ratio.min():.4f} .. {ratio.max():.4f}")
    d = rel(zf, plain_u)
    agree = (zf.argmax(-1) == plain_u.argmax(-1)).float().mean().item()
    check(f"scale {file_scale:g} changes the logits", d > 1e-3, f"rel. RMS difference {d:.3f}, argmax agreement with plain {100 * agree:.1f}%")
    c, g = cached(False), cached(True)
    e = rel(g, c)
    check(f"scale {file_scale:g}, graph vs op by op", e < 3e-2,
          f"rel. RMS error {e:.2e}, argmax agreement {100 * (g.argmax(-1) == c.argmax(-1)).float().mean():.1f}% "
          f"(plain model: {rel(plain_g, plain_c):.2e})")

    print("FAILED: " + "; ".join(FAILED) if FAILED else "all checks passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
