"""
Ground truth for the Inkling-2-Mini port: runs the transformers reference (inkling2_loop) and saves, per prompt,
the token ids, the output of every decoder layer application (48 per token: 6 pre-loop, 3 x 12 loop, 6 post-loop,
in execution order), the routed expert ids of every application, the final hidden state and the logits.

Needs a current transformers (inkling2_loop), so run it in its own venv:

    python util/inkling2/ref_dump.py <model dir> <out dir>

One .npz per prompt:
    ids            int64 [S]
    pos0           int64               position of the first token (0 except for the shifted-position prompt)
    embed          f16   [S, D]        normed embedding (input of layer 0)
    layer_out      f16   [A, S, D]     output of each layer application, A = 48 (full prompts) or a subset (long prompts)
    layer_apps     int32 [A]           which applications are stored
    route          int16 [48, S, k]    expert ids per application (sorted per row)
    final          f16   [S, D]        after the output attention residual and the final norm
    logits_tail    f32   [T, V]        logits of the last T positions (softcapped, padded vocab cut by the model)
    argmax         int64 [S]           greedy next token at every position
    gen            int64 [G]           greedy continuation (cached decode), only for prompts with gen > 0
"""
import argparse, json, os, sys, time

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.models.inkling2_loop import modeling_inkling2_loop as M

LONG_TEXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval", "eval_texts", "variable_man_mod.txt")
TAIL = 16
# Applications stored for long prompts: first, the last pre-loop, the end of each pass, the last
LONG_APPS = [0, 5, 17, 29, 41, 47]


def prompts(tok):
    chat = lambda msgs, **kw: tok.apply_chat_template(msgs, tokenize = False, add_generation_prompt = True, **kw)
    with open(LONG_TEXT, encoding = "utf8") as f:
        long_ids = tok(f.read(), add_special_tokens = False)["input_ids"]
    text = lambda s: tok(s, add_special_tokens = False)["input_ids"]
    out = [
        dict(name = "short_raw", ids = text("The capital of France is Paris. The capital of Japan is"), gen = 0),
        dict(name = "chat_math", ids = text(chat([{"role": "user", "content": "What is 84 * 3 / 2?"}])), gen = 48),
        dict(name = "chat_code", ids = text(chat([
            {"role": "system", "content": "You are a careful programmer."},
            {"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number."},
        ], reasoning_effort = "low")), gen = 48),
        # Past the 1024-token sliding window and the 1024-distance relative bias
        dict(name = "text_1500", ids = long_ids[:1500], gen = 0),
        # Past log_scaling_n_floor (8192): the position scale of the global layers is active. A real 9000-token
        # pass does not fit beside the weights (the reference builds dense [heads, S, S] logits), so a short
        # text is run at shifted positions instead; relative distances do not depend on the shift
        dict(name = "text_300_pos20000", ids = long_ids[2000:2300], gen = 0, pos0 = 20000),
    ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--only", nargs = "*")
    ap.add_argument("--gpu-gib", type = float, nargs = 2, default = [21.0, 21.0])
    ap.add_argument("--cpu-gib", type = float, default = 44.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok = True)

    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype = torch.bfloat16,
        device_map = "auto",
        # The plain per-expert loop of the reference. The default (torch grouped_mm) died with a CUDA illegal
        # memory access on the second prompt (torch 2.11.0+cu128, experts spread over two GPUs and RAM)
        experts_implementation = "eager",
        max_memory = {0: f"{args.gpu_gib[0]}GiB", 1: f"{args.gpu_gib[1]}GiB", "cpu": f"{args.cpu_gib}GiB"},
    )
    model.eval()
    print(f"loaded in {time.time() - t0:.0f} s", flush = True)

    text_model = next(m for m in model.modules() if isinstance(m, M.Inkling2LoopTextModel))
    cfg = text_model.config
    n_apps = cfg.num_hidden_layers + (cfg.loop_num_passes - 1) * (cfg.loop_layer_index_end - cfg.loop_layer_index_start)
    meta = dict(
        n_apps = n_apps, hidden = cfg.hidden_size, loop_start = cfg.loop_layer_index_start,
        loop_end = cfg.loop_layer_index_end, passes = cfg.loop_num_passes, layer_types = list(cfg.layer_types),
        transformers = __import__("transformers").__version__, torch = torch.__version__,
        device_map = {k: str(v) for k, v in getattr(model, "hf_device_map", {}).items()},
    )
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent = 1)

    rec = dict(layers = [], routes = [], embed = None, final = None, keep = None)

    def layer_hook(mod, inp, out):
        a = len(rec["layers"])
        if rec["keep"] is None or a in rec["keep"]:
            rec["layers"].append(out[0].detach().to("cpu", torch.float16))
        else:
            rec["layers"].append(None)

    def route_hook(mod, inp, out):
        rec["routes"].append(out[2].detach().sort(dim = -1).values.to("cpu", torch.int16))

    hooks = []
    for layer in text_model.layers:
        hooks.append(layer.register_forward_hook(layer_hook))
        hooks.append(layer.mlp.gate.register_forward_hook(route_hook))
    hooks.append(text_model.embed_tokens.register_forward_hook(
        lambda m, i, o: rec.__setitem__("embed", o[0].detach().to("cpu", torch.float16))))
    hooks.append(text_model.norm.register_forward_hook(
        lambda m, i, o: rec.__setitem__("final", o[0].detach().to("cpu", torch.float16))))

    dev = text_model.embed_tokens.weight.device
    for p in prompts(tok):
        if args.only and p["name"] not in args.only:
            continue
        ids = torch.tensor([p["ids"]], dtype = torch.long, device = dev)
        rec.update(layers = [], routes = [], embed = None, final = None, keep = LONG_APPS if p.get("long") else None)
        t0 = time.time()
        with torch.inference_mode():
            kw = {}
            if p.get("pos0"):
                kw["position_ids"] = (torch.arange(ids.shape[1], device = dev) + p["pos0"]).unsqueeze(0)
            logits = model(input_ids = ids, use_cache = False, **kw).logits[0].float()
        assert len(rec["layers"]) == n_apps and len(rec["routes"]) == n_apps, (len(rec["layers"]), len(rec["routes"]))
        apps = [a for a, t in enumerate(rec["layers"]) if t is not None]
        save = dict(
            ids = np.array(p["ids"], dtype = np.int64),
            pos0 = np.int64(p.get("pos0", 0)),
            embed = rec["embed"].numpy(),
            layer_out = torch.stack([rec["layers"][a] for a in apps]).numpy(),
            layer_apps = np.array(apps, dtype = np.int32),
            route = torch.stack(rec["routes"]).numpy(),
            final = rec["final"].numpy(),
            logits_tail = logits[-TAIL:].cpu().numpy(),
            argmax = logits.argmax(-1).cpu().numpy(),
        )
        msg = f"{p['name']}: {ids.shape[1]} tokens, forward {time.time() - t0:.1f} s, next: {tok.decode(save['argmax'][-1:])!r}"

        if p["gen"]:
            # Greedy continuation through the cache: checks the incremental path of the reference against its own
            # full pass, and gives the port a decode target
            for h in hooks:
                h.remove()
            hooks.clear()
            with torch.inference_mode():
                g = model.generate(input_ids = ids, max_new_tokens = p["gen"], do_sample = False)
            gen = g[0, ids.shape[1]:].cpu()
            save["gen"] = gen.numpy()
            with torch.inference_mode():
                full = model(input_ids = g[:, :-1], use_cache = False).logits[0, ids.shape[1] - 1:].argmax(-1).cpu()
            n = min(len(full), len(gen))
            same = int((full[:n] == gen[:n]).sum())
            msg += f"\n  gen: {tok.decode(gen)!r}\n  cached decode vs full pass: {same}/{n} tokens equal"
            for layer in text_model.layers:
                hooks.append(layer.register_forward_hook(layer_hook))
                hooks.append(layer.mlp.gate.register_forward_hook(route_hook))
            hooks.append(text_model.embed_tokens.register_forward_hook(
                lambda m, i, o: rec.__setitem__("embed", o[0].detach().to("cpu", torch.float16))))
            hooks.append(text_model.norm.register_forward_hook(
                lambda m, i, o: rec.__setitem__("final", o[0].detach().to("cpu", torch.float16))))

        np.savez(os.path.join(args.out, p["name"] + ".npz"), **save)
        print(msg, flush = True)
        del logits
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
