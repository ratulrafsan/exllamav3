"""
Per-round trace of speculative decoding with the DSpark drafter: for each verified window size, how many rounds,
how many tokens they produced and what they cost (drafter, verification pass, the drafter's cache update), against
plain decoding of the same prompt. Shows where the drafter gains and where it loses.

    python util/inkling2/draft_trace.py <EXL3 model dir> <dspark dir>
        [--conf 0.4] [--tokens 400] [--only essay prose] [--no-pad] [--brief]
"""
import argparse, glob, os, time
from collections import defaultdict
import torch

PROMPTS = [
    ("math", "What is 84 * 3 / 2? Answer briefly."),
    ("code", "Write a Python function that returns the n-th Fibonacci number, with a docstring and type hints."),
    ("prose", "Explain in about 120 words why the sky is blue."),
    ("list", "List the planets of the solar system in order from the sun, one per line, with one fact each."),
    ("essay", "Write a 250-word essay about the history of the printing press."),
    ("code2", "Write a Python class for an LRU cache with get and put, using only the standard library. Include a short usage example."),
    ("quote", "Quote the first three paragraphs of the text above word for word, then stop."),
]
FMT = "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|><|message_user|><|content_text|>{}<|end_message|>"


def sync():
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("draft")
    ap.add_argument("--tokens", type = int, default = 400); ap.add_argument("--cache", type = int, default = 16384)
    ap.add_argument("--conf", type = float, default = 0.4)
    ap.add_argument("--only", nargs = "*")
    ap.add_argument("--context", type = int, default = 0, help = "tokens of unrelated text in front of each prompt")
    ap.add_argument("--no-pad", action = "store_true", help = "do not fill padded verification slots with drafts")
    ap.add_argument("--brief", action = "store_true", help = "one line per prompt")
    ap.add_argument("--rounds", action = "store_true", help = "print every round of the last run")
    args = ap.parse_args()
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator
    from exllamav3.generator.sampler import ArgmaxSampler
    config = Config.from_directory(args.model); model = Model.from_config(config)
    args.cache = max(args.cache, (args.context + 4096 + 255) // 256 * 256)
    cache = Cache(model, max_num_tokens = args.cache, max_batch_size = 1, max_history = 8)
    dconfig = Config.from_directory(args.draft); draft = Model.from_config(dconfig)
    dcache = Cache(draft, max_num_tokens = args.cache)
    draft.load(reserve_per_device = [3.0, 1.0])
    model.load(reserve_per_device = [3.0, 1.0])
    tok = Tokenizer.from_config(config)
    filler = ""
    if args.context:
        here = os.path.dirname(os.path.abspath(__file__))
        text = ""
        for f in sorted(glob.glob(os.path.join(here, "..", "..", "eval", "eval_texts", "*.txt"))):
            with open(f, encoding = "utf8") as fh:
                text += fh.read() + "\n\n"
        ids = tok.encode(text)
        while ids.shape[1] < args.context:
            ids = torch.cat([ids, ids], dim = 1)
        filler = tok.decode(ids[:, : args.context])
        filler = (filler[0] if isinstance(filler, list) else filler) + "\n\n"
    prompts = [p for p in PROMPTS if not args.only or p[0] in args.only]

    def run(gen, prompt):
        out, res = gen.generate(prompt = FMT.format(filler + prompt), max_new_tokens = args.tokens, sampler = ArgmaxSampler(),
                                stop_conditions = [config.eos_token_id], add_bos = False, encode_special_tokens = True,
                                decode_special_tokens = True, completion_only = True, return_last_results = True)
        return out, res

    plain_gen = Generator(model = model, cache = cache, tokenizer = tok)
    plain = {}
    for name, p in prompts * 2:
        out, res = run(plain_gen, p)
        plain[name] = (out, res["new_tokens"] / res["time_generate"])
    del plain_gen

    trace(args, config, model, cache, tok, draft, dcache, prompts, plain, run)


def trace(args, config, model, cache, tok, draft, dcache, prompts, plain, run):
    from exllamav3 import Generator
    print(f"padded verification slots {'unused' if args.no_pad else 'filled with drafts'}", flush = True)
    kw = dict(dynamic_draft_tokens = True, draft_confidence = args.conf) if args.conf > 0 else {}
    gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = draft, draft_cache = dcache,
                    **kw)
    if args.no_pad:
        gen.draft_pass_widths = None
    rounds = []
    cur = {}
    f_draft, f_iter, f_upd = gen.iterate_draftmodel_dflash_gen, gen.iterate_gen, draft.update_kv_from_target

    def w_draft(results, window):
        sync(); t0 = time.perf_counter()
        r = f_draft(results, window)
        sync(); cur["t_draft"] = time.perf_counter() - t0
        return r

    def w_upd(*a, **k):
        sync(); t0 = time.perf_counter()
        r = f_upd(*a, **k)
        sync(); cur["t_upd"] = cur.get("t_upd", 0.0) + time.perf_counter() - t0
        return r

    def w_iter(results, draft_tokens = None):
        jobs = [j for j in gen.active_jobs if j.is_prefill_done()]
        before = sum(j.new_tokens for j in jobs)
        sync(); t0 = time.perf_counter()
        r = f_iter(results, draft_tokens)
        sync(); dt = time.perf_counter() - t0
        if jobs:
            rounds.append(dict(window = 0 if draft_tokens is None else draft_tokens.shape[-1],
                               tokens = sum(j.new_tokens for j in jobs) - before, t_iter = dt,
                               t_draft = cur.get("t_draft", 0.0), t_upd = cur.get("t_upd", 0.0),
                               drafted = "t_draft" in cur))
        cur.clear()
        return r

    gen.iterate_draftmodel_dflash_gen, gen.iterate_gen, draft.update_kv_from_target = w_draft, w_iter, w_upd
    try:
        _trace_runs(args, gen, prompts, plain, run, rounds)
    finally:
        draft.update_kv_from_target = f_upd


def _trace_runs(args, gen, prompts, plain, run, rounds):
    for rnd in range(2):
        for name, p in prompts:
            rounds.clear()
            out, res = run(gen, p)
            if rnd == 0:
                continue                        # the first run builds the graphs
            ref, tps = plain[name]
            print(f"{name}: plain {tps:.1f} T/s, with drafter {res['new_tokens'] / res['time_generate']:.1f} T/s "
                  f"({res['new_tokens']} tokens, {len(rounds)} rounds; "
                  f"text {'equal' if out == ref else 'differs'})", flush = True)
            if args.brief:
                continue
            by = defaultdict(list)
            for r in rounds:
                by[r["window"]].append(r)
            for w in sorted(by):
                rs = by[w]
                n = len(rs)
                tk = sum(r["tokens"] for r in rs)
                ti = sum(r["t_iter"] for r in rs); td = sum(r["t_draft"] for r in rs); tu = sum(r["t_upd"] for r in rs)
                full = sum(1 for r in rs if r["tokens"] == w + 1)
                print(f"  window {w}: {n:4d} rounds, {tk / n:.2f} tokens per round ({100 * full / n:3.0f}% fully accepted); "
                      f"drafter {1e3 * td / n:5.2f} ms + pass {1e3 * (ti - tu) / n:5.2f} ms + cache update {1e3 * tu / n:5.2f} ms "
                      f"= {1e3 * (ti + td) / tk:5.2f} ms per token")
            tk = sum(r["tokens"] for r in rounds); tt = sum(r["t_iter"] + r["t_draft"] for r in rounds)
            print(f"  all: {1e3 * tt / tk:.2f} ms per token inside the rounds ({tk / tt:.1f} T/s); plain {1e3 / tps:.2f} ms")
            if args.rounds:
                print("  " + " ".join(f"{r['window']}:{r['tokens']}" for r in rounds))


if __name__ == "__main__":
    main()
