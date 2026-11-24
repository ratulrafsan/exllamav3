"""
The DSpark drafter (Inkling-2-Mini's MTP head) through ExLlamaV3's generator: greedy generation with and without
it. Speculative decoding verifies every draft with the target, so the text should be the same (up to the model's
rounding sensitivity); reports speed and how many drafts are accepted.

    python util/inkling2/dspark_check.py <EXL3 model dir> <dspark dir>
"""
import argparse, os, time
import torch

PROMPTS = [
    ("math", "What is 84 * 3 / 2? Answer briefly."),
    ("code", "Write a Python function that returns the n-th Fibonacci number, with a docstring and type hints."),
    ("prose", "Explain in about 120 words why the sky is blue."),
    ("list", "List the planets of the solar system in order from the sun, one per line, with one fact each."),
]
FMT = "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|><|message_user|><|content_text|>{}<|end_message|>"

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("draft")
    ap.add_argument("--tokens", type = int, default = 300); ap.add_argument("--cache", type = int, default = 16384)
    ap.add_argument("--draft-device", default = "cuda:1")
    args = ap.parse_args()
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator
    from exllamav3.generator.sampler import ArgmaxSampler
    config = Config.from_directory(args.model); model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = args.cache, max_batch_size = 1, max_history = 8)
    dconfig = Config.from_directory(args.draft); draft = Model.from_config(dconfig)
    dcache = Cache(draft, max_num_tokens = args.cache)
    model.load(reserve_per_device = [3.0, 2.5])
    draft.load(device = args.draft_device)
    tok = Tokenizer.from_config(config)
    print("loaded: " + ", ".join(f"cuda:{d} {torch.cuda.memory_allocated(d) / 2**30:.1f} GiB" for d in range(torch.cuda.device_count())), flush = True)
    gens = {
        "plain": Generator(model = model, cache = cache, tokenizer = tok),
    }
    def run(gen, prompt):
        t0 = time.time()
        out, res = gen.generate(prompt = FMT.format(prompt), max_new_tokens = args.tokens, sampler = ArgmaxSampler(),
                                stop_conditions = [config.eos_token_id], add_bos = False, encode_special_tokens = True,
                                decode_special_tokens = True, completion_only = True, return_last_results = True)
        return out, res
    plain = {}
    for name, p in PROMPTS:
        out, res = run(gens["plain"], p)
        plain[name] = (out, res["new_tokens"] / res["time_generate"])
    del gens["plain"]
    gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = draft, draft_cache = dcache)
    for name, p in PROMPTS:
        g0 = model.graph_steps
        out, res = run(gen, p)
        ref, tps = plain[name]
        same = 0
        for a, b in zip(out, ref):
            if a != b: break
            same += 1
        acc = {k: res[k] for k in res if "draft" in k.lower() or "accept" in k.lower()}
        print(f"{name:6s}: plain {tps:5.1f} T/s, with drafter {res['new_tokens'] / res['time_generate']:5.1f} T/s; "
              f"{res['new_tokens']} tokens in {model.graph_steps - g0} target forwards; text equal for {same}/{len(ref)} chars; {acc}", flush = True)

if __name__ == "__main__":
    main()
