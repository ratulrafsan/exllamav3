"""
SenseNova-Loop-9B MTP drafting (the head in model-mtp.safetensors): greedy text and speed against plain decoding.

    python util/sensenova_loop/mtp_check.py <model dir> [--device cuda:0]
        [--fp16-state] [--draft 4] [--hybrid 7] [--sparse 16384] [--tokens 400]

Each prompt runs greedy with drafting and without; the script reports decode T/s for both and the
first token at which the two texts differ (none = identical). For token-identical output the
author's recipe uses an fp32 recurrent state (default here) and EXL3_INT8_GEMV=0.
"""
import argparse, time

import torch

USER = "<|im_start|>user\n{}<|im_end|>\n"
ASSISTANT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
THINK = "<|im_start|>assistant\n<think>\n"

PROMPTS = (
    ("code", "Write a Python class for a min-heap with push, pop and peek, with docstrings.", ASSISTANT),
    ("math", "Solve step by step: a train leaves at 9:40 and travels 210 km at 84 km/h. When does it arrive?", THINK),
    ("list", "List the planets of the solar system with one fact each.", ASSISTANT),
    ("prose", "Write a short story about a lighthouse keeper who finds a message in a bottle.", ASSISTANT),
    ("edit", "Here is a function:\n```python\ndef area(w, h):\n    # compute the area\n    result = w * h\n    return result\n\n"
             "def perimeter(w, h):\n    # compute the perimeter\n    result = 2 * (w + h)\n    return result\n```\n"
             "Repeat the code exactly, but rename `result` to `value` in both functions.", ASSISTANT),
)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device")
    ap.add_argument("--cache", type = int, default = 16384)
    ap.add_argument("--tokens", type = int, default = 400)
    ap.add_argument("--draft", type = int, default = 4)
    ap.add_argument("--hybrid", type = int, default = 7)
    ap.add_argument("--sparse", type = int, default = 16384)
    ap.add_argument("--fp16-state", action = "store_true")
    args = ap.parse_args()

    from exllamav3 import Config, Model, Cache, Tokenizer, Generator
    from exllamav3.generator.sampler import ArgmaxSampler
    config = Config.from_directory(args.model)
    assert "mtp" in config.model_classes, "no mtp.* tensors found in the model directory"
    hist = max(args.draft, args.hybrid)
    rdt = torch.half if args.fp16_state else torch.float
    load = dict(device = torch.device(args.device)) if args.device else {}

    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = args.cache, max_history = hist, recurrent_dtype = rdt)
    model.load(progressbar = True, **load)
    mtp = Model.from_config(config, component = "mtp")
    mtp_cache = Cache(mtp, max_num_tokens = args.cache, max_history = hist, recurrent_dtype = rdt)
    mtp.load(**load)
    tokenizer = Tokenizer.from_config(config)
    plain = Generator(model = model, cache = cache, tokenizer = tokenizer)
    stop = [config.eos_token_id, tokenizer.single_id("<|im_end|>"), tokenizer.single_id("<|endoftext|>")]
    greedy = ArgmaxSampler()
    print("loaded: " + ", ".join(
        f"cuda:{d} {torch.cuda.memory_allocated(d) / 2**30:.1f} GiB" for d in range(torch.cuda.device_count())))

    def run(g, prompt):
        text, res = g.generate(
            prompt = prompt, max_new_tokens = args.tokens, sampler = greedy, stop_conditions = stop, add_bos = False,
            encode_special_tokens = True, decode_special_tokens = True, completion_only = True,
            return_last_results = True)
        return text, res["new_tokens"] / res["time_generate"], res

    def bench(g, label):
        out = {}
        run(g, USER.format("Count from 1 to 30.") + ASSISTANT)       # warm-up (kernel tuning, workspaces)
        for name, q, tail in PROMPTS:
            out[name] = run(g, USER.format(q) + tail)
        return out

    base = bench(plain, "plain")
    del plain
    drafted = Generator(model = model, cache = cache, tokenizer = tokenizer, draft_model = mtp, draft_cache = mtp_cache,
                        num_draft_tokens = args.draft, hybrid_draft_tokens = args.hybrid, sparse_draft_head = args.sparse)
    fast = bench(drafted, "mtp")

    print(f"\ndraft {args.draft}, hybrid {args.hybrid}, sparse head {args.sparse}, "
          f"recurrent state {'fp16' if args.fp16_state else 'fp32'}")
    ratios = []
    for name, _, _ in PROMPTS:
        (t0, s0, r0), (t1, s1, r1) = base[name], fast[name]
        a, b = (tokenizer.encode(t, add_bos = False, encode_special_tokens = True)[0] for t in (t0, t1))
        m = min(len(a), len(b))
        first = next((i for i in range(m) if a[i].item() != b[i].item()), None)
        same = "identical" if first is None and len(a) == len(b) else f"differ from token {m if first is None else first}"
        acc = ""
        if r1.get("accepted_draft_tokens") is not None:
            acc = f", drafts accepted {r1['accepted_draft_tokens']}/{r1['accepted_draft_tokens'] + r1.get('rejected_draft_tokens', 0)}"
        ratios.append(s1 / s0)
        print(f"  {name:6s} plain {s0:6.1f} T/s, MTP {s1:6.1f} T/s ({s1 / s0:.2f}x), {r0['new_tokens']} / {r1['new_tokens']} tokens, {same}{acc}")
    print(f"  mean speed-up {sum(ratios) / len(ratios):.2f}x")
    print("peak: " + ", ".join(
        f"cuda:{d} {torch.cuda.max_memory_allocated(d) / 2**30:.1f} GiB" for d in range(torch.cuda.device_count())))


if __name__ == "__main__":
    main()
