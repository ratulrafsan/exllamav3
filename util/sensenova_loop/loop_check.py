"""
SenseNova-Loop-9B (upstream PR #609 ported to the local exllamav3 checkout): load, router kernel, generation.

    python util/sensenova_loop/loop_check.py <model dir> [--device cuda:1]

Tests
  router      full-sequence logits with the fused Triton router against the plain torch router
  greedy      three chat prompts, thinking off, greedy; prefill and decode speed
  consistent  the generated tokens (cache, recurrent state, token-by-token) against one full
              forward pass over prompt + output: argmax agreement at every generated position
  reuse       a second turn that extends the first conversation
  long        a prompt of --long tokens: prefill speed and an answer about the text
  batch       two prompts at once
"""
import argparse, os, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LONG_TEXT = os.path.join(HERE, "..", "..", "eval", "eval_texts", "variable_man_mod.txt")
USER = "<|im_start|>user\n{}<|im_end|>\n"
ASSISTANT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"     # thinking off
THINK = "<|im_start|>assistant\n<think>\n"


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device")
    ap.add_argument("--cache", type = int, default = 16384)
    ap.add_argument("--long", type = int, default = 6000)
    ap.add_argument("--tokens", type = int, default = 200)
    args = ap.parse_args()

    from exllamav3 import Config, Model, Cache, Tokenizer, Generator
    from exllamav3.generator.sampler import ArgmaxSampler
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = args.cache, max_batch_size = 2)
    if args.device:
        model.load(device = torch.device(args.device), progressbar = True)
    else:
        model.load(progressbar = True)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer)
    print(f"architecture {config.architecture}, {len(model.fwd_modules)} forward steps over {len(model.modules)} modules")
    print("loaded: " + ", ".join(
        f"cuda:{d} {torch.cuda.memory_allocated(d) / 2**30:.1f} GiB" for d in range(torch.cuda.device_count())))
    stop = [config.eos_token_id, tokenizer.single_id("<|im_end|>"), tokenizer.single_id("<|endoftext|>")]
    greedy = ArgmaxSampler()

    def enc(text):
        return tokenizer.encode(text, add_bos = False, encode_special_tokens = True)

    def gen(prompt, n, **kw):
        t0 = time.time()
        text, res = generator.generate(
            prompt = prompt, max_new_tokens = n, sampler = greedy, stop_conditions = stop, add_bos = False,
            encode_special_tokens = True, decode_special_tokens = True, completion_only = True,
            return_last_results = True, **kw)
        return text, res, time.time() - t0

    def speed(res, dt):
        r = res[0] if isinstance(res, list) else res
        out = f"prompt {r.get('prompt_tokens')} (cached {r.get('cached_tokens')}), new {r.get('new_tokens')}"
        if r.get("time_prefill") and r.get("prompt_tokens") is not None:
            out += f", prefill {(r['prompt_tokens'] - (r.get('cached_tokens') or 0)) / r['time_prefill']:.0f} T/s"
        if r.get("time_generate") and r.get("new_tokens"):
            out += f", decode {r['new_tokens'] / r['time_generate']:.1f} T/s"
        return out + f", wall {dt:.1f} s"

    def full_logits(ids):
        return model.forward(ids, {"attn_mode": "flash_attn_nc"}).float().cpu()

    print("\nrouter (Triton against torch)")
    with open(LONG_TEXT, encoding = "utf8") as f:
        story_text = f.read()
    ids = enc(USER.format(story_text[:3000]) + ASSISTANT)[:, :600]
    os.environ["EXL3_LOOP_ROUTER_TRITON"] = "1"
    a = full_logits(ids)
    os.environ["EXL3_LOOP_ROUTER_TRITON"] = "0"
    t0 = time.time(); b = full_logits(ids); t_torch = time.time() - t0
    os.environ["EXL3_LOOP_ROUTER_TRITON"] = "1"
    t0 = time.time(); a2 = full_logits(ids); t_tri = time.time() - t0
    pa, pb = a.log_softmax(-1), b.log_softmax(-1)
    kl = (pa.exp() * (pa - pb)).sum(-1).mean().item()
    print(f"  {ids.shape[-1]} tokens: top-1 agreement {(a.argmax(-1) == b.argmax(-1)).float().mean().item():.4f}, "
          f"KL {kl:.2e}, max |logit diff| {(a - b).abs().max().item():.3f}; "
          f"repeat with Triton: max |diff| {(a - a2).abs().max().item():.3f}; "
          f"forward {t_tri * 1e3:.0f} ms Triton, {t_torch * 1e3:.0f} ms torch")

    print("\ngreedy + consistent")
    for name, q in (
        ("math", "What is 17 * 23? Show the steps briefly."),
        ("code", "Write a Python function that checks whether a string is a palindrome, ignoring case and spaces."),
        ("german", "Erkläre in drei Sätzen, warum der Himmel blau ist."),
    ):
        prompt = USER.format(q) + ASSISTANT
        text, res, dt = gen(prompt, args.tokens)
        print(f"  {name}: {speed(res, dt)}\n    {text!r}")
        p_ids, o_ids = enc(prompt), enc(text)
        if o_ids.shape[-1] == 0:
            continue
        logits = full_logits(torch.cat([p_ids, o_ids], dim = -1))
        pred = logits[0, p_ids.shape[-1] - 1: -1].argmax(-1)
        same = (pred == o_ids[0]).float()
        first = next((i for i in range(len(same)) if not same[i]), len(same))
        print(f"    full forward agrees with the generated tokens at {int(same.sum())}/{len(same)} positions"
              + ("" if first == len(same) else f" (first difference at {first})"))

    print("\nthinking on")
    text, res, dt = gen(USER.format("A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. "
                                    "What does the ball cost?") + THINK, 600)
    print(f"  {speed(res, dt)}\n    {text!r}")

    print("\nreuse")
    turn1 = USER.format("Name three primary colours. Answer in one short sentence.") + ASSISTANT
    text1, res, dt = gen(turn1, 100)
    print(f"  turn 1: {speed(res, dt)}\n    {text1!r}")
    turn2 = turn1 + text1 + "<|im_end|>\n" + USER.format("Which of them did you name second? One word.") + ASSISTANT
    text2, res, dt = gen(turn2, 100)
    print(f"  turn 2: {speed(res, dt)}\n    {text2!r}")

    print("\nlong")
    story = tokenizer.decode(tokenizer.encode(story_text[:args.long * 8], add_bos = False)[:, :args.long])
    if isinstance(story, list):
        story = story[0]
    q = USER.format(story + "\n\nIn one sentence: who is the first person named in the text above, and what is he doing?") \
        + ASSISTANT
    text, res, dt = gen(q, 150)
    print(f"  {speed(res, dt)}\n    {text!r}")
    q2 = q + text + "<|im_end|>\n" + USER.format("And what is the name of the security commissioner? One short sentence.") \
        + ASSISTANT
    text, res, dt = gen(q2, 150)
    print(f"  follow-up: {speed(res, dt)}\n    {text!r}")

    print("\nbatch")
    prompts = [USER.format(p) + ASSISTANT for p in (
        "What is the capital of Japan? One word.", "What is 17 + 25? Just the number.")]
    texts, res, dt = gen(prompts, 60)
    for t in texts:
        print(f"    {t!r}")
    print(f"  wall {dt:.1f} s")
    print("\npeak: " + ", ".join(
        f"cuda:{d} {torch.cuda.max_memory_allocated(d) / 2**30:.1f} GiB" for d in range(torch.cuda.device_count())))


if __name__ == "__main__":
    main()
