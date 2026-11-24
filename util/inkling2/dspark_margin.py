"""Where greedy text with the drafter first differs from plain greedy text: the target's margin between the two tokens there."""
import sys, torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.generator.sampler import ArgmaxSampler
FMT = "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|><|message_user|><|content_text|>{}<|end_message|>"
P = FMT.format("Explain in about 120 words why the sky is blue.")
@torch.inference_mode()
def main():
    config = Config.from_directory(sys.argv[1]); model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 16384, max_batch_size = 1, max_history = 8)
    draft = Model.from_config(Config.from_directory(sys.argv[2])); dcache = Cache(draft, max_num_tokens = 16384)
    model.load(reserve_per_device = [3.0, 2.5]); draft.load(device = "cuda:1")
    tok = Tokenizer.from_config(config)
    kw = dict(prompt = P, max_new_tokens = 300, sampler = ArgmaxSampler(), stop_conditions = [config.eos_token_id], add_bos = False,
              encode_special_tokens = True, decode_special_tokens = True, completion_only = True)
    a = Generator(model = model, cache = cache, tokenizer = tok).generate(**kw)
    b = Generator(model = model, cache = cache, tokenizer = tok, draft_model = draft, draft_cache = dcache).generate(**kw)
    ia, ib = (tok.encode(t, encode_special_tokens = True)[0] for t in (a, b))
    n = next((i for i in range(min(len(ia), len(ib))) if ia[i] != ib[i]), None)
    if n is None:
        print("no difference"); return
    pre = torch.cat([tok.encode(P, encode_special_tokens = True)[0], ia[:n]]).unsqueeze(0)
    lp = torch.log_softmax(model.forward(pre)[0, -1].float(), -1)
    top = lp.topk(3)
    print(f"first difference at token {n}: plain {tok.decode(ia[n:n+1])!r} logp {lp[ia[n]].item():.4f}, drafted {tok.decode(ib[n:n+1])!r} logp {lp[ib[n]].item():.4f}")
    print("uncached top-3:", [(tok.decode(torch.tensor([i])), round(v, 4)) for v, i in zip(top.values.tolist(), top.indices.tolist())])
main()
