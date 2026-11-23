"""
Quality of a converted model against the unconverted one, over many tokens.

Reference: the source checkpoint (packed experts), fp16 activations. Against it:
  - the same checkpoint with bf16 activations: same weights, different rounding. Top-k routing is discontinuous, so
    this pair already differs; it is the noise floor for every other comparison
  - each model directory given with --models (e.g. the EXL3 conversion)

Per pair: mean KL(reference || model) per token, top-1 agreement, and each model's perplexity on the text (which
does not suffer from the pairing noise). Text: eval/eval_texts, not part of the calibration data.

    python util/inkling2/quality_check.py <source dir> --models <dir> ... \
        [--rows 20] [--cols 1024]
"""
import argparse, gc, glob, math, os, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.inference_mode()
def run(directory, rows, adt = None, ref = None):
    from exllamav3 import Config, Model
    config = Config.from_directory(directory)
    if adt is not None:
        config.activation_dtype = adt
    model = Model.from_config(config)
    model.load(reserve_per_device = [3.0, 1.0])
    out, nll, kl, agree, n = [], 0.0, 0.0, 0, 0
    t0 = time.time()
    for r, ids in enumerate(rows):
        logits = model.forward(ids)[0].float()
        lp = torch.log_softmax(logits, -1)
        tgt = ids[0, 1:].to(lp.device)
        nll += -lp[:-1].gather(1, tgt.unsqueeze(1)).sum().item()
        n += tgt.numel()
        if ref is None:
            out.append(lp.to("cpu", torch.half))
        else:
            rlp = ref[r].to(lp.device).float()
            kl += (rlp.exp() * (rlp - lp)).sum().item()
            agree += (rlp.argmax(-1) == lp.argmax(-1)).sum().item()
        del logits, lp
    dt = time.time() - t0
    model.unload()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    tokens = sum(i.shape[1] for i in rows)
    res = dict(ppl = math.exp(nll / n), seconds = dt)
    if ref is not None:
        res.update(kl = kl / tokens, top1 = agree / tokens)
    return res, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--models", nargs = "*", default = [])
    ap.add_argument("--rows", type = int, default = 20)
    ap.add_argument("--cols", type = int, default = 1024)
    args = ap.parse_args()

    from exllamav3 import Config, Tokenizer
    tokenizer = Tokenizer.from_config(Config.from_directory(args.source))
    text = ""
    for f in sorted(glob.glob(os.path.join(HERE, "..", "..", "eval", "eval_texts", "*.txt"))):
        with open(f, encoding = "utf8") as fh:
            text += fh.read() + "\n\n"
    ids = tokenizer.encode(text[: args.rows * args.cols * 8])
    rows = [ids[:, i * args.cols : (i + 1) * args.cols] for i in range(min(args.rows, ids.shape[1] // args.cols))]
    print(f"{len(rows)} rows of {args.cols} tokens ({len(rows) * args.cols} tokens)", flush = True)

    res, ref = run(args.source, rows)
    print(f"reference (source checkpoint, fp16 activations): perplexity {res['ppl']:.4f}, {res['seconds']:.0f} s", flush = True)
    runs = [("source checkpoint, bf16 activations (noise floor)", args.source, torch.bfloat16)]
    runs += [(os.path.basename(os.path.normpath(m)), m, None) for m in args.models]
    for name, directory, adt in runs:
        res, _ = run(directory, rows, adt, ref)
        print(f"{name}: KL {res['kl']:.5f}, top-1 agreement {100 * res['top1']:.2f}%, perplexity {res['ppl']:.4f}, "
              f"{res['seconds']:.0f} s", flush = True)


if __name__ == "__main__":
    main()
