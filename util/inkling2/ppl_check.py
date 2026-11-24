"""
Perplexity of a model on the evaluation texts (uncached forward, rows of --cols tokens): a quick A/B number for a
change of the prefill path. Same text and rows as quality_check.py, without the reference model. With --save /
--against, also KL and top-1 agreement between two runs (bounded below by the routing noise, see quality_check.py).

    python util/inkling2/ppl_check.py <model dir> [--rows 20] [--cols 1024]
        [--save a.pt] [--against a.pt]
"""
import argparse, glob, math, os, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--rows", type = int, default = 20)
    ap.add_argument("--cols", type = int, default = 1024)
    ap.add_argument("--save")
    ap.add_argument("--against")
    args = ap.parse_args()
    from exllamav3 import Config, Model, Tokenizer
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    model.load(reserve_per_device = [3.0, 1.0])
    tokenizer = Tokenizer.from_config(config)
    text = ""
    for f in sorted(glob.glob(os.path.join(HERE, "..", "..", "eval", "eval_texts", "*.txt"))):
        with open(f, encoding = "utf8") as fh:
            text += fh.read() + "\n\n"
    ids = tokenizer.encode(text[: args.rows * args.cols * 8])
    rows = [ids[:, i * args.cols : (i + 1) * args.cols] for i in range(min(args.rows, ids.shape[1] // args.cols))]
    ref = torch.load(args.against) if args.against else None
    nll, n, kl, agree, out = 0.0, 0, 0.0, 0, []
    t0 = time.time()
    for r, row in enumerate(rows):
        lp = torch.log_softmax(model.forward(row)[0].float(), -1)
        tgt = row[0, 1:].to(lp.device)
        nll += -lp[:-1].gather(1, tgt.unsqueeze(1)).sum().item()
        n += tgt.numel()
        if args.save:
            out.append(lp.to("cpu", torch.half))
        if ref is not None:
            rlp = ref[r].to(lp.device).float()
            kl += (rlp.exp() * (rlp - lp)).sum().item()
            agree += (rlp.argmax(-1) == lp.argmax(-1)).sum().item()
    tokens = len(rows) * args.cols
    line = f"{len(rows)} rows of {args.cols} tokens: perplexity {math.exp(nll / n):.4f}, {time.time() - t0:.0f} s"
    if ref is not None:
        line += f"; against {os.path.basename(args.against)}: KL {kl / tokens:.5f}, top-1 agreement {100 * agree / tokens:.2f}%"
    print(line)
    if args.save:
        torch.save(out, args.save)


if __name__ == "__main__":
    main()
