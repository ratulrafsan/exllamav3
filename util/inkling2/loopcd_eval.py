"""
Does LoopCD (arXiv:2610.02185) help Inkling-2-Mini? Two cheap, teacher-forced measurements over a grid of
strengths, each against the plain model on the same items:

    mc      multiple-choice scoring, as in the paper: the question as a chat turn at thinking effort 0, the
            answer = the option letter with the highest logit at the first token of the reply. Questions from
            util/inkling2/loopcd_fetch_mc.py (ARC-Challenge, MMLU). Reported: accuracy, and the questions whose
            answer changed for the better / for the worse (a paired comparison; p from the sign test)
    text    next-token prediction on the evaluation texts (rows of 1024 tokens, uncached): perplexity and top-1
            accuracy of the guided distribution. Contrastive decoding sharpens the distribution, so perplexity
            is expected to rise; top-1 accuracy is the number to read

Logits mode needs one pass per reference pass: the last pass's and the reference pass's logits are exported and
every strength (fixed and adaptive) is computed from them. Hidden mode changes the forward itself: one pass per
(reference pass, strength), on one model load.

    python util/inkling2/loopcd_eval.py <EXL3 model dir> --mode logits --ref 1
        [--task mc text] [--data util/inkling2/loopcd_data] [--limit N] [--out results.json]
    ... --mode hidden --ref 1 2 --w 0.25 0.5 1.0
    ... --mode plain          the model as it is loaded (for example with a control vector, EXL3_I2_CVEC)
"""
import argparse, glob, json, math, os, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LETTERS = "ABCDE"
PROMPT = ("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
          "<|message_user|><|content_text|>{q}\n\n{opts}\n\nAnswer with the letter of the correct option only."
          "<|end_message|><|message_model|><|content_text|>")
W_FIXED = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
W_ADAPT = (0.25, 0.5, 1.0, 1.5, 2.0)


def sign_p(better: int, worse: int) -> float:
    """Two-sided sign test: probability of a split at least this uneven if better and worse were equally likely"""
    n, k = better + worse, min(better, worse)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def mc_line(name, pred, base, ans, src):
    ok, ok0 = pred == ans, base == ans
    better, worse = int((ok & ~ok0).sum()), int((~ok & ok0).sum())
    per = "  ".join(f"{s} {100 * ok[m].float().mean():.2f}" for s, m in src.items())
    print(f"  {name:<34} acc {100 * ok.float().mean():6.2f}  ({100 * (ok.float().mean() - ok0.float().mean()):+.2f})  "
          f"changed {int((pred != base).sum()):4d}  better {better:3d}  worse {worse:3d}  p {sign_p(better, worse):.3f}   {per}",
          flush = True)
    return {"name": name, "acc": ok.float().mean().item(), "better": better, "worse": worse,
            "changed": int((pred != base).sum()), "p": sign_p(better, worse),
            **{f"acc_{s}": ok[m].float().mean().item() for s, m in src.items()}}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mode", choices = ("logits", "hidden", "plain"), required = True)
    ap.add_argument("--ref", type = int, nargs = "+", default = [1])
    ap.add_argument("--w", type = float, nargs = "+", default = [0.25, 0.5, 1.0], help = "hidden mode: strengths")
    ap.add_argument("--task", nargs = "+", default = ["mc", "text"])
    ap.add_argument("--data", default = os.path.join(HERE, "loopcd_data"))
    ap.add_argument("--limit", type = int, default = 0, help = "questions per file (0: all)")
    ap.add_argument("--rows", type = int, default = 20)
    ap.add_argument("--cols", type = int, default = 1024)
    ap.add_argument("--device")
    ap.add_argument("--out")
    args = ap.parse_args()
    from exllamav3 import Config, Model, Tokenizer
    from exllamav3.modules import inkling2 as I2

    config = Config.from_directory(args.model)
    if args.mode == "plain":
        args.ref = []
    if args.mode == "logits":
        assert len(args.ref) == 1, "logits mode: one reference pass per model load"
        config.set_loopcd("logits", args.ref[0], 0.0, 0.0)
    model = Model.from_config(config)
    if args.device:
        model.load(device = args.device)
    else:
        model.load(reserve_per_device = [3.0, 1.0])
    tok = Tokenizer.from_config(config)
    print(f"-- {os.path.basename(args.model.rstrip('/'))}, LoopCD mode {args.mode}, reference pass(es) {args.ref}", flush = True)
    results = {"model": args.model, "mode": args.mode, "ref": args.ref}
    letter_ids = [tok.encode(c, encode_special_tokens = False)[0, 0].item() for c in LETTERS]

    # ---- multiple choice
    def mc_load():
        qs = []
        for f in sorted(glob.glob(os.path.join(args.data, "*.jsonl"))):
            with open(f, encoding = "utf8") as fh:
                rows = [json.loads(l) for l in fh]
            rows = [r for r in rows if 2 <= len(r["choices"]) <= len(LETTERS)]
            qs += rows[: args.limit] if args.limit else rows
        return qs

    def mc_pass(qs, export):
        """Per question: the letters' logits (last pass, and the reference pass with export), p1 - p2 and argmax"""
        zl, zrl, gap, top = [], [], [], []
        t0 = time.time()
        for n, r in enumerate(qs):
            opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"]))
            ids = tok.encode(PROMPT.format(q = r["question"].strip(), opts = opts), encode_special_tokens = True)
            p = {"last_tokens_only": 1}
            if export:
                p["loopcd_export"] = True
            z = model.forward(ids, p)[0, -1].float()
            if export:
                z, zr = (t[0, -1].float() for t in p["loopcd_logits"])
            mask = torch.full((len(LETTERS),), float("-inf"), device = z.device)
            mask[: len(r["choices"])] = 0
            zl.append((z[letter_ids] + mask).cpu())
            if export:
                zrl.append((zr[letter_ids] + mask).cpu())
            p2 = torch.softmax(z, -1).topk(2).values
            gap.append((p2[0] - p2[1]).item())
            top.append(int(z.argmax()))
            if n % 500 == 499:
                print(f"     {n + 1}/{len(qs)} questions, {time.time() - t0:.0f} s", flush = True)
        out = {"z": torch.stack(zl), "gap": torch.tensor(gap), "top": torch.tensor(top)}
        if export:
            out["zr"] = torch.stack(zrl)
        return out

    if "mc" in args.task:
        qs = mc_load()
        ans = torch.tensor([r["answer"] for r in qs])
        kinds = [r["src"].split("/")[0] for r in qs]
        src = {s: torch.tensor([k == s for k in kinds]) for s in sorted(set(kinds))}
        print(f"-- multiple choice: {len(qs)} questions ({', '.join(f'{s} {int(m.sum())}' for s, m in src.items())})", flush = True)
        res = results["mc"] = []
        if args.mode == "logits":
            d = mc_pass(qs, True)
            z, zr = d["z"], d["zr"]
            is_letter = torch.tensor([t in letter_ids for t in d["top"].tolist()]).float().mean().item()
            print(f"  the reply's first token is an option letter in {100 * is_letter:.1f}% of the questions; "
                  f"mean 1 - (p1 - p2) = {(1 - d['gap']).mean():.3f}")
            base = z.argmax(-1)
            res.append(mc_line("plain (last pass)", base, base, ans, src))
            res.append(mc_line(f"reference pass {args.ref[0]} alone", zr.nan_to_num(neginf = -1e9).argmax(-1), base, ans, src))
            diff = (z - zr).nan_to_num(nan = 0.0)                   # masked letters: -inf - -inf
            for w in W_FIXED:
                res.append(mc_line(f"fixed w = {w:g}", (z + w * diff).argmax(-1), base, ans, src))
            for wm in W_ADAPT:
                w = (wm * (1 - d["gap"])).unsqueeze(1)
                res.append(mc_line(f"adaptive, cap {wm:g}", (z + w * diff).argmax(-1), base, ans, src))
        else:
            model.set_loopcd("off")
            base = mc_pass(qs, False)["z"].argmax(-1)
            res.append(mc_line("plain", base, base, ans, src))
            for ref in args.ref:
                for w in args.w:
                    model.set_loopcd("hidden", ref, strength = w)
                    pred = mc_pass(qs, False)["z"].argmax(-1)
                    res.append(mc_line(f"hidden, ref {ref}, w = {w:g}", pred, base, ans, src))
            model.set_loopcd("off")

    # ---- next-token prediction on text
    if "text" in args.task:
        text = ""
        for f in sorted(glob.glob(os.path.join(HERE, "..", "..", "eval", "eval_texts", "*.txt"))):
            with open(f, encoding = "utf8") as fh:
                text += fh.read() + "\n\n"
        ids = tok.encode(text[: args.rows * args.cols * 8])
        rows = [ids[:, i * args.cols : (i + 1) * args.cols] for i in range(min(args.rows, ids.shape[1] // args.cols))]
        n_tok = sum(r.shape[1] - 1 for r in rows)
        print(f"-- text: {len(rows)} rows of {args.cols} tokens", flush = True)
        res = results["text"] = []

        def score(acc, name, z, tgt):
            lp = torch.log_softmax(z[:-1], -1)
            a = acc.setdefault(name, [0.0, 0])
            a[0] += -lp.gather(1, tgt.unsqueeze(1)).sum().item()
            a[1] += int((lp.argmax(-1) == tgt).sum())

        def report(acc):
            b = acc[next(iter(acc))]
            for name, (nll, hit) in acc.items():
                print(f"  {name:<34} perplexity {math.exp(nll / n_tok):8.4f}   top-1 {100 * hit / n_tok:6.2f}%  "
                      f"({100 * (hit - b[1]) / n_tok:+.2f})", flush = True)
                res.append({"name": name, "ppl": math.exp(nll / n_tok), "top1": hit / n_tok})

        acc = {}
        if args.mode == "logits":
            for row in rows:
                p = {"loopcd_export": True}
                model.forward(row, p)
                z, zr = (t[0].float() for t in p["loopcd_logits"])
                tgt = row[0, 1:].to(z.device)
                score(acc, "plain (last pass)", z, tgt)
                score(acc, f"reference pass {args.ref[0]} alone", zr, tgt)
                for w in W_FIXED:
                    score(acc, f"fixed w = {w:g}", I2.loopcd_logits(z, zr, torch.tensor([w, 0.0], device = z.device)), tgt)
                for wm in W_ADAPT:
                    score(acc, f"adaptive, cap {wm:g}", I2.loopcd_logits(z, zr, torch.tensor([0.0, wm], device = z.device)), tgt)
                del z, zr
        else:
            settings = [("plain", None)] + [(f"hidden, ref {r}, w = {w:g}", (r, w)) for r in args.ref for w in args.w]
            for name, s in settings:
                model.set_loopcd("off") if s is None else model.set_loopcd("hidden", s[0], strength = s[1])
                for row in rows:
                    z = model.forward(row)[0].float()
                    score(acc, name, z, row[0, 1:].to(z.device))
            model.set_loopcd("off")
        report(acc)

    if args.out:
        with open(args.out, "w", encoding = "utf8") as f:
            json.dump(results, f, indent = 1)


if __name__ == "__main__":
    main()
