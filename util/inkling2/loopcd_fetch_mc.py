"""
Multiple-choice questions for util/inkling2/loopcd_eval.py, through the Hugging Face dataset viewer's rows API (no
`datasets` / pyarrow needed): ARC-Challenge test (all 1,172) and MMLU test (pages of 100 at evenly spaced
offsets, so every subject is covered). One JSON object per line: src, question, choices, answer (index).

    python util/inkling2/loopcd_fetch_mc.py <out dir> [--mmlu-pages 30]
"""
import argparse, json, os, time, urllib.parse, urllib.request

API = "https://datasets-server.huggingface.co/rows"


def rows(dataset, config, split, offset, length = 100):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
    for attempt in range(6):
        try:
            req = urllib.request.Request(f"{API}?{q}", headers = {"User-Agent": "dataset-fetch/1.0"})
            with urllib.request.urlopen(req, timeout = 60) as r:
                d = json.load(r)
            return [x["row"] for x in d["rows"]], d["num_rows_total"]
        except Exception as e:
            print(f"  retry {attempt + 1} ({type(e).__name__}: {e})", flush = True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {dataset} at {offset}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mmlu-pages", type = int, default = 30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok = True)

    out, off, total = [], 0, 1
    while off < total:
        page, total = rows("allenai/ai2_arc", "ARC-Challenge", "test", off)
        for r in page:
            labels = r["choices"]["label"]
            if r["answerKey"] not in labels:
                continue
            out.append({"src": "arc_challenge", "question": r["question"], "choices": r["choices"]["text"],
                        "answer": labels.index(r["answerKey"])})
        off += 100
        time.sleep(0.5)
    with open(os.path.join(args.out, "arc_challenge.jsonl"), "w", encoding = "utf8") as f:
        f.writelines(json.dumps(r) + "\n" for r in out)
    print(f"ARC-Challenge: {len(out)} questions")

    _, total = rows("cais/mmlu", "all", "test", 0, 1)
    out = []
    for p in range(args.mmlu_pages):
        page, _ = rows("cais/mmlu", "all", "test", p * (total - 100) // max(1, args.mmlu_pages - 1))
        out += [{"src": "mmlu/" + r["subject"], "question": r["question"], "choices": r["choices"],
                 "answer": int(r["answer"])} for r in page]
        time.sleep(0.5)
    with open(os.path.join(args.out, "mmlu_sample.jsonl"), "w", encoding = "utf8") as f:
        f.writelines(json.dumps(r) + "\n" for r in out)
    print(f"MMLU: {len(out)} questions from {len({r['src'] for r in out})} subjects")


if __name__ == "__main__":
    main()
