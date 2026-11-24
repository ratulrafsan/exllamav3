"""
ExLlamaV3's qbench measurement for Inkling-2: KL divergence and perplexity of converted models against the
unconverted checkpoint on a qbench trace (eval/qbench_prompts.py: conversations sampled from the model itself;
only the sampled response positions are scored). qbench.py streams a model module by module, which this port's
looped model cannot do, so the models run here through their normal forward; the numbers come from qbench's own
code (eval/qbench/measure.py: DiffStats, save_reference_row), full distributions, reference logits stored fp16.

    python util/inkling2/qbench_i2.py <trace.json> --ref <source checkpoint dir>
        --models <dir> ... [--store <dir for the reference logits>] [--out results.json]

Reference: the source checkpoint (NVFP4 / MXFP8 experts as stored) with fp16 activations. The noise floor is the
same checkpoint with bf16 activations: the same weights with different rounding (see quality_check.py).
"""
import argparse, gc, json, os, sys, time, types
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "eval"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))


def load_trace(path):
    with open(path) as f:
        trace = json.load(f)
    rows = []
    for r in trace["rows"]:
        seq = r["input_ids"] + r["response_ids"]
        # logits at [p - 1, len) predict the sampled tokens and the token after them (qbench/data.py)
        rows.append((torch.tensor(seq, dtype = torch.long).unsqueeze(0), len(r["input_ids"]) - 1, len(seq)))
    return rows, trace["vocab_size"], trace.get("meta", {})


@torch.inference_mode()
def run(directory, rows, each, adt = None, reserve = (4.0, 8.0)):
    from exllamav3 import Config, Model
    config = Config.from_directory(directory)
    if adt is not None:
        config.activation_dtype = adt
    model = Model.from_config(config)
    model.load(reserve_per_device = list(reserve))
    stats_dev = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
    t0 = time.time()
    for r, (ids, a, b) in enumerate(rows):
        logits = model.forward(ids, {"last_tokens_only": b - a})
        each(r, logits.to(stats_dev))
        del logits
    dt = time.time() - t0
    model.unload()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace"); ap.add_argument("--ref", required = True)
    ap.add_argument("--models", nargs = "*", default = [])
    ap.add_argument("--store"); ap.add_argument("--out")
    ap.add_argument("--no-floor", action = "store_true")
    args = ap.parse_args()
    # qbench/data.py imports `datasets` for its text corpora, which a trace does not use
    if "datasets" not in sys.modules:
        try:
            import datasets  # noqa: F401
        except ImportError:
            stub = types.ModuleType("datasets")
            stub.load_dataset = None
            sys.modules["datasets"] = stub
    from qbench.measure import DiffStats, save_reference_row, print_stats
    from qbench.data import save_tensors

    rows, vocab, meta = load_trace(args.trace)
    n_scored = sum(b - a - 1 for _, a, b in rows)
    print(f"trace: {len(rows)} rows, {meta.get('input_tokens')} input + {meta.get('output_tokens')} output tokens, "
          f"{n_scored} scored tokens, vocabulary {vocab}", flush = True)
    # The scored positions of every row as a row of their own: DiffStats indexes logits by position
    width = max(b - a for _, a, b in rows)
    ids2 = torch.zeros((len(rows), width), dtype = torch.long)
    for r, (ids, a, b) in enumerate(rows):
        ids2[r, : b - a] = ids[0, a:b]
    ranges = [(0, b - a) for _, a, b in rows]
    store = args.store or os.path.splitext(args.trace)[0] + "_ref"
    out_path = args.out or os.path.splitext(args.trace)[0] + "_results.json"
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}

    def save():
        with open(out_path, "w") as f:
            json.dump(results, f, indent = 1)

    if not os.path.exists(os.path.join(store, "conf.safetensors")):
        os.makedirs(store, exist_ok = True)
        stats, conf_rows = DiffStats(ids2, ranges, vocab, None), []

        def each(r, logits):
            stats(r, logits)
            save_reference_row(store, r, logits, ranges[r], conf_rows)

        dt = run(args.ref, rows, each)
        save_tensors(os.path.join(store, "conf.safetensors"), {"conf": torch.cat(conf_rows).half()})
        results["reference"] = dict(stats.results(), seconds = dt, source = args.ref)
        save()
    print_stats("reference (source, fp16)", results["reference"])

    todo = [] if args.no_floor else [("noise floor (source, bf16)", args.ref, torch.bfloat16)]
    todo += [(os.path.basename(os.path.normpath(m)), m, None) for m in args.models]
    for label, directory, adt in todo:
        if label not in results:
            stats = DiffStats(ids2, ranges, vocab, store)
            dt = run(directory, rows, stats, adt)
            res = stats.results()
            kl = stats.kl_vector()
            results[label] = dict(res, seconds = dt, source = directory)
            if kl is not None:
                save_tensors(os.path.join(store, f"kl_{label.replace(' ', '_').replace('/', '_')}.safetensors"), {"kl": kl})
            save()
        print_stats(label, results[label])
        sys.stdout.flush()


if __name__ == "__main__":
    main()
