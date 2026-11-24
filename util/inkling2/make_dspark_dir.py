"""
Make the model directory of Inkling-2-Mini's draft model (DSpark, mtp.safetensors) for ExLlamaV3: a config.json with
the drafter's architecture and a link to the checkpoint's mtp.safetensors.

    python util/inkling2/make_dspark_dir.py <Inkling-2-Mini dir (source or EXL3)> <output dir> [--copy]
"""
import argparse, json, os, shutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("--copy", action = "store_true")
    args = ap.parse_args()
    with open(os.path.join(args.model, "config.json")) as f:
        cfg = json.load(f)
    m, t = cfg["mtp_config"], cfg["text_config"]
    n = m["num_hidden_layers"]
    local = set(m["local_layer_ids"])
    out = {
        "architectures": ["Inkling2DSparkDraftModel"],
        "hidden_size": m["hidden_size"], "num_hidden_layers": n,
        "num_attention_heads": m["num_attention_heads"], "num_key_value_heads": m["num_key_value_heads"],
        "head_dim": m["head_dim"], "intermediate_size": m["intermediate_size"], "hidden_act": "silu",
        "rms_norm_eps": m["rms_norm_eps"], "rope_theta": m["rope_theta"],
        "max_position_embeddings": m["max_position_embeddings"],
        "layer_types": ["sliding_attention" if i in local else "full_attention" for i in range(n)],
        "sliding_window": m["sliding_window_size"], "is_causal": False,
        "block_size": m["block_size"], "target_layer_ids": m["target_layer_ids"], "tap_shift": 0,
        "mask_token_id": 0,                     # unused: the mask rows are the learned mask_embed
        "markov_rank": m["markov_rank"], "markov_head_type": m["markov_head_type"],
        "vocab_size": t["vocab_size"], "unpadded_vocab_size": t["unpadded_vocab_size"],
        "final_logit_softcapping": t["final_logit_softcapping"],
        "eos_token_id": cfg.get("eos_token_id"),
    }
    assert m["markov_head_type"] == "rnn_hidden", "only the rnn_hidden Markov head is implemented"
    os.makedirs(args.out, exist_ok = True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(out, f, indent = 2)
    src = os.path.abspath(os.path.join(args.model, "mtp.safetensors"))
    dst = os.path.join(args.out, "model.safetensors")
    if os.path.lexists(dst):
        os.remove(dst)
    if args.copy:
        shutil.copyfile(src, dst)
    else:
        os.symlink(src, dst)
    print(f"wrote {args.out}: config.json, model.safetensors -> {src}")

if __name__ == "__main__":
    main()
