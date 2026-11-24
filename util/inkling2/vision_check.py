"""
The vision tower of the port against the transformers reference (inkling/ref_dump_vision.py): the preprocessing
of each test image (cells of 40 x 40 pixels) and the tower's output (one embedding per image token). Then, with
--describe, a short generation about each test image through the generator.

    python util/inkling2/vision_check.py <model dir> [--ref <dir>] [--describe]
"""
import argparse, glob, os, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
EFFORT = "<|message_system|><|content_text|>Thinking effort level: {}<|end_message|>"
IMAGE = "<|message_user|><|content_image|>{}<|end_message|>"
USER = "<|message_user|><|content_text|>{}<|end_message|>"
QUESTIONS = {
    "shapes": "List the shapes in this image with their colours. One short line each.",
    "text": "What text does this image show? Reply with the text only.",
    "odd": "Describe this image in one sentence.",
}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--ref", default = os.path.join(HERE, "ref"))
    ap.add_argument("--describe", action = "store_true"); ap.add_argument("--image", nargs = "*", default = [])
    args = ap.parse_args()
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator
    from exllamav3.generator.sampler import ArgmaxSampler
    config = Config.from_directory(args.model)
    assert "vision" in config.model_classes, "this model directory has no vision tower"
    vision = Model.from_config(config, component = "vision")
    vision.load()
    tok = Tokenizer.from_config(config)
    print(f"vision tower on {vision.tower.device}, {vision.tower.weights_numel() / 1e6:.0f}M parameters, "
          f"{torch.cuda.memory_allocated(vision.tower.device) / 2**20:.0f} MiB")

    names = []
    for f in sorted(glob.glob(os.path.join(args.ref, "vision_*.npz"))):
        name = os.path.basename(f)[len("vision_"):-4]
        names.append(name)
        d = np.load(f)
        im = Image.open(os.path.join(args.ref, f"vision_{name}.png"))
        cells, grid = vision.preprocess(im)
        ref_px = torch.from_numpy(d["pixel_values"])
        px_err = (cells - ref_px).abs().max().item() if cells.shape == ref_px.shape else float("nan")
        t0 = time.time()
        emb = vision.tower.forward(cells, {"grid": grid}).cpu()
        dt = time.time() - t0
        ref = torch.from_numpy(d["embeds"])
        rel = lambda a, b: (torch.sqrt((a - b).square().mean()) / torch.sqrt(b.square().mean())).item()
        cos = torch.nn.functional.cosine_similarity(emb, ref, dim = -1).min().item()
        print(f"{name}: image {im.size}, grid {grid} (reference {tuple(d['grid'][0][1:])}), {cells.shape[0]} tokens; "
              f"cells max abs diff {px_err:.2e}; embeddings rel err {rel(emb, ref):.2e} vs the fp32 reference "
              f"(min cosine {cos:.6f}), {rel(emb, torch.from_numpy(d['embeds_bf16'])):.2e} vs its bf16 run; {1e3 * dt:.0f} ms")

    if not args.describe and not args.image:
        return
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 16384, max_batch_size = 1)
    model.load(reserve_per_device = [3.0, 1.0])
    gen = Generator(model = model, cache = cache, tokenizer = tok)
    todo = [(n, os.path.join(args.ref, f"vision_{n}.png"), QUESTIONS.get(n, "Describe this image.")) for n in names]
    todo += [(os.path.basename(p), p, "Describe this image in two sentences.") for p in args.image]
    for name, path, question in todo:
        mme = vision.get_image_embeddings(tok, Image.open(path))
        prompt = EFFORT.format("0.2") + IMAGE.format(mme.text_alias) + USER.format(question)
        g0 = model.graph_steps
        text, res = gen.generate(
            prompt = prompt, embeddings = [mme], max_new_tokens = 300, sampler = ArgmaxSampler(),
            stop_conditions = [config.eos_token_id], add_bos = False, encode_special_tokens = True,
            decode_special_tokens = True, completion_only = True, return_last_results = True)
        print(f"\n{name}: {mme.mm_length} image tokens, prompt {res['prompt_tokens']} tokens, "
              f"{res['new_tokens']} new tokens at {res['new_tokens'] / res['time_generate']:.1f} T/s "
              f"({model.graph_steps - g0} graph steps)\n  Q: {question}\n  A: {text!r}")


if __name__ == "__main__":
    main()
