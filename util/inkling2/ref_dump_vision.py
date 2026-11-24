"""
Ground truth for the vision tower of Inkling-2-Mini, from the transformers reference (run in the reference venv):
the image processor's patches and the tower's output (one embedding per image token) for a few test images.
Only the vision tower is built; its weights are read from the checkpoint with the reference's key names.

    <reference venv>/bin/python util/inkling2/ref_dump_vision.py <source checkpoint dir> [--out <dir>]

Writes vision_<name>.npz: pixel_values [tokens, 40, 40, 3], grid [1, 3], embeds [tokens, 2304] (fp32 run), and
embeds_bf16 (the tower in bf16, as the full model runs it), plus the test images as PNG files.
"""
import argparse, json, os, re
import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))

RENAMES = [   # transformers/conversion_mapping.py, "inkling2"
    (r"^model\.visual\.", ""),
    (r"^backbone\.embeddings\.", "embeddings."),
    (r"^embeddings\.position_embedding$", "embeddings.position_embedding.weight"),
    (r"^backbone\.ln_pre\.", "pre_layernorm."),
    (r"^backbone\.layers\.", "layers."),
    (r"^proj_in\.", "merger.linear_1."),
    (r"^proj_out\.", "merger.linear_2."),
    (r"^(layers\.\d+)\.ln_1\.", r"\1.layernorm_before."),
    (r"^(layers\.\d+)\.ln_2\.", r"\1.layernorm_after."),
    (r"^(layers\.\d+)\.mlp\.c_fc\.", r"\1.mlp.fc1."),
    (r"^(layers\.\d+)\.mlp\.c_proj\.", r"\1.mlp.fc2."),
    (r"^(layers\.\d+)\.ls_1\.gamma", r"\1.lambda_1"),
    (r"^(layers\.\d+)\.ls_2\.gamma", r"\1.lambda_2"),
]


def test_images():
    """Small synthetic images: shapes with known colours, text, and an odd size that needs padding"""
    out = {}
    im = Image.new("RGB", (320, 240), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.ellipse((30, 40, 130, 140), fill = (220, 30, 30))
    d.rectangle((180, 60, 290, 170), fill = (30, 60, 220))
    d.polygon([(160, 220), (110, 150), (210, 150)], fill = (30, 160, 60))
    out["shapes"] = im
    im = Image.new("RGB", (400, 120), (250, 250, 240))
    d = ImageDraw.Draw(im)
    d.text((20, 30), "CODE 7391", fill = (0, 0, 0), font_size = 56)
    out["text"] = im
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:137, 0:211]
    a = np.stack([xx * 255 // 210, yy * 255 // 136, (xx + yy) % 256], -1).astype(np.uint8)
    a[40:90, 60:150] = rng.integers(0, 256, (50, 90, 3), dtype = np.uint8)
    out["odd"] = Image.fromarray(a)
    return out


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--out", default = os.path.join(HERE, "ref"))
    ap.add_argument("--device", default = "cuda:0")
    args = ap.parse_args()
    from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
    from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel
    from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor
    cfg = json.load(open(os.path.join(args.model, "config.json")))["vision_config"]
    cfg.pop("model_type", None)
    vc = Inkling2VisionConfig(**cfg)
    print("vision config:", {k: getattr(vc, k) for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
          "patch_size", "spatial_merge_size", "posemb_grid_size", "hidden_act", "layer_norm_eps", "rope_parameters",
          "intermediate_size", "projection_hidden_dim", "text_hidden_size")})
    model = Inkling2VisionModel(vc)
    sd = {}
    path = [f for f in sorted(os.listdir(args.model)) if f.startswith("model") and f.endswith(".safetensors")]
    for f in path:
        with safe_open(os.path.join(args.model, f), "pt") as h:
            for k in h.keys():
                if not k.startswith("model.visual."):
                    continue
                n = k
                for a, b in RENAMES:
                    n = re.sub(a, b, n)
                sd[n] = h.get_tensor(k)
    missing, unexpected = model.load_state_dict(sd, strict = False)
    missing = [m for m in missing if "inv_freq" not in m]
    assert not missing and not unexpected, (missing, unexpected)
    proc = Inkling2ImageProcessor(**json.load(open(os.path.join(args.model, "processor_config.json")))["image_processor"])
    os.makedirs(args.out, exist_ok = True)
    images = test_images()
    feats = {name: proc(images = [im], return_tensors = "pt") for name, im in images.items()}
    # All images in fp32 first: casting the model to bf16 also rounds the rotary frequencies (a buffer), and a cast
    # back to fp32 keeps them rounded (3e-4 at the output)
    res = {name: {} for name in images}
    for tag, dt in (("embeds", torch.float32), ("embeds_bf16", torch.bfloat16)):
        m = model.to(args.device, dt).eval()
        for name, feat in feats.items():
            o = m(pixel_values = feat["pixel_values"].to(args.device, dt), image_grid_thw = feat["image_grid_thw"])
            res[name][tag] = o.pooler_output.float().cpu().numpy()
    for name, im in images.items():
        im.save(os.path.join(args.out, f"vision_{name}.png"))
        pv, grid, r = feats[name]["pixel_values"], feats[name]["image_grid_thw"], res[name]
        np.savez(os.path.join(args.out, f"vision_{name}.npz"), pixel_values = pv.float().numpy(), grid = grid.numpy(), **r)
        d = r["embeds"] - r["embeds_bf16"]
        print(f"{name}: image {im.size}, grid {grid.tolist()}, {pv.shape[0]} tokens, embeds rms {np.sqrt((r['embeds'] ** 2).mean()):.4f}, "
              f"bf16 vs fp32 rel err {np.sqrt((d ** 2).mean()) / np.sqrt((r['embeds'] ** 2).mean()):.2e}")

if __name__ == "__main__":
    main()
