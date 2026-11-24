from __future__ import annotations
from typing_extensions import override
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..model.model import Model
from ..modules import Module
from ..tokenizer import Tokenizer, MMEmbedding

"""
Local patch: the vision tower of Inkling-2 (`model.visual.*` in the checkpoint), as the "vision" component of
Inkling2LoopConfig. It follows the transformers reference (models/inkling2: Inkling2ImageProcessor,
Inkling2VisionModel); util/inkling2/vision_check.py compares both steps with dumps from that reference.

  * Preprocessing: no resizing. The image is padded at the bottom and right to whole cells of 40 x 40 pixels (pad
    value -1 on the 0..255 scale) and scaled to [-1, 1]; a cell becomes one image token of the language model.
    Transparent images are put on a grey checkerboard first.
  * Tower: every cell is 2 x 2 patches of 20 x 20 pixels, embedded by one linear layer. Learned position embeddings
    (a 32 x 32 grid, resampled bilinearly to the image's patch grid) are added, a class token goes in front, then a
    layer norm and 23 pre-norm transformer layers (full attention with axial 2D rotary positions, GELU MLP, layer
    scale). The four patch states of a cell are concatenated and projected by a two-layer MLP to the hidden size of
    the language model.
  * The result replaces the embeddings of the image tokens after the text model's embedding norm (see
    Inkling2Embed.lookup). In the prompt an image is a message of its own:
        <|message_user|><|content_image|>{image tokens}<|end_message|>
    get_image_embeddings() returns an MMEmbedding whose alias stands for the image tokens only.

The tower is 303M parameters and runs in fp32, in plain PyTorch, once per image. It loads on the last GPU by
default (EXL3_I2_VISION_DEVICE = first | last | cpu | cuda:N): the first GPU is kept for the looped text layers.
"""

CELL_FILL = -1.0                 # pad value, on the 0..255 scale
MAX_SIDE = 4096                  # the model card: each image dimension between 40 and 4096 pixels
ATTN_MAX_ELEMS = 1 << 27         # attention logits held at once: heads * queries * keys (fp32: 512 MB)


def composite_on_checkerboard(rgba: np.ndarray) -> np.ndarray:
    """uint8 [H, W, 4] -> uint8 [H, W, 3] on the reference's 8-pixel checkerboard (204 / 153)"""
    h, w = rgba.shape[:2]
    rows = np.arange(h)[:, None] // 8
    cols = np.arange(w)[None, :] // 8
    background = np.where((rows + cols) % 2 == 0, 204, 153)[..., None].astype(np.int32)
    px = rgba.astype(np.int32)
    alpha = px[..., 3:4]
    return ((px[..., :3] * alpha + background * (255 - alpha) + 127) // 255).astype(np.uint8)


def preprocess_image(image: Image.Image, cell: int, mean: float, std: float, rescale: float,
                     max_side: int = MAX_SIDE) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    PIL image -> (cells fp32 [rows * cols, cell, cell, 3] in raster order, (rows, cols)). An image with a side over
    max_side is scaled down first (the reference does not resize at all).
    """
    if max(image.size) > max_side:
        s = max_side / max(image.size)
        image = image.resize((max(1, round(image.size[0] * s)), max(1, round(image.size[1] * s))), Image.Resampling.BICUBIC)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        px = composite_on_checkerboard(np.array(image.convert("RGBA")))
    else:
        px = np.array(image.convert("RGB"))
    h, w = px.shape[:2]
    rows, cols = math.ceil(h / cell), math.ceil(w / cell)
    x = torch.full((rows * cell, cols * cell, 3), CELL_FILL, dtype = torch.float32)
    x[:h, :w] = torch.from_numpy(px).float()
    x = (x * rescale - mean) / std
    x = x.reshape(rows, cell, cols, cell, 3).permute(0, 2, 1, 3, 4).reshape(rows * cols, cell, cell, 3)
    return x.contiguous(), (rows, cols)


def _pick_device(device: torch.device) -> torch.device:
    want = os.environ.get("EXL3_I2_VISION_DEVICE", "last")
    if want == "first" or not torch.cuda.is_available():
        return device
    if want == "last":
        return torch.device(f"cuda:{torch.cuda.device_count() - 1}")
    return torch.device(want)


class Inkling2VisionTower(Module):
    """All weights of the tower as plain fp32 tensors; forward for one image"""

    def __init__(self, config, key: str = "model.visual"):
        super().__init__(config, key, None)
        self.W = None
        self.caps.update({"x_cpu": True})

    @override
    def can_defer_load(self):
        return False

    @override
    def optimizer_targets(self):
        return []

    @override
    def get_tensors(self):
        return {}

    @override
    def weights_numel(self):
        return sum(t.numel() for t in self.W.values()) if self.W else 0

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = _pick_device(device)
        stc = self.config.stc
        n = len(self.key) + 1
        self.W = {
            k[n:]: stc.get_tensor(k, self.device, allow_bf16 = True, no_defer = True).float()
            for k in stc.list_tensors(prefix = self.key)
        }
        v = self.config.i2_vision
        dim = v["hidden_size"] // v["num_attention_heads"] // 2
        self.inv_freq = 1.0 / (v["rope_theta"] ** (torch.arange(0, dim, 2, dtype = torch.float, device = self.device) / dim))

    @override
    def unload(self):
        self.W = None
        self.device = None

    def _positions(self, rows: int, cols: int):
        """Per sequence row (class token, then patches cell by cell): the (row, column) used by the rotary positions"""
        m = self.config.i2_vision["merge"]
        hh, ww = torch.meshgrid(
            torch.arange(rows * m, device = self.device), torch.arange(cols * m, device = self.device), indexing = "ij"
        )
        order = lambda t: t.reshape(rows, m, cols, m).transpose(1, 2).flatten()
        zero = torch.zeros(1, dtype = torch.long, device = self.device)
        # patches are 1-indexed, the class token sits at (0, 0)
        return torch.cat([zero, order(hh) + 1]), torch.cat([zero, order(ww) + 1])

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q, k, v: [heads, L, d] -> [heads, L, d]; queries in chunks so that the logits stay bounded"""
        H, L, d = q.shape
        step = max(1, ATTN_MAX_ELEMS // (H * L))
        if step >= L:
            return torch.softmax(torch.matmul(q, k.transpose(1, 2)) * d ** -0.5, dim = -1) @ v
        out = torch.empty_like(q)
        kt = k.transpose(1, 2)
        for a in range(0, L, step):
            out[:, a : a + step] = torch.softmax(torch.matmul(q[:, a : a + step], kt) * d ** -0.5, dim = -1) @ v
        return out

    @override
    @torch.inference_mode
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        """x: cells [rows * cols, cell, cell, 3] (preprocess_image), params["grid"] = (rows, cols) -> [cells, D_text]"""
        W, v = self.W, self.config.i2_vision
        rows, cols = params["grid"]
        m, p, D, nh, eps = v["merge"], v["patch"], v["hidden_size"], v["num_attention_heads"], v["layer_norm_eps"]
        x = x.to(self.device, torch.float)
        n = x.shape[0]
        assert n == rows * cols and x.shape[1] == m * p

        # Patches cell by cell (2 x 2 inside a cell), each flattened as (y, x, channel)
        patches = x.reshape(n, m, p, m, p, x.shape[-1]).transpose(2, 3).reshape(n * m * m, -1)
        h = F.linear(patches, W["backbone.embeddings.patch_embedding.weight"])
        g = v["posemb_grid_size"]
        pos = W["backbone.embeddings.position_embedding"].view(g, g, D).permute(2, 0, 1).unsqueeze(0)
        pos = F.interpolate(pos, size = (rows * m, cols * m), mode = "bilinear", align_corners = False)[0]
        pos = pos.permute(1, 2, 0).reshape(rows, m, cols, m, D).transpose(1, 2).reshape(-1, D)
        cls = W["backbone.embeddings.class_embedding"] + W["backbone.embeddings.cls_position_embedding"]
        h = torch.cat([cls.unsqueeze(0), h + pos], dim = 0)
        h = F.layer_norm(h, (D,), W["backbone.ln_pre.weight"], W["backbone.ln_pre.bias"], eps)
        L = h.shape[0]

        # Axial rotary angles: width frequencies, then height, each for one interleaved channel pair
        hp, wp = self._positions(rows, cols)
        ang = torch.cat([wp[:, None].float() * self.inv_freq, hp[:, None].float() * self.inv_freq], dim = -1)
        ang = ang.repeat_interleave(2, dim = -1)                                       # [L, head_dim]
        cos, sin = ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)

        def rope(t):                                                                   # [L, heads, d]
            t2 = t.view(*t.shape[:-1], -1, 2)
            rot = torch.stack((-t2[..., 1], t2[..., 0]), dim = -1).flatten(-2)
            return t * cos + rot * sin

        for i in range(v["num_hidden_layers"]):
            k = f"backbone.layers.{i}."
            a = F.layer_norm(h, (D,), W[k + "ln_1.weight"], W[k + "ln_1.bias"], eps)
            q = rope(F.linear(a, W[k + "self_attn.q_proj.weight"], W[k + "self_attn.q_proj.bias"]).view(L, nh, -1))
            kk = rope(F.linear(a, W[k + "self_attn.k_proj.weight"], W[k + "self_attn.k_proj.bias"]).view(L, nh, -1))
            vv = F.linear(a, W[k + "self_attn.v_proj.weight"], W[k + "self_attn.v_proj.bias"]).view(L, nh, -1)
            o = self._attention(q.transpose(0, 1), kk.transpose(0, 1), vv.transpose(0, 1)).transpose(0, 1).reshape(L, D)
            o = F.linear(o, W[k + "self_attn.out_proj.weight"], W[k + "self_attn.out_proj.bias"])
            h = h + W[k + "ls_1.gamma"] * o
            a = F.layer_norm(h, (D,), W[k + "ln_2.weight"], W[k + "ln_2.bias"], eps)
            a = F.linear(F.gelu(F.linear(a, W[k + "mlp.c_fc.weight"], W[k + "mlp.c_fc.bias"])),
                         W[k + "mlp.c_proj.weight"], W[k + "mlp.c_proj.bias"])
            h = h + W[k + "ls_2.gamma"] * a

        # The four patch states of a cell, concatenated, through the projection MLP
        cells = h[1:].reshape(n, m * m * D)
        return F.linear(F.gelu(F.linear(cells, W["proj_in.weight"])), W["proj_out.weight"])


class Inkling2VisionModel(Model):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.tower = Inkling2VisionTower(config)
        self.modules += [self.tower]
        self.caps.update({"image_input": True, "supports_tp": False, "autosplit_load_fwd": False})
        # The prompt form of an image, for front ends that build prompts from message parts: one placeholder per
        # image in the chat template's output, to be replaced by the embedding's alias
        self.image_placeholder = config.i2_vision["image_token"]

    @override
    def prepare_layer_map(self):
        # The config's layer map is the text model's loop; this component is one module
        self.fwd_modules = [(m, 0, idx) for idx, m in enumerate(self.modules)]
        self.last_kv_module_idx_instance = (self.last_kv_module_idx, 0)

    def preprocess(self, image: Image.Image):
        v = self.config.i2_vision
        return preprocess_image(image, v["merge"] * v["patch"], v["image_mean"], v["image_std"], v["rescale_factor"])

    def default_load_shape_dtype(self, chunk_size):
        c = self.config.i2_vision["merge"] * self.config.i2_vision["patch"]
        return (4, c, c, 3), torch.float

    def default_load_params(self, max_chunk_size):
        return {"grid": (2, 2)}

    def get_image_embeddings(self, tokenizer: Tokenizer, image: Image.Image | list, text_alias: str | None = None):
        """
        The embeddings of one image as an MMEmbedding: its alias stands for the image tokens, one per cell of
        40 x 40 pixels in raster order. The alias goes into a message of its own:
            <|message_user|><|content_image|>{alias}<|end_message|>
        """
        if isinstance(image, list):
            assert text_alias is None, "Cannot apply single alias to list of images"
            return [self.get_image_embeddings(tokenizer, i) for i in image]
        cells, (rows, cols) = self.preprocess(image)
        emb = self.tower.forward(cells, {"grid": (rows, cols)}).cpu()
        mme = MMEmbedding(
            embeddings = emb,
            text_alias = text_alias,
            token_string = torch.full((1, rows * cols), -1, dtype = torch.long),
        )
        mme.metadata.update({
            "original_size": image.size,
            "grid": (rows, cols),
            "model_architecture": self.config.architecture,
        })
        return mme

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return input_ids
