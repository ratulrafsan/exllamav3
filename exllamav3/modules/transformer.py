from __future__ import annotations
from typing_extensions import override
import torch
from ..util.tensor import to2
from ..util.device_copy import to_device
from ..model.config import Config
from . import Module, RMSNorm, LayerNorm, Attention, GatedDeltaNet, GatedMLP, MLP, BlockSparseMLP, Linear
from .hyperconnections import HyperConnection, hc_flush
from ..util import profile_opt
import os
import re

# Local patch (exl3_control_vector): llama.cpp-style control vectors on the residual stream of the
# trunk's decoder blocks, applied at the end of each block (after the MLP site's residual update, i.e.
# llama.cpp's build_cvec on the layer output).
#
#   EXL3_CVEC=<file.safetensors>   tensors "direction.<k>" ([hidden] f32), llama.cpp numbering:
#                                  direction.k applies after decoder layer k (0-based); convert a
#                                  llama.cpp GGUF with util/cvec_gguf_to_safetensors.py
#   EXL3_CVEC_MODE=project|add     project: h -= s * (h.v) v with v unit-normalized and s = |v| * scale
#                                  (the qwen4exp cvec-projection-mode patch); add: h += scale * v
#                                  (llama.cpp's stock --control-vector-scaled). Default project
#   EXL3_CVEC_SCALE=<float>        default 1.0
#   EXL3_CVEC_LAYERS=<a>-<b>       inclusive layer range, llama.cpp's --control-vector-layer-range.
#                                  Default: every layer with a direction
#
# With hyper-connections the residual is the (bsz, seq, hc, hidden) stream stack and every stream is
# projected (added to) separately, as llama.cpp's apply_to on res_hc. Only blocks keyed
# "...layers.<k>" outside the MTP head are steered; draft heads stay stock (verification is done by the
# steered trunk). Off when EXL3_CVEC is unset.

_cvec_state = None


def _cvec_table():
    global _cvec_state
    if _cvec_state is None:
        _cvec_state = {}
        path = os.environ.get("EXL3_CVEC")
        if path:
            from safetensors.torch import load_file
            mode = os.environ.get("EXL3_CVEC_MODE", "project").strip().lower()
            if mode not in ("project", "add"):
                raise ValueError(f"EXL3_CVEC_MODE must be project or add, not {mode!r}")
            scale = float(os.environ.get("EXL3_CVEC_SCALE", "1.0"))
            rng = os.environ.get("EXL3_CVEC_LAYERS")
            lo, hi = (int(t) for t in rng.split("-")) if rng else (0, 1 << 30)
            dirs = {}
            for k, t in load_file(path).items():
                m = re.fullmatch(r"direction\.(\d+)", k)
                if m and lo <= int(m.group(1)) <= hi:
                    v = t.float().flatten() * scale
                    if mode == "project":
                        n = float(v.norm())
                        if n == 0.0:
                            continue
                        dirs[int(m.group(1))] = (v / n, n)
                    else:
                        dirs[int(m.group(1))] = (v, 1.0)
            _cvec_state = {"mode": mode, "dirs": dirs}
            ls = sorted(dirs)
            print(f" -- Control vector: mode = {mode}, scale = {scale}, layers "
                  f"{ls[0] if ls else '-'}..{ls[-1] if ls else '-'} ({len(ls)} directions) from {path}")
    return _cvec_state


def _cvec_for_block(block, device):
    """(vector on device, s) for this block, or None. Cached per block and device"""
    cached = block.__dict__.get("_cvec_cache")
    if cached is not None and cached[0] == device:
        return cached[1]
    entry = None
    table = _cvec_table()
    if table:
        m = re.search(r"(?:^|\.)layers\.(\d+)$", block.key or "")
        if m and not (block.key.startswith("mtp") or ".mtp." in block.key):
            d = table["dirs"].get(int(m.group(1)))
            if d is not None:
                entry = (d[0].to(device), d[1], table["mode"])
    block._cvec_cache = (device, entry)
    return entry


def _cvec_apply(block, x, params):
    cv = _cvec_for_block(block, x.device)
    if cv is None:
        return x
    v, s, mode = cv
    # A deferred residual update (mHC) must land first, and a successor's weighted copy of the
    # streams made by apply_ (GatedResidual) is stale once they change: drop it, the next mix
    # then weights the streams in its own kernel
    hc_flush(params)
    params.pop("gr_weighted", None)
    if v.dtype != x.dtype:
        v = v.to(x.dtype)
    x2 = x.view(-1, x.shape[-1]) if x.is_contiguous() else None
    if mode == "add":
        x.add_(v)
    elif x2 is not None:
        x2.addr_(x2 @ v, v, alpha = -s)      # in place: x -= s * (x.v) v, no stream-sized temporary
    else:
        x.sub_((x @ v).unsqueeze(-1) * (s * v))
    return x

class TransformerBlock(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int | None = None,
        attn_norm: RMSNorm | LayerNorm | None = None,
        attn: Attention | GatedDeltaNet | None = None,
        attn_post_norm: RMSNorm | LayerNorm | None = None,
        mlp_norm: RMSNorm | LayerNorm | None = None,
        mlp: MLP | GatedMLP | BlockSparseMLP | None = None,
        mlp_post_norm: RMSNorm | LayerNorm | None = None,
        attn_hc: HyperConnection | None = None,
        mlp_hc: HyperConnection | None = None,
        key_layer_scalar: str | None = None,
        key_attn_resid_scalar: str | None = None,
        key_mlp_resid_scalar: str | None = None,
        qmap: str | None = None,
        qbits_key: str = "bits",
        out_dtype: torch.dtype = None
    ):
        super().__init__(config, key, None)

        self.layer_idx = layer_idx
        self.attn_norm = attn_norm
        self.attn = attn
        self.attn_post_norm = attn_post_norm
        self.mlp_norm = mlp_norm
        self.mlp = mlp
        self.mlp_post_norm = mlp_post_norm
        self.attn_hc = attn_hc
        self.mlp_hc = mlp_hc
        self.qbits_key = qbits_key
        self.out_dtype = out_dtype

        self.key_layer_scalar = key_layer_scalar
        self.key_attn_resid_scalar = key_attn_resid_scalar
        self.key_mlp_resid_scalar = key_mlp_resid_scalar
        self.layer_scalar_t = None
        self.layer_scalar_f = None
        self.attn_resid_scalar = None
        self.mlp_resid_scalar = None

        # Hyperconnection sites (mHC): the block's residual is (bsz, seq, hc_mult, hidden)
        # fp32 streams, mixed at each sublayer site instead of the plain residual add
        if attn_hc is not None or mlp_hc is not None:
            assert attn_hc is not None and mlp_hc is not None, \
                "hyperconnections require both attn_hc and mlp_hc"
            assert all(v is None for v in (
                attn_post_norm, mlp_post_norm,
                key_layer_scalar, key_attn_resid_scalar, key_mlp_resid_scalar,
            )), \
                "hyperconnections cannot combine with residual scalars/post-norms"

        self.register_submodule(self.attn_hc)
        self.register_submodule(self.attn_norm)
        self.register_submodule(self.attn)
        self.register_submodule(self.attn_post_norm)
        self.register_submodule(self.mlp_hc)
        self.register_submodule(self.mlp_norm)
        self.register_submodule(self.mlp)
        self.register_submodule(self.mlp_post_norm)

        self.num_slices = mlp.num_slices if mlp else 1


    @override
    def optimizer_targets(self):
        a = self.attn.optimizer_targets() if self.attn else []
        m = self.mlp.optimizer_targets() if self.mlp else []
        return [a, m]

    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        if self.key_layer_scalar:
            self.layer_scalar_t = self.config.stc.get_tensor(
                self.key + "." + self.key_layer_scalar,
                None,
                allow_bf16 = True,
                no_defer = True,
            )
            assert self.layer_scalar_t.numel() == 1
            self.layer_scalar_f = self.layer_scalar_t.float().item()

        # TODO: Residual scalar tensors could be baked into preceding modules for models that use them
        #       (currently only Step3.7 vision tower)
        if self.key_attn_resid_scalar:
            self.attn_resid_scalar = self.config.stc.get_tensor(
                self.key + "." + self.key_attn_resid_scalar,
                device,
                allow_bf16 = True,
                no_defer = True,
            )
        if self.key_mlp_resid_scalar:
            self.mlp_resid_scalar = self.config.stc.get_tensor(
                self.key + "." + self.key_mlp_resid_scalar,
                device,
                allow_bf16 = True,
                no_defer = True,
            )

    def unload(self):
        super().unload()
        self.layer_scalar_t = None
        self.attn_resid_scalar = None
        self.mlp_resid_scalar = None

    def get_tensors(self):
        t = {}
        if self.key_layer_scalar is not None:
            t[self.key + "." + self.key_layer_scalar] = self.layer_scalar_t.data.contiguous()
        if self.key_attn_resid_scalar is not None:
            t[self.key + "." + self.key_attn_resid_scalar] = self.attn_resid_scalar.data.contiguous()
        if self.key_mlp_resid_scalar is not None:
            t[self.key + "." + self.key_mlp_resid_scalar] = self.mlp_resid_scalar.data.contiguous()
        return t

    def weights_numel(self):
        return (
            super().weights_numel() +
            (1 if self.key_layer_scalar is not None else 0) +
            (self.attn_resid_scalar.numel() if self.attn_resid_scalar is not None else 0) +
            (self.mlp_resid_scalar.numel() if self.mlp_resid_scalar is not None else 0)
        )

    @override
    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        # A deferred mHC apply_ (hyperconnections.hc_flush) lands before its streams move to another device
        if "hc_pending" in params and x.device != self.device:
            hc_flush(params)
        return super().prepare_for_device(x, params)

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        export_state = params.get("export_state_layers")
        export_state = export_state and self.layer_idx in export_state and params.get("layer_instance", 0) == 0

        y_resid = None  # pending attn output whose residual add is folded into the MLP input norm

        # Previous block's MLP output whose residual add is folded into this block's input norm (forward_ls sets
        # resid_defer when the next module is a TransformerBlock on the same device)
        pending = params.pop("resid_pending", None)
        fuse_in = (
            pending is not None and self.attn is not None and not self.attn_hc and
            isinstance(self.attn_norm, RMSNorm) and self.attn_norm.can_fuse_residual(x, pending)
        )
        if pending is not None and not fuse_in:
            x += pending

        if self.attn:
            if self.attn_hc:
                hc_post, hc_comb, y, normed = self.attn_hc.mix_norm(x, params, self.attn_norm)
                if self.attn_norm and not normed:
                    # Normalize from the collapse's own precision (fp32 on the carry path, see
                    # HyperConnection.mix_norm) rather than a half copy of it
                    y = self.attn_norm.forward(y, params, out_dtype = torch.half)
                else:
                    y = y.half()
            elif fuse_in:
                y = self.attn_norm.forward(pending, params, out_dtype = torch.half, residual_in = x)
            elif self.attn_norm:
                y = self.attn_norm.forward(x, params, out_dtype = torch.half)
            else:
                y = x.half()
            if self.attn.device != self.device:
                # Attention pinned to the split's attention device (Model.load attn_device): its fp16
                # input and output are what crosses, the residual streams stay here
                y = self.attn.forward(self.attn.prepare_for_device(y, params), params)
                y = to_device(y, self.device)
            else:
                y = self.attn.forward(y, params)
            if params.get("prefill") and not export_state:
                return x
            if self.attn_resid_scalar is not None:
                y *= self.attn_resid_scalar
            if self.attn_hc:
                x = self.attn_hc.apply_(x, y, hc_post, hc_comb, params)
            elif self.attn_post_norm:
                self.attn_post_norm.forward(y, params, residual = x)
            elif self.mlp is not None and self.mlp_norm is not None and self.mlp_norm.can_fuse_residual(x, y):
                y_resid = y
            else:
                x += y

        if self.mlp:
            if self.mlp_hc:
                hc_post, hc_comb, y, normed = self.mlp_hc.mix_norm(x, params, self.mlp_norm)
                if self.mlp_norm and not normed:
                    # Normalize from the collapse's own precision (fp32 on the carry path, see
                    # HyperConnection.mix_norm) rather than a half copy of it
                    y = self.mlp_norm.forward(y, params, out_dtype = torch.half)
                else:
                    y = y.half()
            else:
                params["residual"] = x
                if y_resid is not None:
                    y = self.mlp_norm.forward(y_resid, params, out_dtype = torch.half, residual_in = x)
                elif self.mlp_norm:
                    y = self.mlp_norm.forward(x, params, out_dtype = torch.half)
                else:
                    y = x.half()
            y = self.mlp.forward(y, params)
            if self.mlp_resid_scalar is not None:
                y *= self.mlp_resid_scalar
            if self.mlp_hc:
                x = self.mlp_hc.apply_(x, y, hc_post, hc_comb, params)
            elif self.mlp_post_norm:
                self.mlp_post_norm.forward(y, params, residual = x)
            elif (
                params.get("resid_defer") and not export_state and self.layer_scalar_f is None and
                (out_dtype or self.out_dtype) in (None, x.dtype) and _cvec_for_block(self, x.device) is None
            ):
                params["resid_pending"] = y
                return x
            else:
                x += y
            x = _cvec_apply(self, x, params)     # local patch: control vector

        if export_state:
            s = params.get("export_states")
            if not s:
                s = params["export_states"] = []
            # With hyperconnections the residual is a stream stack; export the stream mean as
            # the collapsed hidden state (streams start as broadcast copies of the embedding)
            x_ = x.mean(dim = 2) if self.attn_hc else x
            if x_.dtype == torch.half:
                s.append(x_.clamp_(-65504.0, 65504.0))
            else:
                x_ = x_.half()
                x_.clamp_(-65504.0, 65504.0)
                s.append(x_)

        if self.layer_scalar_f is not None:
            x *= self.layer_scalar_f

        return to2(x, out_dtype, self.out_dtype)


    def get_name(self):
        name = super().get_name()
        if not self.attn and not self.mlp:
            name += " (no-op)"
        return name


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": TransformerBlock,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "out_dtype": self.out_dtype,
                "key_layer_scalar": self.key_layer_scalar,
                "key_attn_resid_scalar": self.key_attn_resid_scalar,
                "key_mlp_resid_scalar": self.key_mlp_resid_scalar,
            },
            **{name: _export(getattr(self, name, None)) for name in (
                "attn_hc",
                "attn_norm",
                "attn",
                "attn_post_norm",
                "mlp_hc",
                "mlp_norm",
                "mlp",
                "mlp_post_norm",
            )},
            # Per-layer scalars load from the tensor collection, which TP children don't have
            "layer_scalar_f": self.layer_scalar_f,
            "attn_resid_scalar": producer.send(self.attn_resid_scalar),
            "mlp_resid_scalar": producer.send(self.mlp_resid_scalar),
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        device = local_context["device"]

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        module = TransformerBlock(
            config = None,
            **exported["kwargs"],
            attn_hc = _import("attn_hc"),
            attn_norm = _import("attn_norm"),
            attn = _import("attn"),
            attn_post_norm = _import("attn_post_norm"),
            mlp_hc = _import("mlp_hc"),
            mlp_norm = _import("mlp_norm"),
            mlp = _import("mlp"),
            mlp_post_norm = _import("mlp_post_norm"),
        )

        module.layer_scalar_f = exported.get("layer_scalar_f")
        module.attn_resid_scalar = consumer.recv(exported.get("attn_resid_scalar"), cuda = True)
        module.mlp_resid_scalar = consumer.recv(exported.get("mlp_resid_scalar"), cuda = True)
        module.device = device
        return module


class ParallelDecoderBlock(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int | None = None,
        input_norm: RMSNorm | LayerNorm | None = None,
        attn: Attention | None = None,
        mlp: MLP | GatedMLP | None = None,
        qmap: str | None = None,
        qbits_key: str = "bits",
        out_dtype: torch.dtype = None
    ):
        super().__init__(config, key, None)

        self.layer_idx = layer_idx
        self.input_norm = input_norm
        self.attn = attn
        self.mlp = mlp
        self.qbits_key = qbits_key
        self.out_dtype = out_dtype

        self.register_submodule(self.input_norm)
        self.register_submodule(self.attn)
        self.register_submodule(self.mlp)

        self.num_slices = mlp.num_slices if mlp else 1

        self.tp_reduce = False


    @override
    def optimizer_targets(self):
        a = self.attn.optimizer_targets() if self.attn else []
        m = self.mlp.optimizer_targets() if self.mlp else []
        return [a, m]


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        y = self.input_norm.forward(x, params, out_dtype = torch.half)
        y1 = self.attn.forward(y, params)
        if not params.get("prefill"):
            y2 = self.mlp.forward(y, params)
            y1 += y2

            if self.tp_reduce:
                self.tp_collect(params["backend"], y1)

            x += y1

        return to2(x, out_dtype, self.out_dtype)


    def get_name(self):
        name = super().get_name()
        if not self.attn and not self.mlp:
            name += " (no-op)"
        return name


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": ParallelDecoderBlock,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "out_dtype": self.out_dtype,
            },
            **{name: _export(getattr(self, name, None)) for name in (
                "input_norm",
                "attn",
                "mlp",
            )},
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan):
        device = local_context["device"]

        def _import(name, **kwargs):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan, **kwargs) \
                if exported.get(name) else None

        module = ParallelDecoderBlock(
            config = None,
            **exported["kwargs"],
            input_norm = _import("input_norm"),
            attn = _import("attn", skip_reduction = True),
            mlp = _import("mlp", skip_reduction = True),
        )
        module.device = device

        # Use single reduction for sum of mlp and attn
        module.tp_reduce = True
        module.tp_owner = module.tp_single_owner(
            local_context, exported["attn"]["kwargs"]["key"], exported["mlp"]["kwargs"]["key"])
        return module
