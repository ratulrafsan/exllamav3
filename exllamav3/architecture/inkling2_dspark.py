from __future__ import annotations
from typing_extensions import override
import weakref
import torch
import torch.nn.functional as F

from ..cache import Cache
from ..model.config import Config
from ..model.model import Model
from ..modules import Module, Linear, RMSNorm, TransformerBlock, Attention, GatedMLP
from ..modules import inkling2 as I2
from ..modules.attn import prepare_for_attn
from ..util.device_copy import to_device
from ..util.tensor import get_for_device
from .dflash import DFlashConfig

"""
Local patch: the draft model of Inkling-2-Mini (`mtp.safetensors`, "DSpark"), a DFlash-family block drafter with a
recurrent Markov head. It plugs into the generator's DFlash drafting (caps "dflash_draft").

  * A block is the anchor (the last committed token, through the target's normed embedding) and block_size - 1
    learned mask rows. Three dense pre-norm layers (RoPE, RMS-normed Q and K, SiLU MLP) run over the block, with
    attention not causal inside it: layers 0 and 2 see a window of 1024 positions, layer 1 everything.
  * The context those layers attend to comes from the target, not from draft tokens: the target exports the hidden
    state at the input of layer applications 5, 23 and 41 and its final normed hidden state (taps); per draft layer
    the taps are mixed (softmax of a row of `fuse_weight`), RMS-normed, and projected to that layer's context K and
    V (`k_proj_ctx`, `v_proj_ctx`).
  * Head: base logits = the target's unembedding of the block's hidden rows. A small recurrent state walks the
    drafted positions: from (state, a rank-256 embedding of the previous token, the row's hidden state, the mixed
    taps of the last committed position) it updates the state and adds a rank-256 correction to the logits, which
    are soft-capped like the target's. Each draft is the argmax, fed to the next step.

The math follows the reference implementations in 0xSero/inkling2-exl3 (inkling2_draft.py, MIT) and the notes in
ashalliants/ninfer-3090 (docs/maintainer/inkling2-model.md, Apache 2.0).

The model directory is made by util/inkling2/make_dspark_dir.py: a config.json and the checkpoint's mtp.safetensors.

Current limits: unquantized; every draft layer keeps its K/V in the paged cache (the sliding-window layers too);
the Markov chain is plain PyTorch; batch rows are told apart by the first cache page of their sequence.
"""


class Inkling2DSparkConfig(DFlashConfig):
    arch_string = "Inkling2DSparkDraftModel"

    # Taps are layer-application ("visit") numbers of the target, used as they are
    tap_shift = 0

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, {"text": Inkling2DSparkModel}, **kwargs)
        self.markov_rank = self.read_cfg(int, "markov_rank", 256)
        self.final_logit_softcapping = self.read_cfg(float, "final_logit_softcapping", 30.0)
        self.unpadded_vocab_size = self.read_cfg(int, "unpadded_vocab_size", self.vocab_size)
        self.is_causal = False


class Inkling2DSparkParts(Module):
    """
    Everything of the drafter outside its three transformer blocks: the block input (anchor embedding and mask
    rows), the tap mixes with the context K/V projections, and the Markov head. First module of the model.
    """

    def __init__(self, config: Inkling2DSparkConfig):
        super().__init__(config, "dspark", None)
        n = config.num_hidden_layers
        kv = config.num_kv_heads * config.head_dim
        mk = lambda key: Linear(config, key, config.hidden_size, kv, out_dtype = torch.half, pad_to = 1)
        self.k_ctx = [mk(f"layers.{i}.self_attn.k_proj_ctx") for i in range(n)]
        self.v_ctx = [mk(f"layers.{i}.self_attn.v_proj_ctx") for i in range(n)]
        for l in self.k_ctx + self.v_ctx:
            self.register_submodule(l)
        self.attached_model = None
        self.caps.update({"x_cpu": True})
        self.anchor = None
        self.f_rnn = {}                 # first cache page of a sequence -> mixed taps of its last committed position
        self.fold = None

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        g = lambda k: self.config.stc.get_tensor(k, device, allow_bf16 = True, no_defer = True).float()
        n = self.config.num_hidden_layers
        self.mask_embed = g("mask_embed")
        self.alpha = torch.softmax(g("fuse_weight"), dim = -1)                      # [layers + 1, taps]
        self.dflare = [g(f"layers.{i}.dflare_norm.weight") for i in range(n)]
        self.k_norm_ctx = [g(f"layers.{i}.self_attn.k_norm_ctx.weight").half() for i in range(n)]
        self.m_dflare = g("markov_head.dflare_norm.weight")
        self.w1 = g("markov_head.markov_w1.weight").half()                          # [vocab, rank]
        self.rnn_w, self.rnn_b = g("markov_head.rnn_proj.weight"), g("markov_head.rnn_proj.bias")
        self.out_w, self.out_b = g("markov_head.hidden_out.weight"), g("markov_head.hidden_out.bias")

    @override
    def unload(self):
        for name in ("mask_embed", "alpha", "dflare", "k_norm_ctx", "m_dflare", "w1", "rnn_w", "rnn_b", "out_w", "out_b",
                     "fold", "anchor"):
            setattr(self, name, None)
        self.f_rnn = {}
        super().unload()

    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        return x

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        # x: the anchor token ids [bsz, 1]. Block input: the anchor through the target's normed embedding, then
        # block_size - 1 mask rows
        target = self.attached_model()
        a = target.embed.lookup(x, torch.float)
        a = to_device(a, self.device)
        self.anchor = x.to(self.device)
        masks = self.mask_embed.view(1, 1, -1).expand(x.shape[0], self.config.block_size - 1, -1)
        return torch.cat([a, masks], dim = 1)

    def mix(self, taps: torch.Tensor, row: int, w: torch.Tensor) -> torch.Tensor:
        """taps: fp32 [..., taps, D] -> RMS norm (gain w) of their mix by row `row` of the fuse table, fp32"""
        m = (taps * self.alpha[row].view(-1, 1)).sum(-2)
        return I2.rms_norm(m, w, self.config.rms_norm_eps)

    def head_fold(self):
        """The Markov head's output projection folded into the target's unembedding: W_head Wo [V, rank], W_head bo"""
        if self.fold is None:
            head = self.attached_model().head.Hd.lm_head                             # [V, D], on the target's device
            w = torch.empty((head.shape[0], self.out_w.shape[1]), dtype = torch.half, device = head.device)
            b = torch.empty((head.shape[0],), dtype = torch.float, device = head.device)
            ow, ob = self.out_w.to(head.device), self.out_b.to(head.device)
            for s in range(0, head.shape[0], 8192):
                hd = head[s : s + 8192].float()
                w[s : s + 8192] = (hd @ ow).half()
                b[s : s + 8192] = hd @ ob
            # The chain runs on the head's device; its tables go there once (not on every round)
            self.fold = (head, w, b, self.rnn_w.to(head.device), self.rnn_b.to(head.device), self.w1.to(head.device))
        return self.fold


class Inkling2DSparkModel(Model):
    config_class = Inkling2DSparkConfig

    def __init__(self, config: Inkling2DSparkConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.parts = Inkling2DSparkParts(config)
        self.modules += [self.parts]
        self.first_block_idx = len(self.modules)

        self.attn_modules = []
        for idx in range(config.num_hidden_layers):
            window_left, window_right = config.block_window(idx)
            norm = lambda name: RMSNorm(config = config, key = f"layers.{idx}.{name}", rms_norm_eps = config.rms_norm_eps)
            attn = Attention(
                config = config,
                key = f"layers.{idx}.self_attn",
                layer_idx = idx,
                hidden_size = config.hidden_size,
                head_dim = config.head_dim,
                num_q_heads = config.num_q_heads,
                num_kv_heads = config.num_kv_heads,
                rope_settings = config.rope_settings,
                key_q = "q_proj", key_k = "k_proj", key_v = "v_proj", key_o = "o_proj",
                sliding_window = window_left,
                window_right = window_right,
                q_norm = norm("self_attn.q_norm"),
                k_norm = norm("self_attn.k_norm"),
                out_dtype = torch.float,
            )
            self.attn_modules.append(attn)
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"layers.{idx}",
                    layer_idx = idx,
                    attn_norm = norm("input_layernorm"),
                    attn = attn,
                    mlp_norm = norm("post_attention_layernorm"),
                    mlp = GatedMLP(
                        config = config,
                        key = f"layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "up_proj", key_gate = "gate_proj", key_down = "down_proj",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                    ),
                )
            ]
        self.last_kv_module_idx = len(self.modules) - 1
        self.modules += [RMSNorm(config = config, key = "norm", rms_norm_eps = config.rms_norm_eps, out_dtype = torch.half)]
        self.logit_layer_idx = None

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "dflash_draft": True,
            "default_draft_size": config.block_size - 1,
            "autosplit_load_fwd": False,
        })
        self.recurrent_state_cls = None
        self.attached_model = None
        self.draft_verifier_params.update({"export_state_layers": set(config.target_layer_ids)})

    def attach_to(self, target):
        assert hasattr(target.config, "i2"), "the DSpark drafter attaches to an Inkling-2 target"
        self.attached_model = weakref.ref(target)
        self.parts.attached_model = weakref.ref(target)

    def update_kv_from_target(self, target_hidden: list, cache: Cache, params: dict, lengths: list[int] = None):
        """
        Write the context K/V of positions the target has committed. target_hidden: the taps, in forward order,
        each [bsz, T, D]; they sit at positions cache_seqlens .. + T (lengths: rows of each batch row that count).
        """
        parts, cfg = self.parts, self.config
        if lengths is not None:
            n = max(lengths)
            if n <= 0:
                return
            target_hidden = [t[:, :n] for t in target_hidden]
        dev = parts.device
        taps = torch.stack([to_device(t, dev).float() for t in target_hidden], dim = 2)       # [bsz, T, taps, D]
        bsz, T = taps.shape[:2]

        # The Markov head's feature: the mix of the last committed position, per sequence
        first_pages = params["block_table"][:, 0].tolist()
        for b in range(bsz):
            last = (lengths[b] if lengths is not None else T) - 1
            if last >= 0:
                parts.f_rnn[int(first_pages[b])] = parts.mix(taps[b, last], cfg.num_hidden_layers, parts.m_dflare)

        for i, layer in enumerate(self.attn_modules):
            c = parts.mix(taps, i, parts.dflare[i]).half()
            c = to_device(c, layer.device)
            block_table = get_for_device(params, "block_table", layer.device)
            cache_seqlens = get_for_device(params, "cache_seqlens", layer.device)
            k = parts.k_ctx[i].forward(to_device(c, parts.k_ctx[i].device), params)
            v = parts.v_ctx[i].forward(to_device(c, parts.v_ctx[i].device), params)
            k = to_device(k, layer.device).view(bsz, T, layer.num_kv_heads, layer.head_dim)
            v = to_device(v, layer.device).view(bsz, T, layer.num_kv_heads, layer.head_dim)
            # Head RMS norm (the context's own gain) and RoPE at the rows' positions
            k, _ = layer.rope.apply(
                k, None, 0, cache_seqlens, None, True,
                to_device(parts.k_norm_ctx[i], layer.device), None, layer.norm_eps, layer.norm_constant_bias, None,
            )
            cache.update_layer_direct(layer.layer_idx, cache_seqlens, block_table, k, v, T, 0)

    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        """
        state: the block's final hidden rows [bsz, block, D]. Returns ids [bsz, block]: the anchor, then the
        drafts from the Markov chain over the target's head.
        """
        parts, cfg = self.parts, self.config
        head, fold_w, fold_b, rnn_w, rnn_b, w1 = parts.head_fold()
        hdev = head.device
        bsz = state.shape[0]
        K = cfg.block_size - 1
        R = cfg.markov_rank
        hid = to_device(state[:, 1:], hdev).half()                                    # [bsz, K, D]
        base = torch.matmul(hid, head.t()).float()                                    # [bsz, K, V]
        first_pages = params["block_table"][:, 0].tolist()
        f = torch.stack([parts.f_rnn[int(p)] for p in first_pages]).to(hdev)          # [bsz, D]
        # The part of the recurrence input that does not depend on the chain: the row's hidden state and f
        ctx = torch.cat([hid.float(), f.unsqueeze(1).expand(-1, K, -1)], dim = -1) @ rnn_w[:, 2 * R:].t() + rnn_b
        w_r = rnn_w[:, : 2 * R].t()
        cap = cfg.final_logit_softcapping
        V = cfg.unpadded_vocab_size
        st = torch.zeros((bsz, R), dtype = torch.float, device = hdev)
        prev = parts.anchor.to(hdev).view(bsz)
        out = [prev]
        export = bool(params.get("export_draft_conf"))
        conf = [torch.zeros((bsz,), dtype = torch.float, device = hdev)]
        for i in range(K):
            raw = ctx[:, i] + torch.cat([st, w1[prev].float()], dim = -1) @ w_r
            gate, cand, o = raw.chunk(3, dim = -1)
            gate = torch.sigmoid(gate)
            st = gate * st + (1.0 - gate) * torch.tanh(cand)
            z = base[:, i] + torch.tanh(o).half().matmul(fold_w.t()).float() + fold_b
            z = torch.tanh(z / cap) * cap
            if export:
                # Confidence score for the generator's calibrated draft length: the margin between the two
                # best logits (the logits are soft-capped, so the winning logit alone says little)
                top = z[:, :V].topk(2, dim = -1)
                prev = top.indices[:, 0]
                conf.append(top.values[:, 0] - top.values[:, 1])
            else:
                prev = z[:, :V].argmax(dim = -1)
            out.append(prev)
        if export:
            params["draft_conf"] = torch.stack(conf, dim = 1)
        return torch.stack(out, dim = 1)

    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long

    def default_load_params(self, max_chunk_size):
        return {}

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # Attention is not causal inside the block; the windowed layers carry their own bounds
        params["causal"] = False
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError()
