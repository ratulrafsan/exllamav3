from __future__ import annotations
from typing_extensions import override
import json
import os
import weakref
import torch

from ..model.config import Config
from ..model.model import Model
from ..modules import Module, BlockSparseMLP
from ..modules import inkling2 as I2
from ..modules.sliding_attn import SWAState
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence, advance_recurrent_states
from ..cache.recurrent import host_copy
from ..cache.fp16 import CacheLayer_fp16
from ..constants import PAGE_SIZE
from ..tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX

"""
Local patch: Inkling-2-Mini (thinkingmachines/Inkling-2-Mini), the looped Inkling-2 text model, loaded from the
source checkpoint. See modules/inkling2.py for the math.

Three module kinds:

    Inkling2Embed      token embedding + embed_norm; starts the per-forward state
    Inkling2Layer      one decoder layer: attention weights (as stored), the routed experts in the packed form of
                       the checkpoint (NVFP4 / MXFP8, dequantized per routed expert), and for loop layers the
                       message gates and queries
    Inkling2Head       output attention residual, final norm, unembedding, logit soft cap

The loop is expressed with the model's layer map: layers [loop_start, loop_end) appear `passes` times in the forward
order, and the forward loop hands each application its pass number as params["layer_instance"]. The weights are loaded
once.

State that the layers share during one forward lives in params["i2"]: per batch row, the block states and the loop
bookkeeping (I2.LoopState). The hidden state [bsz, seq, hidden] is the tensor passed from module to module.

Cache (params["cache"] given): one cache layer per layer application, so every loop pass has its own.

    global layers      K/V of every token in the paged cache (CacheLayer_fp16), like any full-attention layer
    sliding layers     K/V of the last tokens in a per-slot buffer with ExLlamaV3's sliding-window layout: row =
                       position - state.window_beg, whole pages dropped from the front when it fills up
    all layers         the raw inputs of the four short convolutions (K, V, attention output, MLP output) of the
                       last CONV_RING positions, in a ring indexed by position

The per-sequence state object is Inkling2State (an SWAState): it rolls back in place by moving its position, as
far as both the sliding K/V buffer and the convolution ring still hold the rows.

Current limits: text only, fp16 K/V cache only, plain PyTorch math one batch row at a time (slow), no tensor
parallelism, no speculative decoding.
"""

# Positions of convolution inputs kept per slot and layer application. A rollback needs the conv_k - 1 inputs before
# the position it returns to, so this bounds in-place rollback; SWAState promises one page of it
CONV_RING = PAGE_SIZE + 32
CONV_NAMES = ("k", "v", "attn", "mlp")
# Storage dtype of the convolution ring. fp32 is for tests (with EXL3_I2_ADT=fp32 and I2.ROUND_KV a cached pass then
# reproduces an uncached one)
_CONV_DT = torch.float if os.environ.get("EXL3_I2_CONV_FP32") else torch.half


def ring_shift(cache_pos: int, n: int, size: int) -> int:
    """Pages (in rows) to drop from the front of a sliding buffer of `size` rows before n rows go in at cache_pos"""
    return 0 if cache_pos + n <= size else (cache_pos + n - size + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE


class Inkling2State(SWAState):
    """
    SWAState plus the reach of the convolution ring. `high` is the furthest position the ring was written to: the
    ring holds the inputs of the CONV_RING positions before it, whatever the state's position is after a rollback.
    """

    def __init__(self, cache, slot: int, position: int, clear: bool = True, stashed: dict = None,
                 test_state: bool = False):
        self.high = position
        super().__init__(cache, slot, position, clear = clear, stashed = stashed, test_state = test_state)

    def _conv_floor(self) -> int:
        """Lowest position the state can return to with its convolution inputs still in the ring"""
        return max(0, self.high - CONV_RING + 8)

    def rollback_capacity(self):
        return max(0, min(self.position - self.window_beg, self.position - self._conv_floor()))

    def rewind(self, num_tokens: int):
        assert num_tokens <= self.rollback_capacity(), \
            f"Inkling2State: cannot rewind {num_tokens} tokens, capacity is {self.rollback_capacity()}"
        super().rewind(num_tokens)

    def can_stash_at(self, position: int) -> bool:
        return position >= self._conv_floor() and super().can_stash_at(position)

    def stash_layers(self, layers: dict):
        cache_pos = self.position - self.window_beg
        return {k: l.stash(self.slot, cache_pos, self.position) for k, l in layers.items()}

    def unstash(self, stashed: dict):
        assert self.position == stashed["position"]
        self.window_beg = stashed["window_beg"]
        self.high = self.position
        cache_pos = self.position - self.window_beg
        for k, l in self.cache.get_all_recurrent_layers().items():
            l.unstash(self.slot, stashed[k], cache_pos, self.position)

    def post_advance(self):
        super().post_advance()
        self.high = max(self.high, self.position)

    def reset(self):
        super().reset()
        self.high = 0


class Inkling2LayerState:
    """
    Per layer application: for every state slot, the sliding K/V buffer (sliding layers only) and the ring of raw
    convolution inputs. fp16.
    """

    def __init__(self, module: "Inkling2Layer", max_batch_size: int, max_history: int, cache_id: int):
        c = module.config.i2
        self.module = module
        self.local = not module.is_global
        self.size = module.kv_state_size
        self.window = c.window
        self.taps = c.conv_k - 1
        self.max_batch_size, self.max_history, self.cache_id = max_batch_size, max_history, cache_id
        kv = c.kv_heads * c.head_dim
        meta = lambda *shape, dt = torch.half: torch.empty(shape, dtype = dt, device = "meta")
        self.k_state = meta(max_batch_size, self.size, c.kv_heads, c.head_dim) if self.local else None
        self.v_state = meta(max_batch_size, self.size, c.kv_heads, c.head_dim) if self.local else None
        self.conv = {
            n: meta(max_batch_size, CONV_RING, kv if n in ("k", "v") else c.hidden, dt = _CONV_DT)
            for n in CONV_NAMES
        }
        self.device = None

    def _tensors(self):
        return ([self.k_state, self.v_state] if self.local else []) + [self.conv[n] for n in CONV_NAMES]

    def get_checkpoint_size(self):
        kv = self.window * self.k_state[0, 0].numel() * 2 * 2 if self.local else 0
        return kv + sum(self.taps * self.conv[n].shape[-1] * _CONV_DT.itemsize for n in CONV_NAMES)

    def storage_size(self):
        return sum(t.numel() * t.element_size() for t in self._tensors())

    def alloc(self, device):
        if self.local:
            self.k_state = torch.zeros_like(self.k_state, device = device)
            self.v_state = torch.zeros_like(self.v_state, device = device)
        self.conv = {n: torch.zeros_like(t, device = device) for n, t in self.conv.items()}
        self.device = device

    def free(self):
        if self.local:
            self.k_state = torch.empty_like(self.k_state, device = "meta")
            self.v_state = torch.empty_like(self.v_state, device = "meta")
        self.conv = {n: torch.empty_like(t, device = "meta") for n, t in self.conv.items()}
        self.device = None

    def clear(self, idx: int):
        pass

    def get_state_tensors(self):
        return tuple(self._tensors())

    def rewind(self, slot: int, last_history: int, num_tokens: int):
        pass

    # Convolution ring: the input of position p is row p % CONV_RING

    def conv_hist(self, slot: int, position: int, n_new: int = 1) -> dict:
        """
        The conv_k - 1 inputs before `position`, zeros before the start of the sequence. n_new: rows the caller
        will write at `position` while it still reads the result
        """
        t = self.taps
        r0 = (position - t) % CONV_RING
        if position >= t and r0 + t <= CONV_RING and n_new + t <= CONV_RING:
            # The common case: the rows are contiguous in the ring and the new rows do not wrap around onto them;
            # read-only views
            return {n: self.conv[n][slot, r0 : r0 + t] for n in CONV_NAMES}
        rows = [p % CONV_RING for p in range(max(0, position - t), position)]
        out = {}
        for n in CONV_NAMES:
            h = torch.zeros((t, self.conv[n].shape[-1]), dtype = _CONV_DT, device = self.device)
            if rows:
                h[t - len(rows):] = self.conv[n][slot, rows]
            out[n] = h
        return out

    def conv_write(self, slot: int, position: int, name: str, x: torch.Tensor):
        """Inputs of positions position .. position + len(x) - 1; only the last CONV_RING of them are kept"""
        n = x.shape[0]
        if n == 1:
            self.conv[name][slot, position % CONV_RING] = x[0]
            return
        skip = max(0, n - CONV_RING)
        rows = torch.arange(position + skip, position + n, device = self.device) % CONV_RING
        self.conv[name][slot, rows] = x[skip:].to(_CONV_DT)

    # Sliding K/V buffer: the row of position p is p - window_beg

    def kv_hist(self, slot: int, cache_pos: int) -> tuple:
        """K/V rows a query at cache_pos (and later) can still see"""
        a = max(0, cache_pos - (self.window - 1))
        return self.k_state[slot, a:cache_pos], self.v_state[slot, a:cache_pos]

    def kv_write(self, slot: int, cache_pos: int, k: torch.Tensor, v: torch.Tensor):
        """Append n rows at cache_pos, dropping whole pages from the front when they do not fit (ring_shift)"""
        n = k.shape[0]
        shift = ring_shift(cache_pos, n, self.size)
        for state, new in ((self.k_state, k), (self.v_state, v)):
            new = new.to(torch.half)
            if shift == 0:
                state[slot, cache_pos : cache_pos + n] = new
                continue
            n_keep = cache_pos - shift
            if n_keep > 0:
                state[slot, :n_keep] = state[slot, shift : cache_pos].clone()
                state[slot, n_keep : n_keep + n] = new
            else:
                state[slot, : n + n_keep] = new[-n_keep:]

    # Checkpoints: the attention window (sliding layers) and the convolution inputs before the position

    def stash(self, slot: int, cache_pos: int, position: int):
        out = []
        if self.local:
            a = max(0, cache_pos - self.window)
            out += [host_copy(self.k_state[slot, a:cache_pos]), host_copy(self.v_state[slot, a:cache_pos])]
        hist = self.conv_hist(slot, position)
        out += [host_copy(hist[n]) for n in CONV_NAMES]
        return tuple(out)

    def unstash(self, slot: int, stashed, cache_pos: int, position: int):
        stashed = list(stashed)
        if self.local:
            k, v = stashed[:2]
            stashed = stashed[2:]
            a = max(0, cache_pos - self.window)
            self.k_state[slot].zero_()
            self.v_state[slot].zero_()
            self.k_state[slot, a:cache_pos].copy_(k)
            self.v_state[slot, a:cache_pos].copy_(v)
        t = self.taps
        rows = [p % CONV_RING for p in range(position - t, position) if p >= 0]
        for n, h in zip(CONV_NAMES, stashed):
            if rows:
                self.conv[n][slot, rows] = h[t - len(rows):].to(self.device)


# How EXL3 experts run: "kernel" = ExLlamaV3's BlockSparseMLP (its router and fused expert kernels), "dense" = the
# reference loop of modules/inkling2.py over the per-expert EXL3 linears (slow; for comparing the two)
_MOE_MODE = os.environ.get("EXL3_I2_MOE", "kernel")


_kernel_act_ok = None


def kernel_has_inkling_act(device) -> bool:
    """
    Whether the loaded extension's expert kernels have the Inkling-2 activation (a local patch to the extension
    sources: swiglu_oai with a negative limit). A stock build would run the gpt-oss activation instead, silently.
    Tested by calling the activation, once per process.
    """
    global _kernel_act_ok
    if _kernel_act_ok is None:
        from ..ext import exllamav3_ext as ext
        g = torch.linspace(-9.0, 9.0, 256, device = device).half().view(1, 256).contiguous()
        u = torch.linspace(40.0, -40.0, 256, device = device).half().view(1, 256).contiguous()
        z = torch.empty_like(g)
        try:
            ext.silu_oai_mul(g, u, z, -1.0)
            ref = I2.softcap(g.float(), 4.0) * torch.sigmoid(g.float()) * I2.softcap(u.float(), 25.0)
            _kernel_act_ok = bool(torch.allclose(z.float(), ref, rtol = 2e-2, atol = 2e-2))
        except Exception:
            _kernel_act_ok = False
        if not _kernel_act_ok:
            print(" !! Inkling2: this exllamav3 extension lacks the Inkling-2 expert activation (build it from the "
                  "patched sources). Falling back to the slow per-expert path.")
    return _kernel_act_ok


class Exl3Experts:
    """
    Routed experts of one layer converted to EXL3 (inkling/convert_exl3.py): one Linear per expert and projection,
    `{layer}.mlp.experts.{e}.gate_proj / up_proj / down_proj`, held by a BlockSparseMLP. Same interface as
    I2.PackedExperts, plus `moe` (kernel mode), which runs router and experts in one call.
    """

    def __init__(self, mlp: BlockSparseMLP):
        self.mlp = mlp
        self.params = {}
        if _MOE_MODE == "kernel" and kernel_has_inkling_act(mlp.device):
            self.moe = self._moe

    @staticmethod
    def present(stc, layer: int) -> bool:
        return stc.has_tensor(f"model.llm.layers.{layer}.mlp.experts.0.gate_proj.trellis")

    @staticmethod
    def make_mlp(config, layer: int) -> BlockSparseMLP:
        c = config.i2
        return BlockSparseMLP(
            config = config,
            key = f"model.llm.layers.{layer}.mlp",
            hidden_size = c.hidden,
            intermediate_size = c.inter,
            num_experts = c.experts,
            num_experts_per_tok = c.topk,
            key_up = "experts.{expert_idx}.up_proj",
            key_gate = "experts.{expert_idx}.gate_proj",
            key_down = "experts.{expert_idx}.down_proj",
            key_routing_gate = "gate",
            # Sigmoid scores; the bias only takes part in the top-k selection; weights normalized, then scaled
            router_type = "dots",
            key_e_score_bias = "gate.bias",
            routed_scaling_factor = c.route_scale,
            # swiglu_oai with a negative limit selects the Inkling-2 activation in the kernels (local patch):
            # softcap(gate, 4) * sigmoid(gate) * softcap(up, 25)
            activation_fn = "swiglu_oai",
            act_limit = -1.0,
            interm_dtype = torch.half,
            out_dtype = torch.float,
        )

    def _moe(self, x: torch.Tensor, adt: torch.dtype | None) -> torch.Tensor:
        # adt None: the kernels' fp32 output as it is (the caller rounds it)
        y = self.mlp.forward(x.to(torch.half).unsqueeze(0).contiguous(), self.params, torch.float)
        return y[0] if adt is None else y[0].to(adt)

    def gate_up(self, e, x):
        xh = x.to(torch.half).contiguous()
        g = self.mlp.gates[e].forward(xh, self.params, torch.half)
        u = self.mlp.ups[e].forward(xh, self.params, torch.half)
        return torch.cat([g, u], dim = -1)

    def down(self, e, a):
        return self.mlp.downs[e].forward(a.to(torch.half).contiguous(), self.params, torch.float)


class _CacheView:
    """The cache of one layer application for one batch row, as modules/inkling2.py expects it"""

    def __init__(self, rsl: Inkling2LayerState, slot: int, position: int, cache_pos: int,
                 cl: CacheLayer_fp16 | None, pages: list | None, n_new: int):
        self.rsl, self.slot, self.position, self.cache_pos = rsl, slot, position, cache_pos
        self.cl, self.pages = cl, pages
        self.conv = rsl.conv_hist(slot, position, n_new)

    def record(self, name: str, x: torch.Tensor):
        self.rsl.conv_write(self.slot, self.position, name, x)

    def extend(self, k: torch.Tensor, v: torch.Tensor, adt: torch.dtype):
        n = k.shape[0]
        if self.cl is None:
            hk, hv = self.rsl.kv_hist(self.slot, self.cache_pos)
            q_off = hk.shape[0]
            k_all = torch.cat([hk.to(adt), k], 0) if q_off else k
            v_all = torch.cat([hv.to(adt), v], 0) if q_off else v
            self.rsl.kv_write(self.slot, self.cache_pos, k, v)
            return k_all, v_all, q_off

        # Paged cache: write the new rows, then read the sequence back through its pages
        P, total = self.position, self.position + n
        npg = (total + PAGE_SIZE - 1) // PAGE_SIZE
        pages = self.pages                                         # host list of this row's page numbers
        first = pages[0]
        contiguous = all(pages[j] == first + j for j in range(npg))
        if n == 1:
            rows = pages[P // PAGE_SIZE] * PAGE_SIZE + P % PAGE_SIZE
        else:
            pos = torch.arange(P, total, device = k.device)
            pt = torch.tensor(pages[:npg], dtype = torch.long, device = k.device)
            rows = pt[pos // PAGE_SIZE] * PAGE_SIZE + pos % PAGE_SIZE
        out = []
        for store, new in ((self.cl.k, k), (self.cl.v, v)):
            flat = store.view(-1, *store.shape[2:])
            flat[rows] = new.to(torch.half) if n > 1 else new[0].to(torch.half)
            if contiguous:
                seq = flat[first * PAGE_SIZE : first * PAGE_SIZE + total]
            else:
                pt = torch.tensor(pages[:npg], dtype = torch.long, device = k.device)
                seq = store[pt].view(-1, *store.shape[2:])[:total]
            out.append(seq.to(adt))
        return out[0], out[1], P

# Activation dtype between and inside the layers. fp16 GEMMs are several times more accurate than bf16 ones, and the
# stored bf16 weights convert to fp16 without loss at their magnitudes. bf16 reproduces the transformers reference.
_ADT = {"fp16": torch.half, "bf16": torch.bfloat16, "fp32": torch.float}[os.environ.get("EXL3_I2_ADT", "fp16")]


# Decode steps (one sequence, one token, cached) replayed as CUDA graphs, one per run of layer applications on the
# same GPU. 0 runs every step op by op
_GRAPH = os.environ.get("EXL3_I2_GRAPH", "1") != "0"
_EMBED_GPU = os.environ.get("EXL3_I2_EMBED", "cpu") == "gpu"
# Capacity of the global layers' key buffer inside a graph grows in steps of this many tokens (a new step = a new
# capture); keys past the current position are masked
_GRAPH_KEY_STEP = 4096
# Positions before this run op by op (the convolution history is not full yet)
_GRAPH_MIN_POS = 8
# Widest step that runs as a graph (tokens of one sequence in one forward: the token to decode plus drafted tokens to
# verify), and how many widths keep a captured graph at a time
_GRAPH_MAX_WIDTH = int(os.environ.get("EXL3_I2_GRAPH_WIDTH", 8))
_STATS = int(os.environ.get("EXL3_I2_STATS", 0))
_GRAPH_MAX_GRAPHS = 4
_GRAPH_WIDTHS = (1, 2, 4, 8)
# Debug: report non-finite logits of any forward (costs a device sync per forward)
_NAN_CHECK = os.environ.get("EXL3_I2_NAN_CHECK", "0") != "0"


class Inkling2LoopConfig(Config):
    arch_string = "Inkling2LoopForConditionalGeneration"

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, {"text": Inkling2LoopModel}, config_key = "text_config", **kwargs)
        self.i2 = I2.Inkling2Cfg.from_dict(self.config_dict)
        c = self.i2
        self.num_hidden_layers = c.layers
        self.hidden_size = c.hidden
        self.head_dim = c.head_dim
        self.num_q_heads = c.heads
        self.num_kv_heads = c.kv_heads
        self.rms_norm_eps = c.eps
        self.num_experts = c.experts
        self.num_experts_per_tok = c.topk
        self.moe_intermediate_size = c.inter
        self.sliding_window = c.window
        self.max_position_embeddings = self.read_cfg(int, "model_max_length", 1048576)
        # The unembedding is stored padded; the model's logits cover the unpadded vocabulary
        self.vocab_size = c.unpadded_vocab
        self.activation_dtype = _ADT

        # The loop as a layer map (a user-supplied map would break the pass structure)
        assert self.layer_map is None and self.layer_map_str is None, \
            "Inkling2Loop defines its own layer map (the loop passes); layer_map cannot be combined with it"
        self.layer_map = [layer for layer, _ in c.applications]
        # Position of (layer, pass) in the forward order: the "visit" numbers a drafter's taps count
        self.app_index = {lp: n for n, lp in enumerate(c.applications)}

        # The vision tower, when the model directory has its tensors: the "vision" component
        # (architecture/inkling2_vision.py)
        self.i2_vision = None
        if self.stc.has_tensor("model.visual.proj_out.weight"):
            with open(os.path.join(directory, "config.json"), encoding = "utf8") as f:
                vc = json.load(f).get("vision_config") or {}
            pp = {}
            pp_path = os.path.join(directory, "processor_config.json")
            if os.path.exists(pp_path):
                with open(pp_path, encoding = "utf8") as f:
                    pp = json.load(f)
            ip = pp.get("image_processor") or {}
            first = lambda v, d: (v[0] if isinstance(v, list) else v) if v is not None else d
            self.i2_vision = {
                "hidden_size": vc.get("hidden_size", 1024),
                "num_hidden_layers": vc.get("num_hidden_layers", 23),
                "num_attention_heads": vc.get("num_attention_heads", 16),
                "patch": vc.get("native_patch_size", 20),
                "merge": vc.get("adapter_merge_size", 2),
                "posemb_grid_size": vc.get("posemb_grid_size", 32),
                "layer_norm_eps": vc.get("layer_norm_eps", 1e-5),
                "rope_theta": (vc.get("rope_parameters") or {}).get("rope_theta", 10000.0),
                "image_mean": first(ip.get("image_mean"), 0.5),
                "image_std": first(ip.get("image_std"), 0.5),
                "rescale_factor": ip.get("rescale_factor", 1.0 / 255.0),
                "image_token": pp.get("image_token", "<|unused_200054|>"),
            }
            from .inkling2_vision import Inkling2VisionModel
            self.model_classes["vision"] = Inkling2VisionModel

    @override
    def default_max_position_embeddings(self):
        return 1048576


def _wants_tap(params: dict, visit: int) -> bool:
    """Whether a draft model asked for the hidden state at this layer application (params["export_state_layers"])"""
    want = params.get("export_state_layers")
    return bool(want) and visit in want


def _export_tap(params: dict, t: torch.Tensor):
    """Append a tapped state [bsz, seq, hidden] to params["export_states"] (in forward order)"""
    states = params.get("export_states")
    if states is None:
        states = params["export_states"] = []
    states.append(t)


class _RowState:
    """Per batch row: block states [nb, S, D] (None while empty) and the loop bookkeeping"""

    def __init__(self):
        self.blocks = None
        self.loop = I2.LoopState()

    def to(self, device):
        mv = lambda t: t if t is None or t.device == device else t.to(device)
        self.blocks = mv(self.blocks)
        lp = self.loop
        if lp.pre is not None:
            lp.pre = (mv(lp.pre[0]), mv(lp.pre[1]))
        if lp.src is not None:
            lp.src = (mv(lp.src[0]), mv(lp.src[1]))


class _I2Module(Module):
    """Common parts: tensors are plain attributes, loaded with `get`"""

    def _getter(self, device):
        stc = self.config.stc
        adt = self.config.activation_dtype

        def get(key):
            t = stc.get_tensor(key, device, allow_bf16 = True, no_defer = True)
            return t.to(adt) if t.dtype == torch.bfloat16 and adt != torch.bfloat16 else t
        return get

    @override
    def can_defer_load(self):
        return False

    @override
    def unload(self):
        for n in list(vars(self)):
            if isinstance(getattr(self, n), (torch.Tensor, I2.LayerWeights, I2.PackedExperts, I2.HeadWeights)):
                setattr(self, n, None)
        self.device = None

    @override
    def optimizer_targets(self):
        return []

    @override
    def get_tensors(self):
        return {}


class _KernelOps:
    """
    The fused decode-step kernels of the extension (exllamav3_ext/inkling2.cu) behind the interfaces of
    modules/inkling2.py, for one token. Used inside the CUDA graph step when the extension has them and the
    activations are fp16.
    """

    @staticmethod
    def available(config) -> bool:
        from ..ext import exllamav3_ext as ext
        return os.environ.get("EXL3_I2_KERNELS", "1") != "0" and hasattr(ext, "i2_attn_res") and \
            config.activation_dtype == torch.half and _CONV_DT == torch.half

    def __init__(self):
        from ..ext import exllamav3_ext as ext
        self.ext = ext
        self.has_attn = hasattr(ext, "i2_attn_one") and os.environ.get("EXL3_I2_ATTN_KERNEL", "1") != "0"
        self.has_qkv = self.has_attn and hasattr(ext, "i2_qkv_post") and os.environ.get("EXL3_I2_QKV_KERNEL", "1") != "0"
        # Norm + gain + cast as one kernel, and the residuals, casts and output gate inside the convolution and
        # attention kernels (EXL3_I2_FUSE=0: the PyTorch operations between the kernels, for comparison)
        self.fused = self.has_qkv and hasattr(ext, "i2_norm") and os.environ.get("EXL3_I2_FUSE", "1") != "0"

    def norm(self, x, w, dtype, eps):
        """RMS norm over the last dimension times the gain w ([D], or one row per leading index of x), as dtype"""
        out = torch.empty(x.shape, dtype = dtype, device = x.device)
        self.ext.i2_norm(x, w, out, eps)
        return out

    def attn_res(self, blocks, hidden, query, eps):
        S, D = hidden.shape
        q = query.view(-1, D)
        out = torch.empty((q.shape[0], S, D), dtype = torch.float32, device = hidden.device)
        self.ext.i2_attn_res(blocks, hidden, q, out, eps)
        return out[0] if query.dim() == 1 else out

    def message_gate(self, x, message, w_down, w_forget, w_gain, eps, out = None):
        S, D = x.shape
        if out is None:
            out = torch.empty((S, D), dtype = torch.float32, device = x.device)
        g = torch.empty((S, w_down.shape[0]), dtype = torch.float32, device = x.device)
        # The kernel reads forget / gain as [r, D]; the transposed copies are made once and kept on the tensors
        wt = []
        for w in (w_forget, w_gain):
            t = getattr(w, "_i2_t", None)
            if t is None:
                t = w._i2_t = w.t().contiguous()
            wt.append(t)
        self.ext.i2_message_gate(x, message, w_down, wt[0], wt[1], g, out, eps)
        return out


class _GraphCacheView:
    """
    _CacheView for a few tokens inside a CUDA graph: every row index is computed on the device from `ctl` (long
    [2]: the first token's position, and its row in the sliding buffers), so the captured step is the same for
    every position. More than one token needs the kernels (the PyTorch forms here are for one).
    """

    def __init__(self, rsl: Inkling2LayerState, slot: int, ctl: torch.Tensor, cl: CacheLayer_fp16 | None,
                 base: int, cap: int, consts: dict):
        self.rsl, self.slot, self.cl, self.base, self.cap, self.consts = rsl, slot, cl, base, cap, consts
        self.pos, self.cpos = ctl[0], ctl[1]
        if consts.get("kops") is None:
            rows = (self.pos - rsl.taps + consts["taps"]) % CONV_RING
            self.conv = {n: rsl.conv[n][slot].index_select(0, rows) for n in CONV_NAMES}
        else:
            self.conv = {}                                          # the convolution kernel reads the ring itself
        self.wrow = (self.pos % CONV_RING).view(1)
        self.d = self.visible = None
        self.ctl = ctl
        self.kops = consts.get("kops")

    def conv_step(self, name: str, x: torch.Tensor, w: torch.Tensor, scale: float, dyn: torch.Tensor | None = None,
                  add: torch.Tensor | None = None, round_x: bool = False, out_dtype = torch.float32):
        """
        Convolution over the ring's history and x [S, C], and the ring writes, in one kernel; [S, C] (fp32 unless
        out_dtype says otherwise). add: a residual added to the result; round_x: an fp32 x counts as its fp16
        rounding
        """
        out = torch.empty(x.shape, dtype = out_dtype, device = x.device)
        if add is None and not round_x:
            self.kops.ext.i2_conv_step(x, self.rsl.conv[name][self.slot], w, dyn, self.ctl, out, scale)
        else:
            self.kops.ext.i2_conv_step(x, self.rsl.conv[name][self.slot], w, dyn, self.ctl, out, scale, add, round_x)
        return out

    def record(self, name: str, x: torch.Tensor):
        self.rsl.conv[name][self.slot].index_copy_(0, self.wrow, x.to(_CONV_DT))

    def extend(self, k: torch.Tensor, v: torch.Tensor, adt: torch.dtype):
        rsl = self.rsl
        if self.cl is None:
            # Sliding layer: the window is the `window` rows ending at the new one; rows before the buffer's
            # start are masked
            ks, vs = rsl.k_state[self.slot], rsl.v_state[self.slot]
            row = self.cpos.view(1)
            ks.index_copy_(0, row, k.to(torch.half))
            vs.index_copy_(0, row, v.to(torch.half))
            if self.kops is not None and self.kops.has_attn:
                return None, None, None                             # attend() reads the buffer in place
            idx = (self.cpos - (rsl.window - 1) + self.consts["window"]).clamp(min = 0)
            self.d = self.consts["d_window"]
            self.visible = self.d <= self.pos
            return ks.index_select(0, idx).to(adt), vs.index_select(0, idx).to(adt), None
        # Global layer: the first `cap` rows of the sequence; rows past the token are masked. With contiguous pages
        # the rows are one slice of the cache (base >= 0); otherwise they are gathered through the page table
        fk = self.cl.k.view(-1, *self.cl.k.shape[2:])
        fv = self.cl.v.view(-1, *self.cl.v.shape[2:])
        self.d = self.pos - self.consts["keys"]
        self.visible = self.d >= 0
        in_place = self.kops is not None and self.kops.has_attn     # attend() reads the pages in place
        if self.base >= 0:
            row = (self.base + self.pos).view(1)
            fk.index_copy_(0, row, k.to(torch.half))
            fv.index_copy_(0, row, v.to(torch.half))
            if in_place:
                return None, None, None
            return fk[self.base : self.base + self.cap].to(adt), fv[self.base : self.base + self.cap].to(adt), None
        pages = self.consts["pages"]                                # long [cap / PAGE_SIZE], set before each replay
        row = (pages.index_select(0, (self.pos // PAGE_SIZE).view(1)) * PAGE_SIZE + self.pos % PAGE_SIZE).view(1)
        fk.index_copy_(0, row, k.to(torch.half))
        fv.index_copy_(0, row, v.to(torch.half))
        if in_place:
            return None, None, None
        ka = self.cl.k.index_select(0, pages).view(self.cap, *self.cl.k.shape[2:])
        va = self.cl.v.index_select(0, pages).view(self.cap, *self.cl.v.shape[2:])
        return ka.to(adt), va.to(adt), None


_capture_streams = {}


def _graph_attend(self, q, rel, sinks, proj, scale, ogate = None):
    """
    _GraphCacheView.attend: the attention kernel over this layer application's keys. fp32 [S, H, dh]; with ogate
    (fp16 [S, H * dh]) the output gate is applied as well and the result is fp16
    """
    rsl, ext = self.rsl, self.kops.ext
    S, H, dh = q.shape
    qf, rf = q.float(), rel.float()
    if ogate is None:
        out = torch.empty((S, H, dh), dtype = torch.float32, device = q.device)
        sf, extra = sinks.float(), ()
    else:
        out = torch.empty((S, H, dh), dtype = torch.half, device = q.device)
        sf, extra = sinks, (ogate,)             # the kernel reads the fp16 sinks as they are
    part = self.consts["part"]                  # scratch shared by every layer application on this device
    if self.cl is None:
        ext.i2_attn_one(qf, rf, sf, proj, rsl.k_state[self.slot], rsl.v_state[self.slot], None, self.ctl, part, out,
                        True, 0, rsl.window, rsl.window, scale, *extra)
    else:
        ext.i2_attn_one(qf, rf, sf, proj, self.cl.k, self.cl.v, self.consts["pages"] if self.base < 0 else None,
                        self.ctl, part, out, False, max(self.base, 0), 0, self.cap, scale, *extra)
    return out


_GraphCacheView.attend = _graph_attend


def _graph_qkv_post(self, W, q, r, k, v):
    """_GraphCacheView.qkv_post: fp32 q [S, H, dh] and relative states [S, H, d_rel]; K and V go into the cache"""
    rsl, cfg = self.rsl, W.cfg
    S = q.shape[0]
    H, dh, dr = cfg.heads, cfg.head_dim, cfg.d_rel
    q_out = torch.empty((S, H, dh), dtype = torch.float32, device = q.device)
    rel_out = torch.empty((S, H, dr), dtype = torch.float32, device = q.device)
    local = self.cl is None
    ks, vs = (rsl.k_state[self.slot], rsl.v_state[self.slot]) if local else (self.cl.k, self.cl.v)
    self.kops.ext.i2_qkv_post(
        q, r, k, v, W.q_scale, W.rel_scale, ks, vs,
        self.consts["pages"] if (not local and self.base < 0) else None, self.ctl, q_out, rel_out,
        local, max(self.base, 0), cfg.qk_clip, cfg.rel_clip, cfg.log_alpha,
        float(cfg.log_n_floor) if W.is_global else 0.0, cfg.eps,
    )
    return q_out, rel_out


_GraphCacheView.qkv_post = _graph_qkv_post


class _DecodeGraph:
    """
    One captured decode step for a (cache, state slot, first page, key capacity). The layer applications are split
    into runs on the same GPU; each run is one CUDA graph, and the state that crosses to the next GPU is copied
    between the replays.
    """

    def __init__(self, model: "Inkling2LoopModel", cache, slot: int, base_page: int, cap: int, width: int = 1,
                 taps: frozenset = frozenset()):
        # base_page < 0: the sequence's pages are not contiguous; the global layers gather them
        # width: tokens per step (1 when decoding, more for a pass that verifies drafted tokens)
        # taps: layer applications whose input a draft model reads (see _wants_tap)
        self.key = (id(cache), slot, base_page, cap, width, taps)
        self.width = width
        self.taps = taps
        self.exports = []
        cfg = model.config.i2
        apps = [(m, inst) for m, inst, _ in model.fwd_modules]
        self.segments = []
        for m, inst in apps:
            if not self.segments or self.segments[-1]["device"] != m.device:
                self.segments.append({"device": m.device, "apps": [], "graph": None, "copies": []})
            self.segments[-1]["apps"].append((m, inst))
        devices = {seg["device"] for seg in self.segments}
        self.ctl = {d: torch.zeros(2, dtype = torch.long, device = d) for d in devices}
        # The step's input: the embedding rows of its tokens (the table may be in system RAM)
        self.embed = model.embed
        self.x0 = torch.zeros((1, width, cfg.hidden), dtype = self.embed.weight.dtype, device = self.embed.device)
        ar = lambda n, d: torch.arange(n, dtype = torch.long, device = d)
        kops = _KernelOps() if _KernelOps.available(model.config) else None
        self.kernels = kops is not None
        consts = {d: {
            "taps": ar(cfg.conv_k - 1, d), "window": ar(cfg.window, d),
            "d_window": cfg.window - 1 - ar(cfg.window, d), "keys": ar(cap, d),
            "pages": torch.zeros(cap // PAGE_SIZE, dtype = torch.long, device = d),
            "kops": kops,
            # Scratch of the attention kernel's first launch: per token, K/V head, block of 32 keys and query
            # head, 2 + head_dim floats
            "part": torch.empty(
                (width * cfg.kv_heads * ((max(cap, cfg.window) + 31) // 32) * (cfg.heads // cfg.kv_heads) * (2 + cfg.head_dim),),
                dtype = torch.float32, device = d,
            ) if kops is not None else None,
        } for d in devices}
        self.consts = consts
        self.g = {"ctl": self.ctl, "slot": slot, "base": base_page * PAGE_SIZE if base_page >= 0 else -1,
                  "cap": cap, "consts": consts}
        self.cache = cache
        self.logits = None
        self.pages = None

    def _run(self, capture: bool):
        row = _RowState()
        params = {"cache": self.cache, "i2": [row], "i2_graph": self.g}
        if self.taps:
            params["export_state_layers"] = self.taps
        x = self.x0
        for seg in self.segments:
            dev = seg["device"]
            # State that arrives from another GPU: a buffer on this one, filled before every replay
            copies = []

            def move(t):
                if t is None or t.device == dev:
                    return t
                dst = torch.empty_like(t, device = dev)
                dst.copy_(t)
                copies.append((t, dst))
                return dst

            x = move(x)
            row.blocks = move(row.blocks)
            lp = row.loop
            if lp.pre is not None:
                lp.pre = (move(lp.pre[0]), move(lp.pre[1]))
            if lp.src is not None:
                lp.src = (move(lp.src[0]), move(lp.src[1]))

            def body(x):
                for module, inst in seg["apps"]:
                    params["layer_instance"] = inst
                    x = module.forward(x, params)
                return x

            if capture:
                seg["copies"] = copies
                # The graph object binds to the current device when it is created
                # Every device idle first: entering a capture empties the allocator caches of all devices, which
                # frees blocks that kernels still queued on another GPU may be using
                for d in self.ctl:
                    torch.cuda.synchronize(d)
                with torch.cuda.device(dev):
                    seg["graph"] = torch.cuda.CUDAGraph()
                    # An explicit capture stream on this device: torch.cuda.graph's default one is created once,
                    # on whichever device was current first, and captures nothing on any other
                    stream = _capture_streams.get(dev)
                    if stream is None:
                        stream = _capture_streams[dev] = torch.cuda.Stream(device = dev)
                    with torch.cuda.graph(seg["graph"], stream = stream):
                        x = body(x)
            else:
                x = body(x)
        if capture:
            self.exports = params.get("export_states") or []
        return x

    def build(self):
        # Once op by op (lazy initializations must not happen inside a capture), then the capture. Both write the
        # token's cache rows with the values the replay writes again.
        self._run(False)
        self.logits = self._run(True)

    def set_inputs(self, input_ids: torch.Tensor, position: int, cache_pos: int, pages: list | None):
        w = self.embed.weight
        self.x0.copy_(w[input_ids.to(w.device)])
        host = torch.tensor([position, cache_pos], dtype = torch.long)
        for t in self.ctl.values():
            t.copy_(host)
        if pages is not None and pages != self.pages:
            self.pages = pages
            host = torch.tensor(pages, dtype = torch.long)
            for c in self.consts.values():
                c["pages"].copy_(host)

    def step(self) -> torch.Tensor:
        for seg in self.segments:
            for src, dst in seg["copies"]:
                dst.copy_(src)
            with torch.cuda.device(seg["device"]):
                seg["graph"].replay()
        return self.logits.clone()


class Inkling2Embed(_I2Module):

    def __init__(self, config: Inkling2LoopConfig, key: str = "model.llm.embed"):
        super().__init__(config, key, None)
        self.weight = self.norm = None

    @override
    def load(self, device, **kwargs):
        self.device = device
        get = self._getter(device)
        # The table is read one row per token, so it stays in system RAM (0.9 GiB) and its VRAM goes to the layers:
        # with a long cache that is what lets the last loop layer sit on the first GPU with the others (a loop
        # layer on another GPU costs two crossings per pass). EXL3_I2_EMBED=gpu keeps it on the module's device
        wdev = device if _EMBED_GPU else torch.device("cpu")
        self.weight = self._getter(wdev)(self.key + ".weight")
        self.norm = get("model.llm.embed_norm.weight")
        self.norm_f32 = self.norm.float().contiguous()

    def rows(self, ids: torch.Tensor) -> torch.Tensor:
        """The tokens' embedding rows as stored, on the module's device"""
        w = self.weight
        return w[ids.to(w.device)].to(self.device)

    def lookup(self, ids: torch.Tensor, adt: torch.dtype, params: dict | None = None) -> torch.Tensor:
        """
        The normed embeddings of the tokens, on the module's device. Ids of multimodal embeddings
        (params["indexed_embeddings"], e.g. image tokens) take their rows from the embedding as they are: the
        vision tower's output replaces the normed text embeddings, as in the reference
        """
        emb = (params or {}).get("indexed_embeddings")
        if not emb or int(ids.max()) < FIRST_MM_EMBEDDING_INDEX:
            return I2.rms_norm(self.rows(ids), self.norm, self.config.i2.eps).to(adt)
        ids = ids.to(self.device)
        mm = ids >= FIRST_MM_EMBEDDING_INDEX
        x = I2.rms_norm(self.rows(ids.masked_fill(mm, 0)), self.norm, self.config.i2.eps).to(adt)
        todo = mm.clone()
        for e in emb:
            m = (ids >= e.first_index) & (ids < e.first_index + e.mm_length)
            if m.any():
                x[m] = e.embeddings.to(self.device, adt)[ids[m] - e.first_index]
                todo &= ~m
        assert not todo.any(), "Inkling2: the prompt has multimodal token ids without their embeddings"
        return x

    @override
    def weights_numel(self):
        return self.config.i2.vocab * self.config.i2.hidden

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        if "i2_graph" in params:
            # x: the tokens' embedding rows, looked up before the replay (_DecodeGraph.set_inputs)
            kops = params["i2_graph"]["consts"][self.device]["kops"]
            if kops is not None and kops.fused and x.dtype == torch.half:
                return kops.norm(x, self.norm_f32, self.config.activation_dtype, self.config.i2.eps)
            return I2.rms_norm(x, self.norm, self.config.i2.eps).to(self.config.activation_dtype)
        params["i2"] = [_RowState() for _ in range(x.shape[0])]
        params.pop("i2_pages", None)
        if params.get("cache") is not None:
            # Tokens already in the cache, per row: the position of the first new token
            past = [int(p) for p in params["cache_seqlens"].tolist()]
            for rs, p in zip(params["recurrent_states"], past):
                assert rs.position == p, f"Inkling2: recurrent state at {rs.position}, cache at {p}"
            params["i2_past"] = past
        return self.lookup(x, self.config.activation_dtype, params)


class Inkling2Layer(_I2Module):

    def __init__(self, config: Inkling2LoopConfig, idx: int):
        super().__init__(config, f"model.llm.layers.{idx}", None)
        c = config.i2
        self.layer_idx = idx
        self.W = self.ex = None
        self.is_global = c.is_global(idx)

        # Cache geometry: what the cache layer classes and the state read from an attention module
        self.num_kv_heads = c.kv_heads
        self.head_dim = c.head_dim
        # Rows a checkpoint position must have behind it in the state: the attention window, or for the global
        # layers (K/V in the paged cache) nothing; the convolution inputs are tracked by Inkling2State
        self.sliding_window = 0 if self.is_global else c.window
        # Sliding K/V buffer: the window, one page that can always be rolled back, and one more page so that a
        # few tokens written in place after a page drop (a multi-token graph step) still have their whole window
        # in front of them
        self.kv_state_size = (c.window + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE + 2 * PAGE_SIZE
        self.cache_layers = []
        self.recurrent_layers = []
        self.layer_state_cls = Inkling2LayerState
        self.caps.update({"recurrent_cache": True})

        # EXL3 experts if the model directory has them for this layer, else the checkpoint's packed experts
        self.mlp = None
        if Exl3Experts.present(config.stc, idx):
            self.mlp = Exl3Experts.make_mlp(config, idx)
            self.register_submodule(self.mlp)
        if self.is_global:
            self.caps.update({"kv_cache": True})

    @override
    def load(self, device, **kwargs):
        self.device = device
        get = self._getter(device)
        stc = self.config.stc
        self.W = I2.LayerWeights(get, self.config.i2, self.layer_idx)
        if self.mlp is not None:
            self.mlp.load(device, **kwargs)
            self.ex = Exl3Experts(self.mlp)
        else:
            # Packed experts keep their stored dtypes (uint8 codes, fp8 scales)
            raw = lambda key: stc.get_tensor(key, device, allow_bf16 = True, no_defer = True)
            self.ex = I2.PackedExperts(raw, stc.has_tensor, self.config.i2, self.layer_idx)
        for cl in self.cache_layers:
            assert type(cl) is CacheLayer_fp16, "Inkling2 supports the fp16 cache only"
            cl.alloc(device)
        for rl in self.recurrent_layers:
            rl.alloc(device)

    @override
    def autosplit_extra_measure(self, params):
        # The measuring forward runs one chunk at context 0. With a long history a global layer's attention holds
        # more: one block of logits (I2.ATTN_MAX_ELEMS) and the attention weights made from them, and a copy of
        # the sequence's keys and values when its cache pages are not contiguous
        if not self.is_global or not self.cache_layers:
            return
        c = self.config.i2
        rows = self.cache_layers[0].max_num_tokens
        kv_half = torch.empty((2, rows, c.kv_heads, c.head_dim), dtype = torch.half, device = self.device)
        if I2._prefill_ext() is not None and self.config.activation_dtype == torch.half:
            # Extension path: fp32 logits, the weights as two fp16 terms; K and V are read as stored
            logits = torch.empty((I2.ATTN_MAX_ELEMS,), dtype = torch.float, device = self.device)
            weights = torch.empty((2 * I2.ATTN_MAX_ELEMS,), dtype = torch.half, device = self.device)
            del logits, weights, kv_half
            return
        # PyTorch path: about twice the logits while they are normalized, and fp32 copies of a block of K and V
        logits = torch.empty((2 * I2.ATTN_MAX_ELEMS,), dtype = torch.float, device = self.device)
        rows = min(rows, I2.ATTN_KV_FP32_BYTES // (c.kv_heads * c.head_dim * 4))
        kv = torch.empty((2, rows, c.kv_heads, c.head_dim), dtype = torch.float, device = self.device)
        del logits, kv, kv_half

    @override
    def unload(self):
        if self.mlp is not None:
            self.mlp.unload()
            self.ex = None
        for cl in self.cache_layers:
            cl.free()
        for rl in self.recurrent_layers:
            rl.free()
        super().unload()

    @override
    def weights_numel(self):
        c = self.config.i2
        attn = c.hidden * (2 * c.heads * c.head_dim + 2 * c.kv_heads * c.head_dim + c.heads * (c.head_dim + c.d_rel + 1))
        return attn + c.experts * 3 * c.hidden * c.inter

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        cfg = self.config.i2
        adt = self.config.activation_dtype
        loop_pass = params.get("layer_instance", 0)
        g = params.get("i2_graph")
        if g is not None:
            # One token of one sequence inside a CUDA graph (see _DecodeGraph)
            st = params["i2"][0]
            if self.mlp is not None:
                self.ex.params = params
            key = (self.layer_idx, loop_pass)
            cache = params["cache"]
            ctl = g["ctl"][self.device]
            view = _GraphCacheView(
                cache.get_recurrent_layer(key), g["slot"], ctl, cache.layers[key] if self.is_global else None,
                g["base"], g["cap"], g["consts"][self.device],
            )
            msgs = st.loop.enter(cfg, self.W, x[0], st.blocks, loop_pass, view.kops)
            if _wants_tap(params, self.config.app_index[key]):
                # A copy: the tensor has to stay valid as an output of the graph
                _export_tap(params, st.loop.hidden.unsqueeze(0).clone())
            h, st.blocks = I2.layer_forward(self.W, self.ex, st.loop.hidden, st.loop.blocks, adt, view, ctl[0], msgs)
            st.loop.hidden = st.loop.blocks = None
            st.loop.leave(cfg, self.W, loop_pass)
            return h.unsqueeze(0)
        pos0 = params.get("i2_pos0", 0)
        out = torch.empty_like(x)
        cache = params.get("cache")
        if cache is not None:
            key = (self.layer_idx, loop_pass)
            rsl = cache.get_recurrent_layer(key)
            cl = cache.layers[key] if self.is_global else None
            bt = None
            if self.is_global:
                # Page numbers per row, on the host: read once per forward
                bt = params.get("i2_pages")
                if bt is None:
                    bt = params["i2_pages"] = params["block_table"].tolist()
            rsg, past = params["recurrent_states"], params["i2_past"]
            n = x.shape[1]
        if self.mlp is not None:
            self.ex.params = params
        tap = _wants_tap(params, self.config.app_index[(self.layer_idx, loop_pass)])
        taps = []
        for b, st in enumerate(params["i2"]):
            st.to(self.device)
            view = None
            if cache is not None:
                rs = rsg[b]
                pos0 = past[b]
                cache_pos = pos0 - rs.window_beg
                view = _CacheView(rsl, rs.slot, pos0, cache_pos, cl, bt[b] if self.is_global else None, n)
                # Every layer application drops the same pages from its sliding buffer; the state follows after
                # the forward (SWAState.post_advance)
                rs.wshift = ring_shift(cache_pos, n, self.kv_state_size)
            msgs = st.loop.enter(cfg, self.W, x[b], st.blocks, loop_pass)
            if tap:
                taps.append(st.loop.hidden)
            out[b], st.blocks = I2.layer_forward(
                self.W, self.ex, st.loop.hidden, st.loop.blocks, adt, view, pos0, msgs
            )
            st.loop.hidden = st.loop.blocks = None
            st.loop.leave(cfg, self.W, loop_pass)
        if tap:
            _export_tap(params, torch.stack(taps))
        if params.get("prefill") and _wants_tap(params, len(cfg.applications)):
            # A prefill stops after the last layer, without the head; a draft model's tap of the final normed
            # hidden state is made here
            head = self.head_ref()
            finals = []
            for b, st in enumerate(params["i2"]):
                st.to(head.device)
                finals.append(I2.final_hidden(head.Hd, out[b].to(head.device), st.blocks, adt))
            _export_tap(params, torch.stack(finals))
        return out


class Inkling2Head(_I2Module):

    def __init__(self, config: Inkling2LoopConfig, key: str = "model.llm"):
        super().__init__(config, key, None)
        self.Hd = None
        self.caps.update({"logits_output": True})

    @override
    def load(self, device, **kwargs):
        self.device = device
        self.Hd = I2.HeadWeights(self._getter(device), self.config.i2)

    @override
    def weights_numel(self):
        return self.config.i2.vocab * self.config.i2.hidden

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        adt = self.config.activation_dtype
        n = x.shape[1]
        out, finals = [], []
        for b, st in enumerate(params["i2"]):
            if "i2_graph" not in params:
                st.to(self.device)
            # The forward loop may have cut the hidden state to the last tokens (last_tokens_only)
            blocks = st.blocks[:, -n:]
            g = params.get("i2_graph")
            fin = I2.final_hidden(self.Hd, x[b], blocks, adt, None if g is None else g["consts"][self.device]["kops"])
            finals.append(fin)
            out.append(I2.logits_from_hidden(self.Hd, fin))
        if _wants_tap(params, len(self.config.i2.applications)):
            # The final normed hidden state: the tap after the last layer application
            _export_tap(params, torch.stack(finals).clone())
        if "i2_graph" not in params:
            del params["i2"]
        return out[0].unsqueeze(0) if len(out) == 1 else torch.stack(out)


class Inkling2LoopModel(Model):
    config_class = Inkling2LoopConfig

    def __init__(self, config: Inkling2LoopConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.modules += [Inkling2Embed(config)]
        self.first_block_idx = len(self.modules)
        self.modules += [Inkling2Layer(config, idx) for idx in range(config.num_hidden_layers)]
        self.last_kv_module_idx = len(self.modules) - 1
        self.modules += [Inkling2Head(config)]
        self.logit_layer_idx = len(self.modules) - 1
        self.embed, self.head = self.modules[0], self.modules[-1]
        for m in self.modules[1:-1]:
            m.head_ref = weakref.ref(self.head)

        self.caps.update({
            "supports_tp": False,
            "recurrent_states": True,
            # A checkpoint holds the attention window of every sliding layer application (about 170 MB)
            "default_recurrent_checkpoint_interval": 4096,
            # No pipelined prefill unless asked for (EXL3_I2_PIPELINED_PREFILL=1). The generator's pipeline runs the
            # GPUs of a straight split at the same time, one chunk behind the other; everything else here runs
            # them in turn. On the development machine (RTX 4090 + 3090) the first chunk that had both GPUs
            # under full load power-cycled the computer, three times out of three (2026-11-24), and the path is
            # not validated against the sequential one for this model
            "pipelined_prefill": os.environ.get("EXL3_I2_PIPELINED_PREFILL", "0") == "1",
            # A pass that verifies drafted tokens runs as a CUDA graph of one of these sizes (padded up to it)
            "draft_pass_widths": tuple(w for w in _GRAPH_WIDTHS if w <= _GRAPH_MAX_WIDTH),
        })
        self.recurrent_state_cls = Inkling2State
        self._graphs = {}               # tokens per step -> _DecodeGraph
        self._graph_failed = False
        self._graph_capable = None
        self.graph_steps = 0            # decode steps served by graph replay (for tests and logs)
        self.graph_builds = 0
        self._stats = {}

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        if params.get("cache") is not None:
            prepare_for_recurrence(input_ids, params, self)
            input_ids = prepare_for_attn(input_ids, params)
        return input_ids

    def active_devices_cuda(self):
        return sorted({m.device for m in self.modules if m.device is not None and m.device.type == "cuda"}, key = str)

    @property
    def _graph_ok(self) -> bool:
        if self._graph_capable is None:
            layers = [m for m in self.modules if isinstance(m, Inkling2Layer)]
            self._graph_capable = all(hasattr(m.ex, "moe") for m in layers)
        return self._graph_capable

    def _graph_step(self, input_ids: torch.Tensor, params: dict):
        """
        The forward for one new token of one cached sequence as CUDA graph replays, or None when the step does not
        qualify (then it runs op by op). Leaves the recurrent state as the op-by-op forward would.
        """
        width = input_ids.shape[1]
        if not _GRAPH or self._graph_failed or input_ids.shape[0] != 1 or not 1 <= width <= _GRAPH_MAX_WIDTH:
            return None
        if not self._graphs and torch.cuda.get_allocator_backend() != "native":
            # Graph capture needs PyTorch's own caching allocator: with cudaMallocAsync (tabbyAPI's
            # cuda_malloc_async) the process dies during the first replay
            self._graph_failed = True
            print(" !! Inkling2: CUDA graph decode needs the native CUDA allocator (cuda_malloc_async: false); "
                  "decoding op by op")
            return None
        cache = params.get("cache")
        rsg = params.get("recurrent_states")
        if cache is None or not rsg or len(rsg) != 1 or "i2_pos0" in params:
            return None
        # Image tokens (multimodal embeddings) take the op-by-op forward, which splices them in; a step of text
        # tokens of a sequence that has images runs as a graph like any other
        if int(input_ids.max()) >= FIRST_MM_EMBEDDING_INDEX:
            return None
        # More than one token (a pass that verifies drafted tokens) needs every kernel
        if width > 1 and not (_KernelOps.available(self.config) and _KernelOps().has_qkv):
            return None
        # The packed experts of a source checkpoint read values on the host; only the EXL3 kernels can be captured
        if not self._graph_ok:
            return None
        rs = rsg[0]
        position = int(params["cache_seqlens"][0])
        if position < _GRAPH_MIN_POS or rs.position != position:
            return None
        # The global layers read the sequence's pages as one slice of the cache when they are contiguous up to the
        # token's page, else gathered through the page table (a copy of the keys per layer application)
        pages = params["block_table"][0].tolist()
        # A pass that verifies drafted tokens comes in any width; each width is its own graph, so the pass is
        # padded to the next of a few widths (the last token repeated). The padded rows land past the end of
        # the sequence, where the real tokens of those positions overwrite them later
        real = width
        for w in _GRAPH_WIDTHS:
            if width <= w <= _GRAPH_MAX_WIDTH and (position + w - 1) // PAGE_SIZE < len(pages):
                width = w
                break
        cur = (position + width - 1) // PAGE_SIZE               # page of the last new token
        if cur >= len(pages):
            return None
        if width != real:
            input_ids = torch.cat([input_ids, input_ids[:, -1:].expand(-1, width - real)], dim = 1)
        base_page = pages[0]
        cap = (position + width + 1 + _GRAPH_KEY_STEP - 1) // _GRAPH_KEY_STEP * _GRAPH_KEY_STEP
        contiguous = all(pages[j] == base_page + j for j in range(cur + 1)) and \
            base_page * PAGE_SIZE + cap <= cache.max_num_tokens
        gather = None
        if not contiguous:
            cap = min(cap, cache.max_num_tokens)
            n = cap // PAGE_SIZE
            # Pages past the token's are masked; any valid page number will do for them
            gather = (pages[: cur + 1] + [pages[0]] * n)[:n]
            base_page = -1

        # Sliding buffers: drop pages from the front first when the new row does not fit (as the op-by-op
        # forward does through kv_write)
        size = self.modules[self.first_block_idx].kv_state_size
        cache_pos = position - rs.window_beg
        shift = ring_shift(cache_pos, width, size)
        if shift:
            for rsl in cache.get_all_recurrent_layers().values():
                if rsl.local:
                    for state in (rsl.k_state, rsl.v_state):
                        state[rs.slot, : cache_pos - shift] = state[rs.slot, shift : cache_pos].clone()
            cache_pos -= shift
        rs.wshift = shift

        taps = frozenset(params.get("export_state_layers") or ())
        key = (id(cache), rs.slot, base_page, cap, width, taps)
        try:
            graph = self._graphs.get(width)
            if graph is None or graph.key != key:
                # A graph's memory must not go while a replay of it may still be running
                for d in self.active_devices_cuda():
                    torch.cuda.synchronize(d)
                self._graphs.pop(width, None)
                # One graph per width; beyond a few widths the least recently built one goes
                while len(self._graphs) >= _GRAPH_MAX_GRAPHS:
                    self._graphs.pop(next(w for w in self._graphs if w != 1) if any(w != 1 for w in self._graphs) else next(iter(self._graphs)))
                graph = _DecodeGraph(self, cache, rs.slot, base_page, cap, width, taps)
                graph.set_inputs(input_ids, position, cache_pos, gather)
                graph.build()
                self.graph_builds += 1
                self._graphs[width] = graph
            graph.set_inputs(input_ids, position, cache_pos, gather)
            self.graph_steps += 1
            y = graph.step()
            if graph.exports:
                params["export_states"] = [t[:, :real].clone() for t in graph.exports]
            if width != real:
                y = y[:, :real]
                # The padded rows went through the convolution rings: they bound a later rewind
                rs.high = max(rs.high, position + width)
            num = params.get("last_tokens_only")
            return y[:, -num:].contiguous() if num else y
        except Exception as e:
            self._graphs.clear()
            self._graph_failed = True
            print(f" !! Inkling2: CUDA graph decode disabled ({type(e).__name__}: {str(e).splitlines()[0][:200]})")
            # The step may be half done: its cache rows are rewritten by the op-by-op forward that follows
            rs.wshift = 0
            return None

    @override
    @torch.inference_mode
    def forward(self, input_ids: torch.Tensor, params: dict | None = None):
        # Model.forward for the layer-split path, with the graph step in front
        if params is None:
            params = {}
        if self.pipeline is not None:
            self.pipeline.flush()
        x = self.prepare_inputs(input_ids, params)
        y = self._graph_step(x, params)
        graph = y is not None
        if _STATS and params.get("cache") is not None:
            # Which forwards of a cached sequence ran as graph replays, by (batch, tokens)
            st = self._stats
            k = ("graph" if graph else "op-by-op", tuple(input_ids.shape))
            st[k] = st.get(k, 0) + 1
            st["n"] = st.get("n", 0) + 1
            if st["n"] % _STATS == 0:
                print(f" -- Inkling2 forwards: { {k: v for k, v in st.items() if k != 'n'} }, graphs built {self.graph_builds}", flush = True)
        if y is None:
            y = self.forward_ls(x, params)
        if _NAN_CHECK and not torch.isfinite(y).all():
            rs = (params.get("recurrent_states") or [None])[0]
            print(f" !! Inkling2: non-finite logits ({'graph' if graph else 'op-by-op'} forward, "
                  f"{tuple(input_ids.shape)} tokens at position {getattr(rs, 'position', '?')})", flush = True)
        advance_recurrent_states(input_ids, params, self)
        return y

    @override
    def unload(self):
        self._graphs.clear()
        self._graph_capable = None
        super().unload()

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"<|message_system|><|content_text|>{system_prompt}<|end_message|>"
        p += "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
        p += f"<|message_user|><|content_text|>{prompt}<|end_message|>"
        return p
