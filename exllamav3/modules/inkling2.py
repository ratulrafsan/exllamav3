"""
Local patch: Inkling-2 (thinkingmachines/Inkling-2-Mini, the looped variant) reference math.

Plain PyTorch, one sequence at a time, no dependencies on the rest of the package. The numbers follow the
transformers `inkling2_loop` reference (modeling_inkling2_loop.py). The layer math and the NVFP4 / MXFP8
dequantizers are adapted from 0xSero/inkling2-exl3 (MIT, Copyright (c) 2026 0xSero), which covers the
non-looped Inkling-2-Small; the loop (passes, messages, message gates) is added here.

The model in short:

  * The input of every sublayer is a "block attention residual": a learned query attends over the outputs
    of the finished 6-layer blocks plus the running hidden state, and the softmax-weighted mix of those
    (RMS-normalized) states is the input. At a block boundary the incoming hidden state becomes a block
    state and is not added to the residual.
  * No RoPE. Position enters through a relative logit bias computed from the query's hidden state
    (`wr_du` -> `rel_logits_proj`), non-zero for 0 <= distance < rel_extent, and (global layers only)
    a log-of-position scale on the query and on the bias.
  * 4-tap causal convolutions on K and V (static taps on the sliding-window layers, per-token taps on the
    global layers) and on the attention and MLP outputs.
  * Attention has one sink logit per query and head, L2 clipping on Q, K and the relative states, an
    RMS norm without gain on V, and a sigmoid gate on the output.
  * Sparse MoE: sigmoid router with a bias used for selection only, top-k weights normalized and scaled,
    SwiGLU with tanh soft caps on the gate factor and on `up`.
  * Loop: layers [loop_start, loop_end) run `passes` times with the same weights. Every pass after the
    first restarts from the hidden state and block states before the loop. It receives four "messages"
    per layer (for the Q, K, V and MLP inputs): learned queries attend over the previous pass's block
    states and final hidden state. A rank-`gate_rank` tanh gate mixes each message into its input:
    (1 - forget) * input + gain * message. Each pass has its own K/V and convolution state.

State between layers: `hidden` [S, D] and `blocks` [nb, S, D] (None while empty), the block states, kept
RMS-normalized in fp32 (they are only ever read through the block attention residual).

Incremental decode: the layer functions take a `cache` object for one layer application of one sequence
(see architecture/inkling2_loop.py for the implementation over ExLlamaV3's cache):

    cache.conv                  dict "k" / "v" / "attn" / "mlp" -> the conv_k - 1 raw convolution inputs before
                                the first new token, [conv_k - 1, C] (zeros before the start of the sequence)
    cache.record(name, x)       store the raw convolution inputs [S, C] of the new tokens
    cache.extend(k, v, adt)     store the new K / V rows [S, Hk, dh]; returns (all K, all V, q_off): the rows
                                attention can see, ending with the new ones, and the index of the first new row.
                                q_off None selects the static form for one token: the keys are a buffer of fixed
                                capacity, and cache.d / cache.visible give each key's distance and visibility
"""
from __future__ import annotations
import json, math, os
from dataclasses import dataclass
import torch
import torch.nn.functional as F

# Test switch: reproduce two rounding steps of the transformers reference that the checkpoint does not ask for,
# so that the port can be compared with it bit for bit (inkling/port_check.py --ref-exact):
#   * attention logits, relative bias and attention weights are rounded to the activation dtype (bf16)
#   * the per-expert NVFP4 factor `amax` is rounded to bf16 before the division (its loader casts the source
#     tensors to the parameter dtype first)
# Off, the logits stay fp32 and amax is used as stored.
REF_EXACT = False

# Test hook: ROUTE_HOOK(W, pos0, sel) -> sel, called with every layer application's routed expert ids [S, k] (pos0:
# position of the first row). Lets a test record the routing of one pass and replay it in another. Top-k routing is
# discontinuous, so without this two passes that differ by rounding noise alone drift apart by a few percent
ROUTE_HOOK = None

# Calibration hook: MOE_HOOK(W, e, x, a), called for every expert a layer application routes tokens to, with the
# expert's input rows x [n, D] and its activations a [n, inter] (the inputs of gate/up and of down). Used by
# inkling/convert_exl3.py to collect the Hessians for the EXL3 conversion
MOE_HOOK = None

# Test switch: round K and V to fp16 right after they are computed, cached or not. The cache stores them in fp16, so
# with fp32 activations this makes a cached pass and an uncached pass see the same K/V (inkling/cache_check.py)
ROUND_KV = False

FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
_byte_lut_cache = {}


def _byte_lut(device):
    # byte -> (value of low nibble, value of high nibble); the low nibble is the first element of the pair
    key = str(device)
    if key not in _byte_lut_cache:
        lut = torch.tensor(FP4_VALUES, dtype = torch.float32)
        b = torch.arange(256)
        _byte_lut_cache[key] = torch.stack([lut[b & 15], lut[b >> 4]], dim = -1).to(device)
    return _byte_lut_cache[key]


def dequant_nvfp4(w: torch.Tensor, scale: torch.Tensor, gscale: torch.Tensor) -> torch.Tensor:
    """
    NVFP4 -> fp32. w: uint8 [..., N, K/2] (two E2M1 codes per byte, low nibble first), scale: float8_e4m3
    [..., N, K/16] block scales, gscale: fp32 per-expert factor (amax / outer_scale_denominator)
    """
    q = _byte_lut(w.device)[w.long()]                   # [..., N, K/2, 2]
    q = q.reshape(*w.shape[:-1], w.shape[-1] * 2)
    K = q.shape[-1]
    out = (q.reshape(*q.shape[:-1], K // 16, 16) * scale.float().unsqueeze(-1)).reshape(q.shape)
    return out * gscale.float().reshape(*gscale.shape, *([1] * (out.dim() - gscale.dim())))


def dequant_mxfp8(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    MXFP8 -> fp32. w: float8_e4m3 [..., N, K], scale: uint8 E8M0 [..., N, K/32]: value = e4m3 * 2^(s - 127)
    """
    K = w.shape[-1]
    s = torch.exp2(scale.float() - 127.0)
    return (w.float().reshape(*w.shape[:-1], K // 32, 32) * s.unsqueeze(-1)).reshape(w.shape)


# ----------------------------------------------------------------------------------------------------------
# Configuration

@dataclass
class Inkling2Cfg:
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    d_rel: int
    rel_extent: int
    window: int
    vocab: int
    unpadded_vocab: int
    eps: float
    conv_k: int
    qk_clip: float
    rel_clip: float
    log_n_floor: int
    log_alpha: float
    block_layers: int
    experts: int
    topk: int
    inter: int
    route_scale: float
    gate_cap: float
    up_cap: float
    final_cap: float
    global_layers: tuple
    loop_start: int
    loop_end: int
    passes: int
    gate_rank: int

    @staticmethod
    def from_dict(c: dict) -> "Inkling2Cfg":
        """c: the `text_config` dict of config.json"""
        n = c["num_hidden_layers"]
        local = set(c["local_layer_ids"])
        loop = c.get("loop_config") or {}
        assert not loop.get("is_varloop", False), "variable-pass loops are not supported"
        return Inkling2Cfg(
            hidden = c["hidden_size"], layers = n, heads = c["num_attention_heads"],
            kv_heads = c["num_key_value_heads"], head_dim = c["head_dim"], d_rel = c["d_rel"],
            rel_extent = c["rel_extent"], window = c["sliding_window_size"], vocab = c["vocab_size"],
            unpadded_vocab = c["unpadded_vocab_size"], eps = c["rms_norm_eps"], conv_k = c["sconv_kernel_size"],
            qk_clip = c["qk_l2_clip_threshold"], rel_clip = c["rel_l2_clip_threshold"],
            log_n_floor = c["log_scaling_n_floor"], log_alpha = c["log_scaling_alpha"],
            block_layers = c["n_layers_per_attn_res_block"], experts = c["n_routed_experts"],
            topk = c["num_experts_per_tok"], inter = c["intermediate_size"], route_scale = c["route_scale"],
            gate_cap = c["mlp_gate_softcap"], up_cap = c["mlp_up_softcap"],
            final_cap = c["final_logit_softcapping"],
            global_layers = tuple(i for i in range(n) if i not in local),
            # Without a loop_config: one pass over an empty range, i.e. the non-looped Inkling-2
            loop_start = loop.get("core_start", n), loop_end = loop.get("core_end", n),
            passes = loop.get("num_passes", 1), gate_rank = loop.get("gate_rank", 0),
        )

    @staticmethod
    def from_dir(directory: str) -> "Inkling2Cfg":
        with open(os.path.join(directory, "config.json")) as f:
            return Inkling2Cfg.from_dict(json.load(f)["text_config"])

    def is_global(self, i): return i in self.global_layers
    def is_boundary(self, i): return i % self.block_layers == 0
    def is_loop(self, i): return self.loop_start <= i < self.loop_end

    @property
    def applications(self):
        """(layer, pass) of every layer application, in execution order"""
        pre = [(i, 0) for i in range(self.loop_start)]
        loop = [(i, p) for p in range(self.passes) for i in range(self.loop_start, self.loop_end)]
        post = [(i, 0) for i in range(self.loop_end, self.layers)]
        return pre + loop + post


# ----------------------------------------------------------------------------------------------------------
# Small ops. `adt` is the activation dtype (bf16 in the reference)

def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    """fp32 compute, weight multiplied in fp32; the caller casts"""
    xf = x.float()
    if not REF_EXACT:
        return F.rms_norm(xf, xf.shape[-1:], None if w is None else w.float(), eps)
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim = True) + eps)
    return xf if w is None else w.float() * xf


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.tanh(x / cap) * cap


def l2_clip(x: torch.Tensor, threshold: float, scale: torch.Tensor | None) -> torch.Tensor:
    """x: [..., heads, d] -> fp32: optional learned scale, then vectors longer than `threshold` are shortened to it"""
    xf = x.float()
    if scale is not None:
        xf = xf * scale.float()
    norm = torch.linalg.vector_norm(xf, dim = -1, keepdim = True)
    return xf * (threshold / (norm + 1e-6)).clamp(max = 1.0)


def block_state(hidden: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """A hidden state [S, D] as a block state [1, S, D]: RMS-normalized, fp32 (the form attn_res reads)"""
    return rms_norm(hidden, None, eps).unsqueeze(0)


def attn_res(blocks: torch.Tensor | None, hidden: torch.Tensor, query: torch.Tensor, eps: float = 1e-6,
             chunk: int = 4096, kops = None) -> torch.Tensor:
    """
    Block attention residual. blocks: [nb, S, D] block states (see block_state: already normalized; None or empty
    allowed), hidden: [S, D], query: [D] or [nq, D]. The query attends over the block states and the normalized
    hidden state; the result is the softmax-weighted mix of those. Returns fp32 [S, D] (query [D]) or [nq, S, D].
    The loop messages are the same operation over the previous pass's states.
    """
    single = query.dim() == 1
    q = query.float().reshape(-1, hidden.shape[-1])            # [nq, D]
    S, D = hidden.shape
    has_blocks = blocks is not None and blocks.shape[0] > 0
    if kops is not None:
        # One token, one fused kernel (decode step in a CUDA graph)
        return kops.attn_res(blocks if has_blocks else None, hidden, query, eps)
    if S == 1:
        # One token (a decode step): plain matrix products
        st = rms_norm(hidden, None, eps)                                              # [1, D]
        if has_blocks:
            st = torch.cat([blocks[:, 0], st], dim = 0)                               # [n, D]
        p = torch.softmax(torch.matmul(q, st.t()) * (D ** -0.5), dim = -1)            # [nq, n]
        out = torch.matmul(p, st).unsqueeze(1)                                        # [nq, 1, D]
        return out[0] if single else out
    out = torch.empty((q.shape[0], S, D), dtype = torch.float32, device = hidden.device)
    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        st = rms_norm(hidden[s0:s1], None, eps).unsqueeze(0)
        if has_blocks:
            st = torch.cat([blocks[:, s0:s1], st], dim = 0)                           # [n, s, D]
        scores = torch.einsum("qd,nsd->sqn", q, st) * (D ** -0.5)                     # [s, nq, n]
        p = torch.softmax(scores, dim = -1)
        out[:, s0:s1] = torch.einsum("sqn,nsd->qsd", p, st)
    return out[0] if single else out


def message_gate(x: torch.Tensor, message: torch.Tensor, w_down: torch.Tensor, w_forget: torch.Tensor,
                 w_gain: torch.Tensor, eps: float, adt: torch.dtype, kops = None) -> torch.Tensor:
    """
    (1 - forget) * x + gain * message, fp32 [S, D]. forget and gain are tanh of rank-r projections of the
    RMS-normalized [x | message]. w_down: [r, 2D] (applied in `adt`), w_forget / w_gain: [D, r] (fp32)
    """
    if kops is not None:
        return kops.message_gate(x, message, w_down, w_forget, w_gain, eps)
    xf, mf = x.float(), message.float()
    g = torch.cat([rms_norm(xf, None, eps), rms_norm(mf, None, eps)], dim = -1)
    g = F.linear(g.to(adt), w_down.to(adt)).float()
    forget = torch.tanh(F.linear(g, w_forget.float()))
    gain = torch.tanh(F.linear(g, w_gain.float()))
    return (1.0 - forget) * xf + gain * mf


def short_conv(x: torch.Tensor, w: torch.Tensor, k: int, adt: torch.dtype, state: torch.Tensor | None = None):
    """
    Causal depthwise conv + residual, / sqrt(k + 1). x: [S, C], w: [C, 1, k].
    state: the previous k-1 raw inputs [k-1, C] (incremental decode). Returns adt [S, C], or (out, new_state)
    when state is given
    """
    xf = x.float()
    S, C = xf.shape
    if REF_EXACT and state is None:
        y = F.conv1d(xf.t().unsqueeze(0), w.float().view(C, 1, k), padding = k - 1, groups = C)[0, :, :S].t()
    else:
        # Explicit taps instead of F.conv1d: cuDNN convolutions run in TF32 by default (10-bit mantissa) and pick
        # their algorithm by input length, so the same token got different values in passes of different lengths
        # (2-3% at the logits after 48 layer applications). Tap j weights the input k - 1 - j positions back
        hist = xf.new_zeros((k - 1, C)) if state is None else state.float()
        ctx = torch.cat([hist, xf], 0)
        wf = w.float().view(C, k)
        if S == 1:
            y = (ctx * wf.t()).sum(0, keepdim = True)
        else:
            y = ctx[0:S] * wf[:, 0]
            for j in range(1, k):
                y = y + ctx[j : j + S] * wf[:, j]
    out = ((y + xf) / math.sqrt(k + 1)).to(adt)
    if state is None:
        return out
    return out, torch.cat([state, x.to(state.dtype)], 0)[-(k - 1):]


def dyn_short_conv(x: torch.Tensor, dyn_in: torch.Tensor, static_w: torch.Tensor, w_proj: torch.Tensor,
                   n_heads: int, k: int, adt: torch.dtype, state: torch.Tensor | None = None,
                   dyn_taps: torch.Tensor | None = None):
    """
    Token-dependent causal conv (global layers): taps = static per-channel taps + per-token taps projected
    from `dyn_in`, shared within each K/V head. x: [S, C]; static_w: [k, C]; w_proj: [n_heads * k, hidden].
    Tap index is the backward distance. / sqrt(2k + 1)
    """
    S, C = x.shape
    # dyn_taps: the projection of dyn_in when the caller has it already (it is stacked with the K / V projection)
    dyn = (F.linear(dyn_in, w_proj) if dyn_taps is None else dyn_taps).float().view(S, n_heads, 1, k)
    weights = static_w.float().t().reshape(n_heads, -1, k) + dyn                       # [S, nh, C/nh, k]
    xf = x.float()
    ctx = F.pad(xf.t(), (k - 1, 0)) if state is None else torch.cat([state.float(), xf], 0).t()   # [C, S + k - 1]
    win = ctx.unfold(-1, k, 1).flip(-1).transpose(0, 1).reshape(S, n_heads, -1, k)    # [..., lag]
    y = (win * weights).sum(-1).reshape(S, C)
    out = ((y + xf) / math.sqrt(2 * k + 1)).to(adt)
    if state is None:
        return out
    return out, torch.cat([state, x.to(state.dtype)], 0)[-(k - 1):]


ATTN_MAX_ELEMS = 1 << 26        # logits held at once: heads * queries * keys (fp32: 256 MB)
ATTN_KEY_BLOCK = 16384          # keys converted to fp32 at a time, when K / V are too large to convert whole
ATTN_KV_FP32_BYTES = 3 << 29    # K (and V) of a call are converted to fp32 whole up to this size (1.5 GiB each)
ATTN_WINDOW_CHUNK = 512         # queries per chunk on the sliding-window layers


def attention_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rel: torch.Tensor, sinks: torch.Tensor,
                   proj: torch.Tensor, window: int | None, extent: int, scale: float,
                   max_elems: int | None = None, q_off: int = 0) -> torch.Tensor:
    """
    Single-sequence causal attention with a relative-position logit bias and sink logits.
      q: [S, H, dh], k/v: [M, Hk, dh]; query i sits at key index q_off + i (q_off > 0 when k holds history)
      rel: [S, H, d_rel], proj: [d_rel, extent]: bias[h, i, j] = (rel[i, h] @ proj)[i - j] for 0 <= i - j < extent
      sinks: [S, H]: one extra softmax logit per query with a zero value
      window: keys with distance >= window are masked (None: all earlier keys visible)
    Returns fp32 [S, H, dh]. Logits are fp32. Memory is bounded for long histories: queries go in chunks so that
    heads * chunk * keys stays under max_elems, and the bias is only formed for the keys within `extent` of the
    chunk.
    """
    S, H, dh = q.shape
    M, Hk = k.shape[0], k.shape[1]
    G = H // Hk
    out = torch.empty((S, H, dh), dtype = torch.float32, device = q.device)
    if REF_EXACT:
        return _attention_core_ref(q, k, v, rel, sinks, proj, window, extent, scale, q_off)
    max_elems = max_elems or ATTN_MAX_ELEMS
    relf, projf, sinkf = rel.float(), proj.float(), sinks.float()
    if S == 1 and M <= ATTN_KEY_BLOCK:
        # One query (a decode step): every key from the window start on is visible, and the key d positions back
        # gets bias column d
        a1 = q_off + 1
        j0 = 0 if window is None else max(0, a1 - window)
        kl = a1 - j0
        qg = q.float().view(Hk, G, dh)
        sc = torch.bmm(qg, k[j0:a1].float().permute(1, 2, 0)).view(H, kl) * scale
        n = min(kl, extent)
        sc[:, kl - n:] += torch.matmul(relf[0], projf)[:, :n].flip(-1)
        sk = sinkf.view(H, 1)
        m = torch.maximum(sc.amax(-1, keepdim = True), sk)
        sc = torch.exp(sc - m)
        sc = sc / (sc.sum(-1, keepdim = True) + torch.exp(sk - m))
        o = torch.bmm(sc.view(Hk, G, kl), v[j0:a1].float().permute(1, 0, 2))
        return o.view(1, H, dh)

    # Many queries (prefill). The work is passes over the logits [H, queries, keys], so:
    #   - sliding layers take the queries in chunks of ATTN_WINDOW_CHUNK: a chunk of c queries needs c + window - 1
    #     keys, and with large chunks most of those logits are masked
    #   - bias and masks go in with one gather and one add: the bias table gets a zero column (beyond the extent)
    #     and a -inf column (future keys, keys outside the window)
    #   - the sink is folded in after the softmax: p' = softmax(logits) / (1 + exp(sink - logsumexp(logits)))
    #   - K and V are converted to fp32 once per call when they fit, not once per chunk of queries
    qf = q.float() * scale
    whole = M * Hk * dh * 4 <= ATTN_KV_FP32_BYTES
    kf_all = k.float().permute(1, 2, 0).contiguous() if whole else None              # [Hk, dh, M]
    vf_all = v.float().permute(1, 0, 2).contiguous() if whole else None              # [Hk, M, dh]
    kv_max = M if window is None else min(M, ATTN_WINDOW_CHUNK + window)
    qc = max(16, min(2048 if window is None else ATTN_WINDOW_CHUNK, (max_elems // (H * max(kv_max, 1))) // 16 * 16))
    neg = torch.full((1, 1, 1), float("-inf"), dtype = torch.float32, device = q.device)
    for i0 in range(0, S, qc):
        i1 = min(S, i0 + qc)
        a0, a1 = q_off + i0, q_off + i1
        j0 = 0 if window is None else max(0, a0 - window + 1)
        nq, kl = i1 - i0, a1 - j0
        qg = qf[i0:i1].reshape(nq, Hk, G, dh).permute(1, 2, 0, 3).reshape(Hk, G * nq, dh)

        if whole:
            sc = torch.bmm(qg, kf_all[:, :, j0:a1]).view(H, nq, kl)
        else:
            sc = torch.empty((H, nq, kl), dtype = torch.float32, device = q.device)
            for b0 in range(0, kl, ATTN_KEY_BLOCK):
                b1 = min(kl, b0 + ATTN_KEY_BLOCK)
                kg = k[j0 + b0 : j0 + b1].float().permute(1, 2, 0)                 # [Hk, dh, block]
                sc[:, :, b0:b1] = torch.bmm(qg, kg).reshape(H, nq, b1 - b0)

        # Keys within `extent` of the chunk: bias, causal mask and window mask in one gather
        n0 = max(j0, a0 - extent + 1)
        dist = torch.arange(a0, a1, device = q.device)[:, None] - torch.arange(n0, a1, device = q.device)[None, :]
        idx = dist.clamp(max = extent)                                            # extent: the zero column
        hidden_keys = dist < 0
        if window is not None:
            hidden_keys = hidden_keys | (dist >= window)
        idx = idx.masked_fill(hidden_keys, extent + 1)                            # extent + 1: the -inf column
        R = torch.matmul(relf[i0:i1], projf).permute(1, 0, 2)                      # [H, nq, extent]
        R = torch.cat([R, torch.zeros_like(R[:, :, :1]), neg.expand(H, nq, 1)], dim = -1)
        sc[:, :, n0 - j0:] += R.gather(2, idx.unsqueeze(0).expand(H, -1, -1))
        del R, idx, dist, hidden_keys

        lse = torch.logsumexp(sc, dim = -1, keepdim = True)                       # [H, nq, 1]
        pw = torch.softmax(sc, dim = -1)
        del sc
        pw = pw.view(Hk, G * nq, kl)
        if whole:
            o = torch.bmm(pw, vf_all[:, j0:a1])
        else:
            o = torch.zeros((Hk, G * nq, dh), dtype = torch.float32, device = q.device)
            for b0 in range(0, kl, ATTN_KEY_BLOCK):
                b1 = min(kl, b0 + ATTN_KEY_BLOCK)
                o += torch.bmm(pw[:, :, b0:b1], v[j0 + b0 : j0 + b1].float().permute(1, 0, 2))
        del pw
        o = o.view(H, nq, dh) / (1.0 + torch.exp(sinkf[i0:i1].t().unsqueeze(-1) - lse))
        out[i0:i1] = o.view(Hk, G, nq, dh).permute(2, 0, 1, 3).reshape(nq, H, dh)
        del o
    return out


def attention_one(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rel: torch.Tensor, sinks: torch.Tensor,
                  proj: torch.Tensor, extent: int, scale: float, d: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """
    attention_core for one query over a key buffer of fixed capacity, with no shape or value read on the host, so
    that a decode step can be captured in a CUDA graph. q: [1, H, dh], k/v: [C, Hk, dh], d: long [C], the distance
    from the query back to each key, visible: bool [C]. Returns fp32 [1, H, dh].
    """
    H, dh = q.shape[1], q.shape[2]
    C, Hk = k.shape[0], k.shape[1]
    G = H // Hk
    sc = torch.bmm(q.float().view(Hk, G, dh), k.float().permute(1, 2, 0)).view(H, C) * scale
    R = torch.matmul(rel.float()[0], proj.float())                                 # [H, extent]
    sc = sc + R.index_select(1, d.clamp(0, extent - 1)) * (d < extent)
    sc = sc.masked_fill(~visible, float("-inf"))
    sk = sinks.float().view(H, 1)
    m = torch.maximum(sc.amax(-1, keepdim = True), sk)
    sc = torch.exp(sc - m)
    sc = sc / (sc.sum(-1, keepdim = True) + torch.exp(sk - m))
    o = torch.bmm(sc.view(Hk, G, C), v.float().permute(1, 0, 2))
    return o.view(1, H, dh)


def _attention_core_ref(q, k, v, rel, sinks, proj, window, extent, scale, q_off):
    """attention_core with the reference's arithmetic (REF_EXACT): everything but the softmax in the input dtype"""
    S, H, dh = q.shape
    M, Hk = k.shape[0], k.shape[1]
    G = H // Hk
    adt = q.dtype
    out = torch.empty((S, H, dh), dtype = torch.float32, device = q.device)
    kk = k.permute(1, 0, 2).repeat_interleave(G, 0)                               # [H, M, dh]
    vv = v.permute(1, 0, 2).repeat_interleave(G, 0)
    qc = max(16, min(1024, ((1 << 26) // (H * max(M, 1))) // 16 * 16))
    kj = torch.arange(M, device = q.device)
    for i0 in range(0, S, qc):
        i1 = min(S, i0 + qc)
        dist = torch.arange(q_off + i0, q_off + i1, device = q.device)[:, None] - kj[None, :]
        vis = dist >= 0
        if window is not None:
            vis &= dist < window
        sc = torch.matmul(q[i0:i1].permute(1, 0, 2), kk.transpose(1, 2)) * scale   # [H, nq, M]
        R = torch.matmul(rel[i0:i1], proj.to(adt)).permute(1, 0, 2)                # [H, nq, extent]
        bias = R.gather(2, dist.clamp(0, extent - 1).unsqueeze(0).expand(H, -1, -1))
        bias = bias.masked_fill(((dist < 0) | (dist >= extent)).unsqueeze(0), 0.0)
        mask = torch.zeros(dist.shape, dtype = adt, device = q.device).masked_fill(~vis, torch.finfo(adt).min)
        sc = sc + bias
        sc = sc + mask
        comb = torch.cat([sc, sinks[i0:i1].t().unsqueeze(-1)], dim = -1).float()
        pw = torch.softmax(comb, dim = -1)[..., :-1].to(adt)
        out[i0:i1] = torch.matmul(pw, vv).permute(1, 0, 2).float()
    return out


# ----------------------------------------------------------------------------------------------------------
# Weights. `get(key)` returns the tensor of the source checkpoint on the target device, in its stored dtype

GATE_NAMES = ("attn", "k", "v", "mlp")          # checkpoint names of the Q, K, V and MLP message gates


class LayerWeights:
    """Non-expert tensors of one decoder layer (bf16 as stored; the small ones are used in fp32)"""

    def __init__(self, get, cfg: Inkling2Cfg, i: int):
        self.cfg, self.i = cfg, i
        self.is_global = cfg.is_global(i)
        p = f"model.llm.layers.{i}."
        a = p + "attn."
        self.wq, self.wk, self.wv = get(a + "wq_du.weight"), get(a + "wk_dv.weight"), get(a + "wv_dv.weight")
        self.wr, self.wo = get(a + "wr_du.weight"), get(a + "wo_ud.weight")
        self.wgate, self.wsink = get(a + "gate_attn.weight"), get(a + "attn_sink_proj.weight")
        self.q_scale, self.rel_scale = get(a + "q_l2_clip.scale"), get(a + "rel_l2_clip.scale")
        self.rel_proj = get(a + "rel_logits_proj.proj")
        if self.is_global:
            self.kc_static, self.kc_proj = get(a + "k_sconv.static_w"), get(a + "k_sconv.weight_proj")
            self.vc_static, self.vc_proj = get(a + "v_sconv.static_w"), get(a + "v_sconv.weight_proj")
            self.q_norm, self.k_norm, self.v_norm = (
                get(p + "q_input_norm.weight"), get(p + "k_input_norm.weight"), get(p + "v_input_norm.weight")
            )
        else:
            self.kc_w, self.vc_w = get(a + "k_sconv.weight"), get(a + "v_sconv.weight")
            self.attn_norm = get(p + "attn_norm.weight")
        self.attn_sconv = get(p + "attn_sconv.weight")
        self.mlp_sconv = get(p + "mlp_sconv.weight")
        self.mlp_norm = get(p + "mlp_norm.weight")
        self.router_w = get(p + "mlp.gate.weight")
        self.router_b = get(p + "mlp.gate.bias")
        self.attn_query = get(f"model.llm.attn_res.attn_queries.{i - 1}") if i > 0 else None
        self.mlp_query = get(f"model.llm.attn_res.mlp_queries.{i}")
        # Loop layers: message gates (down, forget, gain) for the Q, K, V and MLP inputs, and the four message
        # queries [4, D] in the same order. The checkpoint keeps the queries of all loop layers in two tensors
        # of [2 * loop layers, D]: (Q, MLP) pairs and (K, V) pairs
        # Projections that read the same input, stacked into one matrix (one product instead of up to six); the
        # single projections stay available as row views of it. Query side: q, relative states, output gate, sinks
        self.n_q = [t.shape[0] for t in (self.wq, self.wr, self.wgate, self.wsink)]
        if self.is_global:
            self.w_qside = torch.cat([self.wq, self.wr, self.wgate, self.wsink], 0).contiguous()
            self.wk_cat = torch.cat([self.wk, self.kc_proj], 0).contiguous()
            self.wv_cat = torch.cat([self.wv, self.vc_proj], 0).contiguous()
            self.wk, self.kc_proj = self.wk_cat[: self.wk.shape[0]], self.wk_cat[self.wk.shape[0]:]
            self.wv, self.vc_proj = self.wv_cat[: self.wv.shape[0]], self.wv_cat[self.wv.shape[0]:]
        else:
            self.w_all = torch.cat([self.wq, self.wr, self.wgate, self.wsink, self.wk, self.wv], 0).contiguous()
            self.w_qside = self.w_all[: sum(self.n_q)]
            self.wk = self.w_all[sum(self.n_q) : sum(self.n_q) + self.wk.shape[0]]
            self.wv = self.w_all[sum(self.n_q) + self.wk.shape[0]:]
        o = 0
        views = []
        for n in self.n_q:
            views.append(self.w_qside[o : o + n])
            o += n
        self.wq, self.wr, self.wgate, self.wsink = views

        # The small tensors are only ever used in fp32; converting them once keeps the casts out of every forward.
        # The convolution taps as [C, k]
        f32 = lambda t: None if t is None else t.float().contiguous()
        self.q_scale, self.rel_scale, self.rel_proj = f32(self.q_scale), f32(self.rel_scale), f32(self.rel_proj)
        self.attn_query, self.mlp_query = f32(self.attn_query), f32(self.mlp_query)
        self.mlp_norm, self.router_w, self.router_b = f32(self.mlp_norm), f32(self.router_w), f32(self.router_b)
        self.attn_sconv, self.mlp_sconv = f32(self.attn_sconv), f32(self.mlp_sconv)
        as_taps = lambda t: t.view(t.shape[0], -1).contiguous()          # [C, 1, k] -> [C, k]
        self.attn_sconv, self.mlp_sconv = as_taps(self.attn_sconv), as_taps(self.mlp_sconv)
        if not self.is_global:
            self.kc_w, self.vc_w = as_taps(f32(self.kc_w)), as_taps(f32(self.vc_w))
        if self.is_global:
            self.kc_static, self.vc_static = f32(self.kc_static), f32(self.vc_static)
            # The static taps [k, C] (tap = backward distance) as [C, k] with the current token last, the layout
            # of the other convolutions (for the fused convolution kernel)
            self.kc_fwd = self.kc_static.flip(0).t().contiguous()
            self.vc_fwd = self.vc_static.flip(0).t().contiguous()
            self.q_norm, self.k_norm, self.v_norm = f32(self.q_norm), f32(self.k_norm), f32(self.v_norm)
        else:
            self.attn_norm = f32(self.attn_norm)

        self.gates = self.msg_query = None
        if cfg.is_loop(i) and cfg.passes > 1:
            c = p + "controllers."
            self.gates = [
                (get(f"{c}{n}.down_weight"), f32(get(f"{c}{n}.forget_weight")), f32(get(f"{c}{n}.gain_weight")))
                for n in GATE_NAMES
            ]
            j = i - cfg.loop_start
            qm = get("model.llm.middle_cycle_cross_pass_queries")[2 * j : 2 * j + 2]
            kv = get("model.llm.middle_cycle_cross_pass_kv_queries")[2 * j : 2 * j + 2]
            self.msg_query = torch.cat([qm[:1], kv, qm[1:]], dim = 0).float().contiguous()


class HeadWeights:
    def __init__(self, get, cfg: Inkling2Cfg, with_lm_head: bool = True):
        self.cfg = cfg
        self.out_query = get("model.llm.attn_res.output_query")
        self.norm = get("model.llm.norm.weight")
        self.lm_head = get("model.llm.unembed.weight") if with_lm_head else None


class PackedExperts:
    """
    Routed experts of one layer in the packed form of the source checkpoint (NVFP4 or MXFP8), dequantized per
    expert on demand. The stored w13 rows alternate gate / up.
    """

    def __init__(self, get, has, cfg: Inkling2Cfg, layer: int):
        self.cfg, self.layer = cfg, layer
        p = f"model.llm.layers.{layer}.mlp.experts."
        self.nvfp4 = has(p + "w13_weight_amax")
        self.w13, self.s13 = get(p + "w13_weight"), get(p + "w13_weight_scale")
        self.w2, self.s2 = get(p + "w2_weight"), get(p + "w2_weight_scale")
        if self.nvfp4:
            amax = (lambda t: t.to(torch.bfloat16).float()) if REF_EXACT else (lambda t: t.float())
            self.g13 = amax(get(p + "w13_weight_amax")) / get(p + "w13_weight_outer_scale_denominator").float()
            self.g2 = amax(get(p + "w2_weight_amax")) / get(p + "w2_weight_outer_scale_denominator").float()

    def w13_f32(self, e: int) -> torch.Tensor:
        """[2 * inter, D], gate rows first"""
        if self.nvfp4:
            w = dequant_nvfp4(self.w13[e], self.s13[e], self.g13[e])
        else:
            w = dequant_mxfp8(self.w13[e], self.s13[e])
        return torch.cat([w[0::2], w[1::2]], dim = 0)

    def w2_f32(self, e: int) -> torch.Tensor:
        """[D, inter]"""
        if self.nvfp4:
            return dequant_nvfp4(self.w2[e], self.s2[e], self.g2[e])
        return dequant_mxfp8(self.w2[e], self.s2[e])

    def gate_up(self, e, x): return F.linear(x, self.w13_f32(e).to(x.dtype))
    def down(self, e, a): return F.linear(a, self.w2_f32(e).to(a.dtype))


# ----------------------------------------------------------------------------------------------------------
# Layer computation

def layer_attention(W: LayerWeights, hidden: torch.Tensor, blocks: torch.Tensor | None, adt = torch.bfloat16,
                    cache = None, pos0: int = 0, msgs: torch.Tensor | None = None):
    """
    Attention half of a decoder layer for one sequence. hidden: [S, D] adt, blocks: [nb, S, D] block states or None,
    msgs: fp32 [4, S, D] (Q, K, V, MLP messages) in the passes after the first.
    Returns the post-attention-conv output [S, D] (adt).
    """
    cfg, i = W.cfg, W.i
    S = hidden.shape[0]
    H, Hk, dh, dr = cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.d_rel
    # Fused kernels for one token (decode step in a CUDA graph), when the cache offers them
    kops = getattr(cache, "kops", None)
    gate = lambda n, x: message_gate(x, msgs[n], *W.gates[n], cfg.eps, adt, kops)
    if W.is_global:
        if i == 0:
            xq = xk = xv = hidden.float()
        else:
            xq, xk, xv = attn_res(blocks, hidden, W.attn_query, cfg.eps, kops = kops)   # each fp32 [S, D]
        if msgs is not None:
            xq, xk, xv = gate(0, xq), gate(1, xk), gate(2, xv)
        xq = rms_norm(xq, W.q_norm, cfg.eps).to(adt)
        xk = rms_norm(xk, W.k_norm, cfg.eps).to(adt)
        xv = rms_norm(xv, W.v_norm, cfg.eps).to(adt)
    else:
        x = hidden.float() if i == 0 else attn_res(blocks, hidden, W.attn_query, cfg.eps, kops = kops)
        if msgs is not None:
            xq = rms_norm(gate(0, x), W.attn_norm, cfg.eps).to(adt)
            xk = rms_norm(gate(1, x), W.attn_norm, cfg.eps).to(adt)
            xv = rms_norm(gate(2, x), W.attn_norm, cfg.eps).to(adt)
        else:
            xq = xk = xv = rms_norm(x, W.attn_norm, cfg.eps).to(adt)

    # Projections: stacked matrices per input (see LayerWeights). The reference-exact mode keeps them apart
    dk = dv = None
    if REF_EXACT:
        q, k, v = F.linear(xq, W.wq), F.linear(xk, W.wk), F.linear(xv, W.wv)
        r, ogate, sinks = F.linear(xq, W.wr), F.linear(xq, W.wgate), F.linear(xq, W.wsink)
    else:
        cut = (lambda t: t) if S == 1 else (lambda t: t.contiguous())
        if not W.is_global and xk is xq and xv is xq:
            parts = F.linear(xq, W.w_all).split(W.n_q + [W.wk.shape[0], W.wv.shape[0]], dim = -1)
            q, r, ogate, sinks, k, v = (cut(t) for t in parts)
        else:
            q, r, ogate, sinks = (cut(t) for t in F.linear(xq, W.w_qside).split(W.n_q, dim = -1))
            if W.is_global:
                k, dk = (cut(t) for t in F.linear(xk, W.wk_cat).split([W.wk.shape[0], W.kc_proj.shape[0]], dim = -1))
                v, dv = (cut(t) for t in F.linear(xv, W.wv_cat).split([W.wv.shape[0], W.vc_proj.shape[0]], dim = -1))
            else:
                k, v = F.linear(xk, W.wk), F.linear(xv, W.wv)
    cs = cache.conv if cache is not None else {}
    if kops is not None:
        # Convolution, ring read and ring write in one kernel each; fp32 results
        ck = cfg.conv_k
        if W.is_global:
            sc = 1.0 / math.sqrt(2 * ck + 1)
            dk = dk.float().view(S, Hk, ck).flip(-1).contiguous()
            dv = dv.float().view(S, Hk, ck).flip(-1).contiguous()
            k = cache.conv_step("k", k, W.kc_fwd, sc, dk)
            v = cache.conv_step("v", v, W.vc_fwd, sc, dv)
        else:
            sc = 1.0 / math.sqrt(ck + 1)
            k = cache.conv_step("k", k, W.kc_w, sc)
            v = cache.conv_step("v", v, W.vc_w, sc)
    else:
        if cache is not None:
            cache.record("k", k)
            cache.record("v", v)
        if W.is_global:
            k = dyn_short_conv(k, xk, W.kc_static, W.kc_proj, Hk, cfg.conv_k, adt, cs.get("k"), dk)
            v = dyn_short_conv(v, xv, W.vc_static, W.vc_proj, Hk, cfg.conv_k, adt, cs.get("v"), dv)
        else:
            k = short_conv(k, W.kc_w, cfg.conv_k, adt, cs.get("k"))
            v = short_conv(v, W.vc_w, cfg.conv_k, adt, cs.get("v"))
        if cache is not None:
            k, v = k[0], v[0]

    if kops is not None and kops.has_qkv:
        # Clips, scales, value norm and the K / V cache writes in one kernel, then the attention kernel
        qf, relf = cache.qkv_post(W, q, r, k, v)
        o = cache.attend(qf, relf, sinks, W.rel_proj, dh ** -0.5).to(adt)
        o = (o.reshape(S, H * dh).float() * torch.sigmoid(ogate.float())).to(adt)
        o = F.linear(o, W.wo)
        return cache.conv_step("attn", o, W.attn_sconv, 1.0 / math.sqrt(cfg.conv_k + 1))

    q = l2_clip(q.view(S, H, dh), cfg.qk_clip, W.q_scale)
    rel = l2_clip(r.view(S, H, dr), cfg.rel_clip, W.rel_scale)
    if W.is_global:
        if torch.is_tensor(pos0):
            pos = pos0.float().view(1)                      # one token, position given on the device (graph)
        else:
            pos = torch.arange(pos0, pos0 + S, device = hidden.device, dtype = torch.float32)
        tau = 1.0 + cfg.log_alpha * torch.log(((pos + 1.0) / cfg.log_n_floor).clamp(min = 1.0))
        tau = tau[:, None, None]
        q = q.to(adt).float() * tau
        rel = rel.to(adt).float() * tau
    q, rel = q.to(adt), rel.to(adt)
    k = l2_clip(k.view(S, Hk, dh), cfg.qk_clip, None).to(adt)
    vf = v.view(S, Hk, dh).float()
    v = (vf * torch.rsqrt(vf.square().mean(-1, keepdim = True) + cfg.eps)).to(adt)

    if ROUND_KV:
        k, v = k.half().to(adt), v.half().to(adt)
    q_off = 0
    k_all, v_all = k, v
    if cache is not None:
        k_all, v_all, q_off = cache.extend(k, v, adt)
    if q_off is None and k_all is None:
        # One token, attention kernel reading the keys in place (decode step in a CUDA graph)
        o = cache.attend(q, rel, sinks, W.rel_proj, dh ** -0.5).to(adt)
    elif q_off is None:
        # Static form (a decode step inside a CUDA graph): fixed-capacity keys, distances and visibility as data
        o = attention_one(q, k_all, v_all, rel, sinks, W.rel_proj, cfg.rel_extent, dh ** -0.5, cache.d, cache.visible).to(adt)
    else:
        o = attention_core(
            q, k_all, v_all, rel, sinks, W.rel_proj, None if W.is_global else cfg.window, cfg.rel_extent,
            dh ** -0.5, q_off = q_off
        ).to(adt)
    o = o.reshape(S, H * dh)
    o = (o.float() * torch.sigmoid(ogate.float())).to(adt)
    o = F.linear(o, W.wo)
    if cache is None:
        return short_conv(o, W.attn_sconv, cfg.conv_k, adt)
    if kops is not None:
        return cache.conv_step("attn", o, W.attn_sconv, 1.0 / math.sqrt(cfg.conv_k + 1))
    cache.record("attn", o)
    return short_conv(o, W.attn_sconv, cfg.conv_k, adt, cs["attn"])[0]


def layer_mlp_input(W: LayerWeights, hidden: torch.Tensor, blocks: torch.Tensor | None, attn_out: torch.Tensor,
                    adt = torch.bfloat16, msgs: torch.Tensor | None = None, kops = None):
    """
    Everything between the attention output and the experts. Returns (resid fp32 [S, D], x_mlp adt [S, D], the
    block states after this layer). At a block boundary the incoming hidden state becomes a block state and is not
    added to the residual.
    """
    cfg, i = W.cfg, W.i
    a = attn_out.float()
    if cfg.is_boundary(i):
        resid = a
        new = block_state(hidden, cfg.eps)
        bl = new if blocks is None or blocks.shape[0] == 0 else torch.cat([blocks, new], 0)
    else:
        resid = hidden.float() + a
        bl = blocks
    x = attn_res(bl, resid, W.mlp_query, cfg.eps, kops = kops)
    if msgs is not None:
        x = message_gate(x, msgs[3], *W.gates[3], cfg.eps, adt, kops)
    x = rms_norm(x, W.mlp_norm, cfg.eps).to(adt)
    return resid, x, bl


def route(W: LayerWeights, x: torch.Tensor, pos0: int = 0):
    """Sigmoid router. x: [T, D]. Returns (indices [T, k], fp32 weights [T, k] including route_scale)"""
    cfg = W.cfg
    scores = torch.sigmoid(F.linear(x.float(), W.router_w.float()))
    sel = torch.topk(scores + W.router_b.float(), cfg.topk, dim = -1, sorted = False).indices
    if ROUTE_HOOK is not None:
        sel = ROUTE_HOOK(W, pos0, sel)
    w = scores.gather(1, sel)
    w = w / (w.sum(-1, keepdim = True) + 1e-20)
    return sel, w * cfg.route_scale


def expert_act(gu: torch.Tensor, cfg: Inkling2Cfg, adt) -> torch.Tensor:
    """gu: [n, 2 * inter], gate in the first half, up in the second"""
    g, u = gu.float().chunk(2, dim = -1)
    return (softcap(g, cfg.gate_cap) * torch.sigmoid(g) * softcap(u, cfg.up_cap)).to(adt)


def moe_forward(x: torch.Tensor, sel: torch.Tensor, rw: torch.Tensor, ex, cfg: Inkling2Cfg,
                adt = torch.bfloat16, W = None) -> torch.Tensor:
    """Routed experts. x: [T, D] adt, sel / rw: [T, k]. Returns [T, D] adt (fp32 accumulation, like the reference)"""
    T, D = x.shape
    k = sel.shape[1]
    out = torch.zeros((T, D), dtype = torch.float32, device = x.device)
    flat = sel.reshape(-1)
    order = torch.argsort(flat, stable = True)
    counts = torch.bincount(flat, minlength = cfg.experts).tolist()
    tok_sorted = order // k
    w_sorted = rw.reshape(-1)[order]
    pos = 0
    for e, n in enumerate(counts):
        if n == 0:
            continue
        tok = tok_sorted[pos : pos + n]
        we = w_sorted[pos : pos + n]
        pos += n
        xe = x[tok]
        a = expert_act(ex.gate_up(e, xe), cfg, adt)
        if MOE_HOOK is not None:
            MOE_HOOK(W, e, xe, a)
        y = ex.down(e, a)
        out.index_add_(0, tok, y.float() * we.unsqueeze(1))
    return out.to(adt)


def layer_finish(W: LayerWeights, resid: torch.Tensor, moe_out: torch.Tensor, adt = torch.bfloat16,
                 cache = None) -> torch.Tensor:
    if cache is None:
        y = short_conv(moe_out, W.mlp_sconv, W.cfg.conv_k, adt)
    elif getattr(cache, "kops", None) is not None:
        y = cache.conv_step("mlp", moe_out, W.mlp_sconv, 1.0 / math.sqrt(W.cfg.conv_k + 1))
    else:
        cache.record("mlp", moe_out)
        y = short_conv(moe_out, W.mlp_sconv, W.cfg.conv_k, adt, cache.conv["mlp"])[0]
    return (resid + y.float()).to(adt)


def layer_forward(W: LayerWeights, ex, hidden, blocks, adt = torch.bfloat16, cache = None, pos0: int = 0,
                  msgs: torch.Tensor | None = None, trace: dict | None = None):
    """One decoder layer application for one sequence. Returns (new_hidden, new_blocks)"""
    cfg, i = W.cfg, W.i
    a = layer_attention(W, hidden, blocks, adt, cache, pos0, msgs)
    resid, x, blocks = layer_mlp_input(W, hidden, blocks, a, adt, msgs, getattr(cache, "kops", None))
    if hasattr(ex, "moe"):
        # The experts run the whole sparse MLP themselves, router included (EXL3 kernels)
        y = ex.moe(x, adt)
    else:
        sel, rw = route(W, x, pos0)
        if trace is not None:
            trace["route"] = sel
        y = moe_forward(x, sel, rw, ex, cfg, adt, W)
    new_hidden = layer_finish(W, resid, y, adt, cache)
    return new_hidden, blocks


class LoopState:
    """
    Bookkeeping for the looped layers, driven once per layer application in execution order:

        msgs = state.enter(cfg, W, hidden, blocks)       # may replace hidden / blocks: see below
        hidden, blocks = layer_forward(W, ex, state.hidden, state.blocks, ..., msgs = msgs)
        state.leave(hidden, blocks)

    At the first loop layer of the first pass it saves the hidden and block states. At the first loop layer of
    each later pass it keeps the previous pass's block states and final hidden state as message sources and
    restarts from the saved states.
    """

    def __init__(self):
        self.hidden = self.blocks = None
        self.pre = None                 # (hidden, blocks) before the loop
        self.src = None                 # (blocks, hidden) of the previous pass: message keys and values

    def enter(self, cfg: Inkling2Cfg, W: LayerWeights, hidden, blocks, loop_pass: int, kops = None):
        if cfg.is_loop(W.i) and W.i == cfg.loop_start:
            if loop_pass == 0:
                self.pre = (hidden, blocks)
                self.src = None
            else:
                # Block states the previous pass appended, without the first: that one is the pre-loop hidden
                # state, the restart point of every pass
                npre = 0 if self.pre[1] is None else self.pre[1].shape[0]
                self.src = (blocks[npre + 1:], hidden)
                hidden, blocks = self.pre
        self.hidden, self.blocks = hidden, blocks
        if cfg.is_loop(W.i) and loop_pass > 0:
            return attn_res(self.src[0], self.src[1], W.msg_query, cfg.eps, kops = kops)   # fp32 [4, S, D]
        return None

    def leave(self, cfg: Inkling2Cfg, W: LayerWeights, loop_pass: int):
        if W.i == cfg.loop_end - 1 and loop_pass == cfg.passes - 1:
            self.pre = self.src = None


def embed_tokens(embed: torch.Tensor, embed_norm: torch.Tensor, ids: torch.Tensor, eps: float, adt = torch.bfloat16):
    return rms_norm(embed[ids], embed_norm, eps).to(adt)


def final_hidden(Hd: HeadWeights, hidden, blocks, adt = torch.bfloat16):
    x = attn_res(blocks, hidden, Hd.out_query, Hd.cfg.eps)
    return rms_norm(x, Hd.norm, Hd.cfg.eps).to(adt)


def logits_from_hidden(Hd: HeadWeights, h: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """h: [S, D] -> fp32 [S, unpadded vocab], soft-capped. In row chunks: the logits of a long sequence are large"""
    cap, V = Hd.cfg.final_cap, Hd.cfg.unpadded_vocab
    out = torch.empty((h.shape[0], V), dtype = torch.float32, device = h.device)
    for a in range(0, h.shape[0], chunk):
        z = F.linear(h[a : a + chunk], Hd.lm_head)[:, :V].float()
        out[a : a + chunk] = z.div_(cap).tanh_().mul_(cap)
    return out
