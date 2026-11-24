#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>

// Local patch: Inkling-2 decode-step kernels, for S consecutive new tokens of one sequence (S is small: one token
// when decoding, a few when verifying drafted tokens). Each replaces a chain of small PyTorch operations of
// modules/inkling2.py with one or two launches on the current stream; none reads anything on the host, so they can
// be captured in a CUDA graph. All arithmetic is fp32.
//
// ctl: long [2] on the device: the position of the first token, and its row in a sliding K/V buffer. Token s is at
// position ctl[0] + s and row ctl[1] + s.

// Short causal convolution with its ring of raw inputs. x: [S, C] (half or float), ring: half [R, C] (row =
// position % R), w: float [C, k], tap j weighting the input k - 1 - j positions back (the last tap is the token
// itself), dyn: optional float [S, heads, k] added to the taps of each head's channels.
// out[s, c] = (sum_j taps[c, j] * input[s - (k - 1) + j][c] + x[s, c]) * scale, float [S, C]; inputs before the
// first token come from the ring. Writes x into the ring rows of the tokens.
// dyn may instead be half [S, heads * k] with each head's taps in reverse order (the projection as it comes).
// add: optional [S, C] (half or float) added to the result (a residual). out may be half. round_x: a float x is
// rounded to fp16 before use (the value the ring keeps).
void i2_conv_step
(
    const at::Tensor& x,
    at::Tensor ring,
    const at::Tensor& w,
    const c10::optional<at::Tensor>& dyn,
    const at::Tensor& ctl,
    at::Tensor out,
    float scale,
    const c10::optional<at::Tensor>& add = c10::nullopt,
    bool round_x = false
);

// RMS norm of each row of x ([..., D], half or float) times an optional gain w (float [D], or [nw, D]: consecutive
// groups of rows / nw rows take gain 0, 1, ...). out: the same number of elements, half or float.
void i2_norm
(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& w,
    at::Tensor out,
    float eps
);

// Block attention residual. blocks: optional float [n, S, D] (already RMS-normalized), hidden: [S, D] (half or
// float), queries: float [nq, D], out: float [nq, S, D]. Per token, each query attends over the block states and
// the normalized hidden state (scores scaled by D^-0.5); out is the softmax-weighted mix of those states.
void i2_attn_res
(
    const c10::optional<at::Tensor>& blocks,
    const at::Tensor& hidden,
    const at::Tensor& queries,
    at::Tensor out,
    float eps
);

// Message gate: out = (1 - tanh(Wf g)) * x + tanh(Wg g) * m with g = Wd [rmsnorm(x) | rmsnorm(m)], per token.
// x, m: float [S, D], wd: half [r, 2D], wf, wg: the transposed weights, float [r, D], g: float [S, r] scratch,
// out: float [S, D].
void i2_message_gate
(
    const at::Tensor& x,
    const at::Tensor& m,
    const at::Tensor& wd,
    const at::Tensor& wf,
    const at::Tensor& wg,
    at::Tensor g,
    at::Tensor out,
    float eps
);

// Q / K / V post-processing and the K / V cache writes. Per token and head: q = clip(q_lin * q_scale) * tau,
// rel = clip(r_lin * rel_scale) * tau, K = clip(k), V = unit RMS norm of v; clip shortens a head vector to the
// threshold. tau = 1 + log_alpha * log(max((position + 1) / log_n_floor, 1)), or 1 when log_n_floor <= 0.
//   q_lin: half [S, H * dh], r_lin: half [S, H * dr], k, v: float [S, Hk * dh] (after their convolutions)
//   q_scale: float [H, dh], rel_scale: float [H, dr]; q_out: float [S, H, dh], rel_out: float [S, H, dr]
//   K and V go to k_store / v_store (half [rows, Hk, dh]) at each token's row: ctl[1] + s when local, else
//   base + position, or pages[position / 256] * 256 + position % 256 when pages (long, on the device) is given
void i2_qkv_post
(
    const at::Tensor& q_lin,
    const at::Tensor& r_lin,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& q_scale,
    const at::Tensor& rel_scale,
    at::Tensor k_store,
    at::Tensor v_store,
    const c10::optional<at::Tensor>& pages,
    const at::Tensor& ctl,
    at::Tensor q_out,
    at::Tensor rel_out,
    bool local,
    int64_t base,
    float qk_clip,
    float rel_clip,
    float log_alpha,
    float log_n_floor,
    float eps
);

// Attention with the relative-position bias and the sink, keys read in place from a K/V store (half [rows, Hk,
// dh]). Two launches: per (K/V head, block of 32 keys), for every token of the pass, the scores of the G query
// heads that share the K/V head, their block maximum, sum of exponentials and weighted value sums (K and V are
// read once for all S <= 8 tokens); then per (token, query head) the merge of its blocks with the sink logit. A
// token sees the keys at or before its own position.
//   q: float [S, H, dh], rel: float [S, H, dr], sinks: float [S, H], proj: float [dr, extent]
//   local: a token's keys are the last min(position + 1, window) rows ending at its row
//   else:  positions 0 .. position, at rows base + p, or through pages
//   part: float scratch [S * Hk * blocks * G * (2 + dh)], blocks = the key capacity / 32, or (window + S - 1) /
//         32 rounded up when local (the launch grid; blocks past the keys of the pass return at once)
//   out: float [S, H, dh]
//   sinks may be half. ogate: optional half [S, H * dh]: out is then half, the fp16-rounded attention output
//   times sigmoid(ogate)
void i2_attn_one
(
    const at::Tensor& q,
    const at::Tensor& rel,
    const at::Tensor& sinks,
    const at::Tensor& proj,
    const at::Tensor& k_store,
    const at::Tensor& v_store,
    const c10::optional<at::Tensor>& pages,
    const at::Tensor& ctl,
    at::Tensor part,
    at::Tensor out,
    bool local,
    int64_t base,
    int64_t window,
    int64_t capacity,
    float scale,
    const c10::optional<at::Tensor>& ogate = c10::nullopt
);

// Prefill attention (many queries of one sequence), in three steps that keep fp32 accuracy while the matrix
// products run on fp16 inputs (tensor cores, half the memory traffic of the fp32 forms):
//
//   i2_attn_qk    raw logits: q half [Hk, R, dh] (the R = G * nq query rows that share a K/V head), k half
//                 [M, Hk, dh] as stored (read in place), sc float [Hk, R, M]. fp16 inputs, fp32 accumulation
//   i2_attn_soft  per logit row: scale, relative bias, causal / window mask, and the exponentials relative to the
//                 row maximum (the sink logit included in the maximum). Row (h, i) of sc [H, nq, kl] is query
//                 position a0 + i against key positions j0 .. j0 + kl - 1. bias: float [nq, H, extent], indexed by
//                 distance; window 0 = no window. Writes e * 2^14 split into two fp16 terms, p half [Hk, 2, G, nq,
//                 kl] (term 0 the rounded value, term 1 the rounding remainder * 2^11), and den float [H, nq] =
//                 sum of the row's exponentials + the sink's
//   i2_attn_pv    weighted values: p half [Hk, R2, M], v half [M, Hk, dh] as stored, o float [Hk, R2, dh]
//
// attention output = (o[term 0] + o[term 1] * 2^-11) * 2^-14 / den
void i2_attn_qk(const at::Tensor& q, const at::Tensor& k, at::Tensor sc);
void i2_attn_soft
(
    const at::Tensor& sc,
    const at::Tensor& bias,
    const at::Tensor& sinks,
    at::Tensor p,
    at::Tensor den,
    int64_t a0,
    int64_t j0,
    int64_t window,
    float scale
);
void i2_attn_pv(const at::Tensor& p, const at::Tensor& v, at::Tensor o);
