#include <cuda_fp16.h>
#include "inkling2.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"

// Local patch: Inkling-2 decode-step kernels, see inkling2.cuh

#define I2_THREADS 256
#define I2_MAX_STATES 16
#define I2_MAX_TAPS 8
#define I2_MAX_RANK 64
#define I2C_X_ROUND 1
#define I2C_ADD_HALF 2
#define I2C_OUT_HALF 4
#define I2C_DYN_HALF_REV 8

// ---------------------------------------------------------------------------------------------------------------
// Convolution step

template <bool X_HALF>
__launch_bounds__(I2_THREADS)
__global__ void i2_conv_step_kernel
(
    const void* __restrict__ x_,
    half* __restrict__ ring,
    const float* __restrict__ w,
    const void* __restrict__ dyn,
    const int64_t* __restrict__ ctl,
    void* __restrict__ out,
    const void* __restrict__ add,
    const int S,
    const int C,
    const int R,
    const int k,
    const int heads,
    const float scale,
    const int flags
)
{
    const int idx = blockIdx.x * I2_THREADS + threadIdx.x;
    if (idx >= S * C) return;
    const int s = idx / C, c = idx % C;
    const int64_t pos = ctl[0];
    // A float input counts as its fp16 rounding (I2C_X_ROUND): the value the ring keeps, and the one a later step
    // or a prefill reads for this position
    #define I2_X(si) (X_HALF ? __half2float(((const half*) x_)[(size_t) (si) * C + c]) : \
        ((flags & I2C_X_ROUND) ? __half2float(__float2half_rn(((const float*) x_)[(size_t) (si) * C + c])) : ((const float*) x_)[(size_t) (si) * C + c]))
    const float xv = I2_X(s);
    // The token's taps: float [k] in tap order, or (I2C_DYN_HALF_REV) half [k] in reverse order, as projected
    const size_t dbase = ((size_t) s * heads + c / (C / heads)) * k;
    #define I2_DYN(j) (!dyn ? 0.0f : (flags & I2C_DYN_HALF_REV) ? __half2float(((const half*) dyn)[dbase + k - 1 - (j)]) : ((const float*) dyn)[dbase + (j)])
    float acc = 0.0f;
    for (int j = 0; j < k - 1; ++j)
    {
        // The input k - 1 - j positions back: one of the new tokens, or (before the first of them) the ring. The
        // ring rows read here are never among the rows this launch writes
        const int si = s - (k - 1) + j;
        const float in = si >= 0 ? I2_X(si) : __half2float(ring[(size_t) ((pos + si) % R) * C + c]);
        acc += (w[c * k + j] + I2_DYN(j)) * in;
    }
    acc += (w[c * k + k - 1] + I2_DYN(k - 1)) * xv;
    float res = (acc + xv) * scale;
    if (add)
        res = ((flags & I2C_ADD_HALF) ? __half2float(((const half*) add)[idx]) : ((const float*) add)[idx]) + res;
    if (flags & I2C_OUT_HALF) ((half*) out)[idx] = __float2half_rn(res);
    else ((float*) out)[idx] = res;
    ring[(size_t) ((pos + s) % R) * C + c] = __float2half_rn(xv);
    #undef I2_X
    #undef I2_DYN
}

void i2_conv_step
(
    const at::Tensor& x,
    at::Tensor ring,
    const at::Tensor& w,
    const c10::optional<at::Tensor>& dyn,
    const at::Tensor& ctl,
    at::Tensor out,
    float scale,
    const c10::optional<at::Tensor>& add,
    bool round_x
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dim() == 2, "i2_conv_step: x must be [S, C]");
    const int S = (int) x.size(0), C = (int) x.size(1);
    TORCH_CHECK(ring.dim() == 2 && ring.size(1) == C && ring.scalar_type() == at::kHalf, "i2_conv_step: ring");
    TORCH_CHECK(w.dim() == 2 && w.size(0) == C && w.scalar_type() == at::kFloat && w.is_contiguous(), "i2_conv_step: w");
    TORCH_CHECK(out.numel() == (int64_t) S * C && (out.scalar_type() == at::kFloat || out.scalar_type() == at::kHalf), "i2_conv_step: out");
    int flags = (round_x ? I2C_X_ROUND : 0) | (out.scalar_type() == at::kHalf ? I2C_OUT_HALF : 0);
    const void* ap = nullptr;
    if (add.has_value())
    {
        const at::Tensor& a = add.value();
        TORCH_CHECK(a.numel() == (int64_t) S * C && a.is_contiguous() && (a.scalar_type() == at::kFloat || a.scalar_type() == at::kHalf), "i2_conv_step: add");
        if (a.scalar_type() == at::kHalf) flags |= I2C_ADD_HALF;
        ap = a.data_ptr();
    }
    TORCH_CHECK(ctl.scalar_type() == at::kLong, "i2_conv_step: ctl");
    TORCH_CHECK(x.is_contiguous() && ring.is_contiguous() && out.is_contiguous(), "i2_conv_step: contiguity");
    const int k = (int) w.size(1);
    TORCH_CHECK(k >= 1 && k <= I2_MAX_TAPS && S + k - 1 <= ring.size(0), "i2_conv_step: taps / ring size");
    const void* dp = nullptr;
    int heads = 1;
    if (dyn.has_value())
    {
        const at::Tensor& d = dyn.value();
        TORCH_CHECK(d.is_contiguous() && d.size(0) == S && d.numel() % ((int64_t) S * k) == 0, "i2_conv_step: dyn");
        if (d.scalar_type() == at::kHalf)
            flags |= I2C_DYN_HALF_REV;      // half [S, heads * k], taps in reverse order (the projection's output)
        else
            TORCH_CHECK(d.dim() == 3 && d.size(2) == k && d.scalar_type() == at::kFloat, "i2_conv_step: dyn must be float [S, heads, k]");
        heads = (int) (d.numel() / ((int64_t) S * k));
        TORCH_CHECK(C % heads == 0, "i2_conv_step: heads");
        dp = d.data_ptr();
    }
    const int blocks = CEIL_DIVIDE(S * C, I2_THREADS);
    if (x.scalar_type() == at::kHalf)
        i2_conv_step_kernel<true><<<blocks, I2_THREADS, 0, stream>>>(
            x.data_ptr(), (half*) ring.data_ptr(), (const float*) w.data_ptr(), dp, (const int64_t*) ctl.data_ptr(),
            out.data_ptr(), ap, S, C, (int) ring.size(0), k, heads, scale, flags);
    else
    {
        TORCH_CHECK(x.scalar_type() == at::kFloat, "i2_conv_step: x dtype");
        i2_conv_step_kernel<false><<<blocks, I2_THREADS, 0, stream>>>(
            x.data_ptr(), (half*) ring.data_ptr(), (const float*) w.data_ptr(), dp, (const int64_t*) ctl.data_ptr(),
            out.data_ptr(), ap, S, C, (int) ring.size(0), k, heads, scale, flags);
    }
    cuda_check(cudaPeekAtLastError());
}

// ---------------------------------------------------------------------------------------------------------------
// Block attention residual: one block per query

#define I2R_THREADS 1024
#define I2R_WARPS (I2R_THREADS / 32)

// One block of 1024 threads per (query, token): a thread takes two or three channels, and the partial sums are
// reduced with warp shuffles (per warp, then over the warps by the first warp)
template <bool H_HALF>
__launch_bounds__(I2R_THREADS)
__global__ void i2_attn_res_kernel
(
    const float* __restrict__ blocks,
    const void* hidden_,
    const float* __restrict__ queries,
    float* __restrict__ out,
    const int n,
    const int S,
    const int D,
    const float eps
)
{
    // partial sums per warp: [0] = sum of squares of hidden, [1] = q . hidden, [2 + s] = q . block s
    __shared__ float part[I2_MAX_STATES + 2][I2R_WARPS];
    __shared__ float tot[I2_MAX_STATES + 2];
    __shared__ float p[I2_MAX_STATES + 1];
    __shared__ float rn_s;
    const int t = threadIdx.x;
    const int lane = t & 31, warp = t >> 5;
    const int tok = blockIdx.y;
    const float* q = queries + (size_t) blockIdx.x * D;
    float* o = out + ((size_t) blockIdx.x * S + tok) * D;
    // This token's rows: hidden [S, D], blocks [n, S, D]
    hidden_ = (const void*) ((const char*) hidden_ + (size_t) tok * D * (H_HALF ? sizeof(half) : sizeof(float)));
    const float* blocks_t = blocks + (size_t) tok * D;
    const size_t bstep = (size_t) S * D;

    float acc[I2_MAX_STATES + 2];
    for (int s = 0; s < n + 2; ++s) acc[s] = 0.0f;
    for (int i = t; i < D; i += I2R_THREADS)
    {
        const float h = H_HALF ? __half2float(((const half*) hidden_)[i]) : ((const float*) hidden_)[i];
        const float qv = q[i];
        acc[0] += h * h;
        acc[1] += qv * h;
        for (int s = 0; s < n; ++s) acc[2 + s] += qv * blocks_t[s * bstep + i];
    }
    for (int s = 0; s < n + 2; ++s)
    {
        float v = acc[s];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) part[s][warp] = v;
    }
    __syncthreads();
    if (warp == 0)
    {
        for (int s = 0; s < n + 2; ++s)
        {
            float v = part[s][lane];
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
            if (lane == 0) tot[s] = v;
        }
        if (lane == 0)
        {
            const float rn = rsqrtf(tot[0] / (float) D + eps);
            const float sc = rsqrtf((float) D);
            // scores: blocks first, hidden last (the order of the stacked states)
            float mx = tot[1] * rn * sc;
            for (int s = 0; s < n; ++s) mx = fmaxf(mx, tot[2 + s] * sc);
            float sum = 0.0f;
            for (int s = 0; s < n; ++s) { p[s] = expf(tot[2 + s] * sc - mx); sum += p[s]; }
            p[n] = expf(tot[1] * rn * sc - mx); sum += p[n];
            for (int s = 0; s <= n; ++s) p[s] /= sum;
            rn_s = rn;
        }
    }
    __syncthreads();
    const float rn = rn_s;
    for (int i = t; i < D; i += I2R_THREADS)
    {
        const float h = H_HALF ? __half2float(((const half*) hidden_)[i]) : ((const float*) hidden_)[i];
        float v = p[n] * h * rn;
        for (int s = 0; s < n; ++s) v += p[s] * blocks_t[s * bstep + i];
        o[i] = v;
    }
}

void i2_attn_res
(
    const c10::optional<at::Tensor>& blocks,
    const at::Tensor& hidden,
    const at::Tensor& queries,
    at::Tensor out,
    float eps
)
{
    const at::cuda::OptionalCUDAGuard device_guard(hidden.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(hidden.dim() == 2, "i2_attn_res: hidden must be [S, D]");
    const int S = (int) hidden.size(0), D = (int) hidden.size(1);
    TORCH_CHECK(queries.dim() == 2 && queries.size(1) == D && queries.scalar_type() == at::kFloat && queries.is_contiguous(), "i2_attn_res: queries");
    const int nq = (int) queries.size(0);
    TORCH_CHECK(out.numel() == (int64_t) nq * S * D && out.scalar_type() == at::kFloat && out.is_contiguous(), "i2_attn_res: out");
    TORCH_CHECK(hidden.is_contiguous(), "i2_attn_res: hidden");
    int n = 0;
    const float* bp = nullptr;
    if (blocks.has_value())
    {
        const at::Tensor& b = blocks.value();
        n = (int) (b.numel() / ((int64_t) S * D));
        TORCH_CHECK(b.numel() == (int64_t) n * S * D && b.scalar_type() == at::kFloat && b.is_contiguous(), "i2_attn_res: blocks");
        TORCH_CHECK(n <= I2_MAX_STATES, "i2_attn_res: too many block states");
        bp = (const float*) b.data_ptr();
    }
    if (hidden.scalar_type() == at::kHalf)
        i2_attn_res_kernel<true><<<dim3(nq, S), I2R_THREADS, 0, stream>>>(
            bp, hidden.data_ptr(), (const float*) queries.data_ptr(), (float*) out.data_ptr(), n, S, D, eps);
    else
    {
        TORCH_CHECK(hidden.scalar_type() == at::kFloat, "i2_attn_res: hidden dtype");
        i2_attn_res_kernel<false><<<dim3(nq, S), I2R_THREADS, 0, stream>>>(
            bp, hidden.data_ptr(), (const float*) queries.data_ptr(), (float*) out.data_ptr(), n, S, D, eps);
    }
    cuda_check(cudaPeekAtLastError());
}

// ---------------------------------------------------------------------------------------------------------------
// RMS norm with an optional gain, one block per row; output half or float

template <bool X_HALF, bool O_HALF>
__launch_bounds__(I2R_THREADS)
__global__ void i2_norm_kernel
(
    const void* __restrict__ x_,
    const float* __restrict__ w,
    void* __restrict__ out_,
    const int D,
    const int nw,
    const int rows_per_w,
    const float eps
)
{
    __shared__ float part[I2R_WARPS];
    __shared__ float rn_s;
    const int t = threadIdx.x;
    const int lane = t & 31, warp = t >> 5;
    const size_t row = blockIdx.x;
    const size_t off = row * D;
    if (w) w += (size_t) (nw > 1 ? row / rows_per_w : 0) * D;
    #define I2N_X(i) (X_HALF ? __half2float(((const half*) x_)[off + (i)]) : ((const float*) x_)[off + (i)])
    float acc = 0.0f;
    for (int i = t; i < D; i += I2R_THREADS) { const float v = I2N_X(i); acc += v * v; }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (warp == 0)
    {
        float v = part[lane];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0) rn_s = rsqrtf(v / (float) D + eps);
    }
    __syncthreads();
    const float rn = rn_s;
    for (int i = t; i < D; i += I2R_THREADS)
    {
        float v = I2N_X(i) * rn;
        if (w) v *= w[i];
        if (O_HALF) ((half*) out_)[off + i] = __float2half_rn(v);
        else ((float*) out_)[off + i] = v;
    }
    #undef I2N_X
}

void i2_norm
(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& w,
    at::Tensor out,
    float eps
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dim() >= 2 && x.is_contiguous(), "i2_norm: x must be [..., D], contiguous");
    const int D = (int) x.size(-1);
    const int rows = (int) (x.numel() / D);
    TORCH_CHECK(out.numel() == x.numel() && out.is_contiguous(), "i2_norm: out");
    const float* wp = nullptr;
    int nw = 1;
    if (w.has_value())
    {
        const at::Tensor& wt = w.value();
        TORCH_CHECK(wt.scalar_type() == at::kFloat && wt.is_contiguous() && wt.size(-1) == D, "i2_norm: w");
        nw = (int) (wt.numel() / D);
        TORCH_CHECK(nw >= 1 && rows % nw == 0, "i2_norm: rows per gain");
        wp = (const float*) wt.data_ptr();
    }
    const bool xh = x.scalar_type() == at::kHalf, oh = out.scalar_type() == at::kHalf;
    TORCH_CHECK(xh || x.scalar_type() == at::kFloat, "i2_norm: x dtype");
    TORCH_CHECK(oh || out.scalar_type() == at::kFloat, "i2_norm: out dtype");
    #define I2N_LAUNCH(XH, OH) i2_norm_kernel<XH, OH><<<rows, I2R_THREADS, 0, stream>>>( \
        x.data_ptr(), wp, out.data_ptr(), D, nw, rows / nw, eps)
    if (xh && oh) I2N_LAUNCH(true, true);
    else if (xh) I2N_LAUNCH(true, false);
    else if (oh) I2N_LAUNCH(false, true);
    else I2N_LAUNCH(false, false);
    #undef I2N_LAUNCH
    cuda_check(cudaPeekAtLastError());
}

// ---------------------------------------------------------------------------------------------------------------
// Message gate: two launches. The first computes g (one block per row of Wd, each also taking the two RMS values),
// the second the per-channel forget / gain (one thread per channel)

__launch_bounds__(I2_THREADS)
__global__ void i2_message_gate_g_kernel
(
    const float* x,
    const float* m,
    const half* __restrict__ wd,
    float* g,
    const int D,
    const float eps
)
{
    __shared__ float red[3][I2_THREADS];
    const int t = threadIdx.x;
    const half* wr = wd + (size_t) blockIdx.x * 2 * D;
    x += (size_t) blockIdx.y * D;
    m += (size_t) blockIdx.y * D;
    g += (size_t) blockIdx.y * gridDim.x;
    float sx = 0.0f, sm = 0.0f, ax = 0.0f, am = 0.0f;
    for (int i = t; i < D; i += I2_THREADS)
    {
        const float xv = x[i], mv = m[i];
        sx += xv * xv;
        sm += mv * mv;
        ax += __half2float(wr[i]) * xv;
        am += __half2float(wr[D + i]) * mv;
    }
    red[0][t] = sx; red[1][t] = sm;
    __syncthreads();
    for (int stride = I2_THREADS / 2; stride > 0; stride >>= 1)
    {
        if (t < stride) { red[0][t] += red[0][t + stride]; red[1][t] += red[1][t + stride]; }
        __syncthreads();
    }
    const float rx = rsqrtf(red[0][0] / (float) D + eps);
    const float rm = rsqrtf(red[1][0] / (float) D + eps);
    __syncthreads();
    red[2][t] = ax * rx + am * rm;
    __syncthreads();
    for (int stride = I2_THREADS / 2; stride > 0; stride >>= 1)
    {
        if (t < stride) red[2][t] += red[2][t + stride];
        __syncthreads();
    }
    if (t == 0) g[blockIdx.x] = red[2][0];
}

// wf, wg: [r, D]. One block per 32 channels: warp p takes the ranks p, p + 8, ... of those channels (its lanes read
// neighbouring weights), and the first warp adds the eight partial sums. Many small blocks instead of a few threads
// with long chains: the launch is over as soon as the longest chain is
#define I2G_PARTS (I2_THREADS / 32)
__launch_bounds__(I2_THREADS)
__global__ void i2_message_gate_apply_kernel
(
    const float* x,
    const float* m,
    const float* g,
    const float* __restrict__ wf,
    const float* __restrict__ wg,
    float* out,
    const int D,
    const int r
)
{
    __shared__ float gs[I2_MAX_RANK];
    __shared__ float pf[I2G_PARTS][32], pg[I2G_PARTS][32];
    x += (size_t) blockIdx.y * D;
    m += (size_t) blockIdx.y * D;
    out += (size_t) blockIdx.y * D;
    g += (size_t) blockIdx.y * r;
    if (threadIdx.x < r) gs[threadIdx.x] = g[threadIdx.x];
    __syncthreads();
    const int lane = threadIdx.x & 31, part = threadIdx.x >> 5;
    const int d = blockIdx.x * 32 + lane;
    float f = 0.0f, gn = 0.0f;
    if (d < D)
        for (int j = part; j < r; j += I2G_PARTS)
        {
            const float gv = gs[j];
            f += wf[(size_t) j * D + d] * gv;
            gn += wg[(size_t) j * D + d] * gv;
        }
    pf[part][lane] = f;
    pg[part][lane] = gn;
    __syncthreads();
    if (part != 0 || d >= D) return;
    float fs = 0.0f, gsum = 0.0f;
    for (int p = 0; p < I2G_PARTS; ++p) { fs += pf[p][lane]; gsum += pg[p][lane]; }
    out[d] = (1.0f - tanhf(fs)) * x[d] + tanhf(gsum) * m[d];
}

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
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.dim() == 2, "i2_message_gate: x must be [S, D]");
    const int S = (int) x.size(0), D = (int) x.size(1);
    TORCH_CHECK(wd.dim() == 2 && wd.size(1) == 2 * D && wd.scalar_type() == at::kHalf && wd.is_contiguous(), "i2_message_gate: wd");
    const int r = (int) wd.size(0);
    TORCH_CHECK(r <= I2_MAX_RANK, "i2_message_gate: rank");
    TORCH_CHECK(wf.dim() == 2 && wf.size(0) == r && wf.size(1) == D && wf.scalar_type() == at::kFloat && wf.is_contiguous(), "i2_message_gate: wf must be [r, D]");
    TORCH_CHECK(wg.dim() == 2 && wg.size(0) == r && wg.size(1) == D && wg.scalar_type() == at::kFloat && wg.is_contiguous(), "i2_message_gate: wg must be [r, D]");
    TORCH_CHECK(x.scalar_type() == at::kFloat && m.scalar_type() == at::kFloat && m.numel() == (int64_t) S * D, "i2_message_gate: x, m");
    TORCH_CHECK(out.numel() == (int64_t) S * D && out.scalar_type() == at::kFloat, "i2_message_gate: out");
    TORCH_CHECK(x.is_contiguous() && m.is_contiguous() && out.is_contiguous(), "i2_message_gate: contiguity");
    TORCH_CHECK(g.numel() >= (int64_t) S * r && g.scalar_type() == at::kFloat && g.is_contiguous(), "i2_message_gate: g");
    i2_message_gate_g_kernel<<<dim3(r, S), I2_THREADS, 0, stream>>>(
        (const float*) x.data_ptr(), (const float*) m.data_ptr(), (const half*) wd.data_ptr(),
        (float*) g.data_ptr(), D, eps);
    i2_message_gate_apply_kernel<<<dim3(CEIL_DIVIDE(D, 32), S), I2_THREADS, 0, stream>>>(
        (const float*) x.data_ptr(), (const float*) m.data_ptr(), (const float*) g.data_ptr(),
        (const float*) wf.data_ptr(), (const float*) wg.data_ptr(), (float*) out.data_ptr(), D, r);
    cuda_check(cudaPeekAtLastError());
}

// ---------------------------------------------------------------------------------------------------------------
// Attention for one query

#define I2A_THREADS 256
#define I2A_KB 32               // keys per block
#define I2A_PARTS 8             // threads per key in the score stage (I2A_THREADS / I2A_KB)
#define I2A_MAX_G 4
#define I2A_MAX_DH 128
#define I2A_MDIMS 16            // value dimensions per merge block

// Eight halves from one 16-byte load
__device__ __forceinline__ void i2_load8(const half* p, float* f)
{
    const uint4 w = *((const uint4*) p);
    f[0] = __half2float(__ushort_as_half((unsigned short) (w.x & 0xffff)));
    f[1] = __half2float(__ushort_as_half((unsigned short) (w.x >> 16)));
    f[2] = __half2float(__ushort_as_half((unsigned short) (w.y & 0xffff)));
    f[3] = __half2float(__ushort_as_half((unsigned short) (w.y >> 16)));
    f[4] = __half2float(__ushort_as_half((unsigned short) (w.z & 0xffff)));
    f[5] = __half2float(__ushort_as_half((unsigned short) (w.z >> 16)));
    f[6] = __half2float(__ushort_as_half((unsigned short) (w.w & 0xffff)));
    f[7] = __half2float(__ushort_as_half((unsigned short) (w.w >> 16)));
}

#define I2A_SPAN (I2A_MAX_DH / I2A_PARTS)       // dimensions per thread: 16

// A block takes 32 keys. Thread t takes key t % 32 (its lane) and one eighth of the head dimensions (its warp), for
// the scores and again for the weighted values, and reads its 16 halves of K or V with two 16-byte loads: the kernel
// is bound by memory traffic, and a load of one half per instruction reaches a third of the bandwidth. Sums over
// the keys of a block are warp-shuffle reductions.
__launch_bounds__(I2A_THREADS)
__global__ void i2_attn_part_kernel
(
    const float* q,
    const float* rel,
    const float* __restrict__ proj,
    const half* __restrict__ k_store,
    const half* __restrict__ v_store,
    const int64_t* __restrict__ pages,
    const int64_t* __restrict__ ctl,
    float* __restrict__ part,
    const int G,
    const int Hk,
    const int dh,
    const int dr,
    const int extent,
    const int blocks,
    const bool local,
    const int64_t base,
    const int64_t window,
    const float scale
)
{
    __shared__ float qs[I2A_MAX_G][I2A_MAX_DH];
    __shared__ float pd[I2A_MAX_G][I2A_PARTS][I2A_KB];
    __shared__ float es[I2A_MAX_G][I2A_KB];

    const int hk = blockIdx.x, b = blockIdx.y, tok = blockIdx.z, t = threadIdx.x;
    const int H = Hk * G;
    const int64_t pos = ctl[0] + tok, cpos = ctl[1] + tok;      // this token's position and sliding-buffer row
    const int64_t n = local ? min(pos + 1, window) : pos + 1;
    float* po = part + (((size_t) tok * Hk + hk) * blocks + b) * G * (2 + dh);
    q += (size_t) tok * H * dh;
    rel += (size_t) tok * H * dr;
    if ((int64_t) b * I2A_KB >= n)
    {
        if (t < G) { po[(size_t) t * (2 + dh)] = -INFINITY; po[(size_t) t * (2 + dh) + 1] = 0.0f; }
        return;
    }
    for (int i = t; i < G * dh; i += I2A_THREADS) qs[i / dh][i % dh] = q[(size_t) hk * G * dh + i];
    __syncthreads();

    const int key = t & (I2A_KB - 1), pt = t / I2A_KB;
    const int64_t j = (int64_t) b * I2A_KB + key;
    const bool valid = j < n;
    const int64_t d = n - 1 - j;
    const int64_t p = pos - d;
    const int64_t row = !valid ? 0 : (local ? cpos - d : (pages ? pages[p >> 8] * 256 + (p & 255) : base + p));
    const size_t off = ((size_t) row * Hk + hk) * dh + pt * I2A_SPAN;

    // Scores
    float f[I2A_SPAN];
    float s[I2A_MAX_G];
    for (int g = 0; g < G; ++g) s[g] = 0.0f;
    if (valid)
    {
        i2_load8(k_store + off, f);
        i2_load8(k_store + off + 8, f + 8);
        for (int g = 0; g < G; ++g)
        {
            float a0 = 0.0f, a1 = 0.0f;
            for (int i = 0; i < I2A_SPAN; i += 2)
            {
                a0 += qs[g][pt * I2A_SPAN + i] * f[i];
                a1 += qs[g][pt * I2A_SPAN + i + 1] * f[i + 1];
            }
            s[g] = a0 + a1;
        }
    }
    for (int g = 0; g < G; ++g) pd[g][pt][key] = s[g];
    __syncthreads();

    // Per head: the scores of the 32 keys with bias, their maximum, exponentials and sum (the first warp)
    if (pt == 0)
    {
        for (int g = 0; g < G; ++g)
        {
            float v = -INFINITY;
            if (valid)
            {
                v = 0.0f;
                for (int e = 0; e < I2A_PARTS; ++e) v += pd[g][e][key];
                v *= scale;
                if (d < extent)
                {
                    const float* rp = rel + (size_t) (hk * G + g) * dr;
                    float bias = 0.0f;
                    for (int r = 0; r < dr; ++r) bias += rp[r] * proj[(size_t) r * extent + d];
                    v += bias;
                }
            }
            float m = v;
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
            const float e = valid ? expf(v - m) : 0.0f;
            float den = e;
            for (int o = 16; o > 0; o >>= 1) den += __shfl_xor_sync(0xffffffff, den, o);
            es[g][key] = e;
            if (key == 0)
            {
                po[(size_t) g * (2 + dh)] = m;
                po[(size_t) g * (2 + dh) + 1] = den;
            }
        }
    }
    __syncthreads();

    // Weighted value sums over the keys: per thread 16 dimensions of its key, summed over the lanes
    if (valid)
    {
        i2_load8(v_store + off, f);
        i2_load8(v_store + off + 8, f + 8);
    }
    else
        for (int i = 0; i < I2A_SPAN; ++i) f[i] = 0.0f;
    for (int g = 0; g < G; ++g)
    {
        const float w = es[g][key];
        for (int i = 0; i < I2A_SPAN; ++i)
        {
            float v = w * f[i];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            if (key == 0) po[(size_t) g * (2 + dh) + 2 + pt * I2A_SPAN + i] = v;
        }
    }
}

// Merge of a head's blocks with the sink: one block per (head, 16 value dimensions)
__launch_bounds__(I2A_THREADS)
__global__ void i2_attn_merge_kernel
(
    const float* __restrict__ part,
    const void* __restrict__ sinks,
    const int64_t* __restrict__ ctl,
    void* out_,
    const half* __restrict__ ogate,
    const int G,
    const int Hk,
    const int dh,
    const int blocks,
    const bool local,
    const int64_t window,
    const bool sinks_half
)
{
    __shared__ float red[I2A_THREADS];
    __shared__ float stat[2];
    const int h = blockIdx.x, tok = blockIdx.z, t = threadIdx.x;
    const int hk = h / G, g = h % G;
    const int64_t pos = ctl[0] + tok;
    const int64_t n = local ? min(pos + 1, window) : pos + 1;
    const int nb = (int) ((n + I2A_KB - 1) / I2A_KB);
    const float* pb = part + ((((size_t) tok * Hk + hk) * blocks) * G + g) * (2 + dh);
    const size_t step = (size_t) G * (2 + dh);
    const size_t si = (size_t) tok * Hk * G + h;
    const float sink = sinks_half ? __half2float(((const half*) sinks)[si]) : ((const float*) sinks)[si];
    const size_t obase = (size_t) tok * Hk * G * dh;

    float mx = -INFINITY;
    for (int b = t; b < nb; b += I2A_THREADS) mx = fmaxf(mx, pb[b * step]);
    red[t] = mx;
    __syncthreads();
    for (int stride = I2A_THREADS / 2; stride > 0; stride >>= 1)
    {
        if (t < stride) red[t] = fmaxf(red[t], red[t + stride]);
        __syncthreads();
    }
    if (t == 0) stat[0] = fmaxf(red[0], sink);
    __syncthreads();
    mx = stat[0];

    float den = 0.0f;
    for (int b = t; b < nb; b += I2A_THREADS) den += expf(pb[b * step] - mx) * pb[b * step + 1];
    red[t] = den;
    __syncthreads();
    for (int stride = I2A_THREADS / 2; stride > 0; stride >>= 1)
    {
        if (t < stride) red[t] += red[t + stride];
        __syncthreads();
    }
    if (t == 0) stat[1] = red[0] + expf(sink - mx);
    __syncthreads();
    den = stat[1];

    const int chains = I2A_THREADS / I2A_MDIMS;
    const int dl = t % I2A_MDIMS, chain = t / I2A_MDIMS;
    const int dim = blockIdx.y * I2A_MDIMS + dl;
    float v = 0.0f;
    if (dim < dh)
        for (int b = chain; b < nb; b += chains) v += expf(pb[b * step] - mx) * pb[b * step + 2 + dim];
    red[t] = v;
    __syncthreads();
    if (t < I2A_MDIMS && dim < dh)
    {
        float sum = 0.0f;
        for (int c = 0; c < chains; ++c) sum += red[c * I2A_MDIMS + t];
        const size_t oi = obase + (size_t) h * dh + dim;
        if (ogate)
        {
            // The gated output, fp16: the attention output is rounded to fp16 first, as the unfused chain does
            const float o = __half2float(__float2half_rn(sum / den));
            const float gt = 1.0f / (1.0f + expf(-__half2float(ogate[oi])));
            ((half*) out_)[oi] = __float2half_rn(o * gt);
        }
        else
            ((float*) out_)[oi] = sum / den;
    }
}

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
    const c10::optional<at::Tensor>& ogate
)
{
    const at::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(q.dim() == 3 && q.scalar_type() == at::kFloat && q.is_contiguous(), "i2_attn_one: q must be [S, H, dh]");
    const int S = (int) q.size(0), H = (int) q.size(1), dh = (int) q.size(2);
    TORCH_CHECK(k_store.scalar_type() == at::kHalf && v_store.scalar_type() == at::kHalf, "i2_attn_one: K/V dtype");
    TORCH_CHECK(k_store.is_contiguous() && v_store.is_contiguous() && k_store.dim() >= 3, "i2_attn_one: K/V layout");
    const int Hk = (int) k_store.size(-2);
    TORCH_CHECK(k_store.size(-1) == dh && H % Hk == 0, "i2_attn_one: heads");
    const int G = H / Hk;
    TORCH_CHECK(G <= I2A_MAX_G && dh == I2A_MAX_DH && I2A_THREADS == I2A_KB * I2A_PARTS && I2A_KB == 32, "i2_attn_one: geometry");
    TORCH_CHECK(rel.dim() == 3 && rel.size(0) == S && rel.size(1) == H && rel.scalar_type() == at::kFloat && rel.is_contiguous(), "i2_attn_one: rel");
    const int dr = (int) rel.size(2);
    TORCH_CHECK(proj.dim() == 2 && proj.size(0) == dr && proj.scalar_type() == at::kFloat && proj.is_contiguous(), "i2_attn_one: proj");
    const int extent = (int) proj.size(1);
    TORCH_CHECK(sinks.numel() == (int64_t) S * H && (sinks.scalar_type() == at::kFloat || sinks.scalar_type() == at::kHalf) && sinks.is_contiguous(), "i2_attn_one: sinks");
    const half* gp = nullptr;
    if (ogate.has_value())
    {
        const at::Tensor& og = ogate.value();
        TORCH_CHECK(og.numel() == (int64_t) S * H * dh && og.scalar_type() == at::kHalf && og.is_contiguous(), "i2_attn_one: ogate");
        gp = (const half*) og.data_ptr();
    }
    TORCH_CHECK(out.numel() == (int64_t) S * H * dh && out.scalar_type() == (gp ? at::kHalf : at::kFloat) && out.is_contiguous(), "i2_attn_one: out");
    TORCH_CHECK(ctl.scalar_type() == at::kLong && ctl.numel() >= 2, "i2_attn_one: ctl");
    const int blocks = (int) CEIL_DIVIDE(local ? window : capacity, (int64_t) I2A_KB);
    TORCH_CHECK(part.scalar_type() == at::kFloat && part.numel() >= (int64_t) S * Hk * blocks * G * (2 + dh), "i2_attn_one: part");
    const int64_t* pp = nullptr;
    if (pages.has_value())
    {
        TORCH_CHECK(pages.value().scalar_type() == at::kLong, "i2_attn_one: pages");
        pp = (const int64_t*) pages.value().data_ptr();
    }
    dim3 grid(Hk, blocks, S);
    i2_attn_part_kernel<<<grid, I2A_THREADS, 0, stream>>>(
        (const float*) q.data_ptr(), (const float*) rel.data_ptr(), (const float*) proj.data_ptr(),
        (const half*) k_store.data_ptr(), (const half*) v_store.data_ptr(), pp, (const int64_t*) ctl.data_ptr(),
        (float*) part.data_ptr(), G, Hk, dh, dr, extent, blocks, local, base, window, scale);
    dim3 mgrid(H, CEIL_DIVIDE(dh, I2A_MDIMS), S);
    i2_attn_merge_kernel<<<mgrid, I2A_THREADS, 0, stream>>>(
        (const float*) part.data_ptr(), sinks.data_ptr(), (const int64_t*) ctl.data_ptr(),
        out.data_ptr(), gp, G, Hk, dh, blocks, local, window, sinks.scalar_type() == at::kHalf);
    cuda_check(cudaPeekAtLastError());
}


// ---------------------------------------------------------------------------------------------------------------
// Q / K / V post-processing: one block per query head; the first Hk blocks also take a K / V head

#define I2Q_THREADS 128

__launch_bounds__(I2Q_THREADS)
__global__ void i2_qkv_post_kernel
(
    const half* q_lin,
    const half* r_lin,
    const float* k,
    const float* v,
    const float* __restrict__ q_scale,
    const float* __restrict__ rel_scale,
    half* __restrict__ k_store,
    half* __restrict__ v_store,
    const int64_t* __restrict__ pages,
    const int64_t* __restrict__ ctl,
    float* q_out,
    float* rel_out,
    const int Hk,
    const int dh,
    const int dr,
    const bool local,
    const int64_t base,
    const float qk_clip,
    const float rel_clip,
    const float log_alpha,
    const float log_n_floor,
    const float eps
)
{
    __shared__ float red[4][I2Q_THREADS];
    __shared__ float f[4];
    const int h = blockIdx.x, tok = blockIdx.y, t = threadIdx.x;
    const int H = gridDim.x;
    const bool kv = h < Hk;
    q_lin += (size_t) tok * H * dh; r_lin += (size_t) tok * H * dr;
    k += (size_t) tok * Hk * dh; v += (size_t) tok * Hk * dh;
    q_out += (size_t) tok * H * dh; rel_out += (size_t) tok * H * dr;
    const float qv = t < dh ? __half2float(q_lin[h * dh + t]) * q_scale[h * dh + t] : 0.0f;
    const float rv = t < dr ? __half2float(r_lin[h * dr + t]) * rel_scale[h * dr + t] : 0.0f;
    const float kk = kv && t < dh ? k[h * dh + t] : 0.0f;
    const float vv = kv && t < dh ? v[h * dh + t] : 0.0f;
    red[0][t] = qv * qv; red[1][t] = rv * rv; red[2][t] = kk * kk; red[3][t] = vv * vv;
    __syncthreads();
    for (int stride = I2Q_THREADS / 2; stride > 0; stride >>= 1)
    {
        if (t < stride)
            for (int s = 0; s < 4; ++s) red[s][t] += red[s][t + stride];
        __syncthreads();
    }
    if (t == 0)
    {
        const float pos = (float) (ctl[0] + tok);
        const float tau = log_n_floor > 0.0f ? 1.0f + log_alpha * logf(fmaxf((pos + 1.0f) / log_n_floor, 1.0f)) : 1.0f;
        f[0] = fminf(qk_clip / (sqrtf(red[0][0]) + 1e-6f), 1.0f) * tau;
        f[1] = fminf(rel_clip / (sqrtf(red[1][0]) + 1e-6f), 1.0f) * tau;
        f[2] = fminf(qk_clip / (sqrtf(red[2][0]) + 1e-6f), 1.0f);
        f[3] = rsqrtf(red[3][0] / (float) dh + eps);
    }
    __syncthreads();
    if (t < dh) q_out[h * dh + t] = qv * f[0];
    if (t < dr) rel_out[h * dr + t] = rv * f[1];
    if (kv && t < dh)
    {
        const int64_t pos = ctl[0] + tok;
        const int64_t row = local ? ctl[1] + tok : (pages ? pages[pos >> 8] * 256 + (pos & 255) : base + pos);
        k_store[((size_t) row * Hk + h) * dh + t] = __float2half_rn(kk * f[2]);
        v_store[((size_t) row * Hk + h) * dh + t] = __float2half_rn(vv * f[3]);
    }
}

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
)
{
    const at::cuda::OptionalCUDAGuard device_guard(q_lin.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(q_scale.dim() == 2 && q_scale.scalar_type() == at::kFloat && q_scale.is_contiguous(), "i2_qkv_post: q_scale");
    const int H = (int) q_scale.size(0), dh = (int) q_scale.size(1);
    TORCH_CHECK(rel_scale.dim() == 2 && rel_scale.size(0) == H && rel_scale.scalar_type() == at::kFloat && rel_scale.is_contiguous(), "i2_qkv_post: rel_scale");
    const int dr = (int) rel_scale.size(1);
    TORCH_CHECK(dh <= I2Q_THREADS && dr <= I2Q_THREADS, "i2_qkv_post: head size");
    TORCH_CHECK(k_store.scalar_type() == at::kHalf && v_store.scalar_type() == at::kHalf && k_store.dim() >= 3, "i2_qkv_post: stores");
    const int Hk = (int) k_store.size(-2);
    TORCH_CHECK(k_store.size(-1) == dh && Hk <= H && k_store.is_contiguous() && v_store.is_contiguous(), "i2_qkv_post: store layout");
    TORCH_CHECK(q_lin.dim() == 2, "i2_qkv_post: q_lin must be [S, H * dh]");
    const int S = (int) q_lin.size(0);
    TORCH_CHECK(q_lin.scalar_type() == at::kHalf && q_lin.numel() == (int64_t) S * H * dh && q_lin.is_contiguous(), "i2_qkv_post: q_lin");
    TORCH_CHECK(r_lin.scalar_type() == at::kHalf && r_lin.numel() == (int64_t) S * H * dr && r_lin.is_contiguous(), "i2_qkv_post: r_lin");
    TORCH_CHECK(k.scalar_type() == at::kFloat && k.numel() == (int64_t) S * Hk * dh && k.is_contiguous(), "i2_qkv_post: k");
    TORCH_CHECK(v.scalar_type() == at::kFloat && v.numel() == (int64_t) S * Hk * dh && v.is_contiguous(), "i2_qkv_post: v");
    TORCH_CHECK(q_out.scalar_type() == at::kFloat && q_out.numel() == (int64_t) S * H * dh && q_out.is_contiguous(), "i2_qkv_post: q_out");
    TORCH_CHECK(rel_out.scalar_type() == at::kFloat && rel_out.numel() == (int64_t) S * H * dr && rel_out.is_contiguous(), "i2_qkv_post: rel_out");
    TORCH_CHECK(ctl.scalar_type() == at::kLong && ctl.numel() >= 2, "i2_qkv_post: ctl");
    const int64_t* pp = nullptr;
    if (pages.has_value())
    {
        TORCH_CHECK(pages.value().scalar_type() == at::kLong, "i2_qkv_post: pages");
        pp = (const int64_t*) pages.value().data_ptr();
    }
    i2_qkv_post_kernel<<<dim3(H, S), I2Q_THREADS, 0, stream>>>(
        (const half*) q_lin.data_ptr(), (const half*) r_lin.data_ptr(), (const float*) k.data_ptr(), (const float*) v.data_ptr(),
        (const float*) q_scale.data_ptr(), (const float*) rel_scale.data_ptr(), (half*) k_store.data_ptr(),
        (half*) v_store.data_ptr(), pp, (const int64_t*) ctl.data_ptr(), (float*) q_out.data_ptr(), (float*) rel_out.data_ptr(),
        Hk, dh, dr, local, base, qk_clip, rel_clip, log_alpha, log_n_floor, eps);
    cuda_check(cudaPeekAtLastError());
}


// ---------------------------------------------------------------------------------------------------------------
// Prefill attention: fp16-input products with fp32 accumulation, and the row softmax between them

#include "cublas_handle.cuh"

// K/V stores are [M, Hk, dh]: head hk of it is a column-major [dh x M] matrix with leading dimension Hk * dh, and
// the heads are a strided batch with stride dh
void i2_attn_qk(const at::Tensor& q, const at::Tensor& k, at::Tensor sc)
{
    const at::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && sc.dim() == 3, "i2_attn_qk: dims");
    const int Hk = (int) q.size(0), R = (int) q.size(1), dh = (int) q.size(2), M = (int) k.size(0);
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && sc.scalar_type() == at::kFloat, "i2_attn_qk: dtypes");
    TORCH_CHECK(k.size(1) == Hk && k.size(2) == dh && sc.size(0) == Hk && sc.size(1) == R && sc.size(2) == M, "i2_attn_qk: shapes");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && sc.is_contiguous(), "i2_attn_qk: contiguity");
    if (!R || !M) return;
    cublasHandle_t handle = exl3_cublas_handle(stream);
    const float alpha = 1.0f, beta = 0.0f;
    // Column-major: C' [M x R] = op(K') [M x dh] * Q' [dh x R], K' stored [dh x M]
    for (int h = 0; h < Hk; ++h)
    {
        auto r = cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, M, R, dh,
            &alpha, (const half*) k.data_ptr() + (size_t) h * dh, CUDA_R_16F, Hk * dh,
                    (const half*) q.data_ptr() + (size_t) h * R * dh, CUDA_R_16F, dh,
            &beta,  (float*) sc.data_ptr() + (size_t) h * R * M, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cublas_check(r);
    }
    cuda_check(cudaPeekAtLastError());
}

void i2_attn_pv(const at::Tensor& p, const at::Tensor& v, at::Tensor o)
{
    const at::cuda::OptionalCUDAGuard device_guard(p.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(p.dim() == 3 && v.dim() == 3 && o.dim() == 3, "i2_attn_pv: dims");
    const int Hk = (int) p.size(0), R = (int) p.size(1), M = (int) p.size(2), dh = (int) v.size(2);
    TORCH_CHECK(p.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf && o.scalar_type() == at::kFloat, "i2_attn_pv: dtypes");
    TORCH_CHECK(v.size(0) == M && v.size(1) == Hk && o.size(0) == Hk && o.size(1) == R && o.size(2) == dh, "i2_attn_pv: shapes");
    TORCH_CHECK(p.is_contiguous() && v.is_contiguous() && o.is_contiguous(), "i2_attn_pv: contiguity");
    if (!R || !M) return;
    cublasHandle_t handle = exl3_cublas_handle(stream);
    const float alpha = 1.0f, beta = 0.0f;
    // Column-major: O' [dh x R] = V' [dh x M] * P' [M x R]
    for (int h = 0; h < Hk; ++h)
    {
        auto r = cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, dh, R, M,
            &alpha, (const half*) v.data_ptr() + (size_t) h * dh, CUDA_R_16F, Hk * dh,
                    (const half*) p.data_ptr() + (size_t) h * R * M, CUDA_R_16F, M,
            &beta,  (float*) o.data_ptr() + (size_t) h * R * dh, CUDA_R_32F, dh,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cublas_check(r);
    }
    cuda_check(cudaPeekAtLastError());
}

#define I2S_THREADS 256

__launch_bounds__(I2S_THREADS)
__global__ void i2_attn_soft_kernel
(
    const float* __restrict__ sc,
    const float* __restrict__ bias,
    const float* __restrict__ sinks,
    half* __restrict__ p,
    float* __restrict__ den,
    const int H,
    const int G,
    const int nq,
    const int kl,
    const int extent,
    const int64_t a0,
    const int64_t j0,
    const int window,
    const float scale
)
{
    __shared__ float red[I2S_THREADS];
    __shared__ float mx;
    const int i = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    const float* row = sc + ((size_t) h * nq + i) * kl;
    const float* b = bias + ((size_t) i * H + h) * extent;
    const int hk = h / G, g = h % G;
    half* hi = p + ((size_t) ((hk * 2 + 0) * G + g) * nq + i) * kl;
    half* lo = p + ((size_t) ((hk * 2 + 1) * G + g) * nq + i) * kl;
    // Visible keys: distance d = qd - j in [0, window)
    const int64_t qd = a0 + i - j0;
    const int jhi = (int) (qd < kl - 1 ? qd : kl - 1);
    const int jlo = window > 0 && qd - window + 1 > 0 ? (int) (qd - window + 1) : 0;
    const float sink = sinks[(size_t) i * H + h];

    float m = -INFINITY;
    for (int j = jlo + t; j <= jhi; j += I2S_THREADS)
    {
        const int64_t d = qd - j;
        const float val = row[j] * scale + (d < extent ? b[d] : 0.0f);
        m = fmaxf(m, val);
    }
    red[t] = m;
    __syncthreads();
    for (int s = I2S_THREADS / 2; s > 0; s >>= 1)
    {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    if (t == 0) mx = fmaxf(red[0], sink);
    __syncthreads();
    const float top = mx;

    float sum = 0.0f;
    for (int j = t; j < kl; j += I2S_THREADS)
    {
        if (j < jlo || j > jhi)
        {
            hi[j] = __float2half_rn(0.0f); lo[j] = __float2half_rn(0.0f);
            continue;
        }
        const int64_t d = qd - j;
        const float val = row[j] * scale + (d < extent ? b[d] : 0.0f);
        const float e = expf(val - top);
        sum += e;
        const float e14 = e * 16384.0f;
        const half eh = __float2half_rn(e14);
        hi[j] = eh;
        lo[j] = __float2half_rn((e14 - __half2float(eh)) * 2048.0f);
    }
    red[t] = sum;
    __syncthreads();
    for (int s = I2S_THREADS / 2; s > 0; s >>= 1)
    {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    if (t == 0) den[(size_t) h * nq + i] = red[0] + expf(sink - top);
}

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
)
{
    const at::cuda::OptionalCUDAGuard device_guard(sc.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(sc.dim() == 3 && sc.scalar_type() == at::kFloat && sc.is_contiguous(), "i2_attn_soft: sc must be float [H, nq, kl]");
    const int H = (int) sc.size(0), nq = (int) sc.size(1), kl = (int) sc.size(2);
    TORCH_CHECK(bias.dim() == 3 && bias.size(0) == nq && bias.size(1) == H && bias.scalar_type() == at::kFloat && bias.is_contiguous(), "i2_attn_soft: bias");
    const int extent = (int) bias.size(2);
    TORCH_CHECK(sinks.numel() == (int64_t) nq * H && sinks.scalar_type() == at::kFloat && sinks.is_contiguous(), "i2_attn_soft: sinks");
    TORCH_CHECK(p.dim() == 5 && p.scalar_type() == at::kHalf && p.is_contiguous() && p.size(1) == 2, "i2_attn_soft: p must be half [Hk, 2, G, nq, kl]");
    const int Hk = (int) p.size(0), G = (int) p.size(2);
    TORCH_CHECK(Hk * G == H && p.size(3) == nq && p.size(4) == kl, "i2_attn_soft: p shape");
    TORCH_CHECK(den.numel() == (int64_t) H * nq && den.scalar_type() == at::kFloat && den.is_contiguous(), "i2_attn_soft: den");
    TORCH_CHECK(a0 >= j0, "i2_attn_soft: positions");
    if (!nq || !kl) return;
    i2_attn_soft_kernel<<<dim3(nq, H), I2S_THREADS, 0, stream>>>(
        (const float*) sc.data_ptr(), (const float*) bias.data_ptr(), (const float*) sinks.data_ptr(),
        (half*) p.data_ptr(), (float*) den.data_ptr(), H, G, nq, kl, extent, a0, j0, (int) window, scale);
    cuda_check(cudaPeekAtLastError());
}
