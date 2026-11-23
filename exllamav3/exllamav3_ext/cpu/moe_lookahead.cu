#include "moe_lookahead.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include "../util.h"
#include "../util.cuh"

// Local patch: expert lookahead for the CPU expert split.
//
// The next split layer's router, applied to this layer's router input, names most of the experts the next
// layer is about to pick (measured on Qwen3.8-Flash-Next: 59% of its CPU-resident picks, 67% of all). The
// experts it names that are still CPU-resident are copied into free GPU slots while this layer computes, so
// the next layer finds them in VRAM. They stay resident; the host's promote step demotes cold residents to
// refill the free slots.
//
// Everything runs on the GPU in stream order, with no host thread in the loop:
//   compute stream   select: reads the placement map, the in-flight set and the free-slot pool (all device
//                    memory, only ever written in compute-stream order, except the map flips below)
//   side stream      upload: a kernel reads the experts straight from the pinned arena (mapped host memory,
//                    PCIe bandwidth), restores native trellis order into the GPU slot, copies the aux tensors
//                    commit: flips the map entries and logs the transfers to pinned host memory
// A flip can land at any point relative to the compute stream. Both placements are valid across it: the GPU
// slot was free (nothing mapped to it) until the flip, and the worker slot keeps the expert until the host
// has seen the log entry and reuses the slot for a demotion, which it orders behind the compute stream.

#define LA_THREADS 128

struct LaGeom
{
    int nproj;
    int off_b[3];
    int tiles_k[3];
    int tiles_n[3];
    int tile_b[3];
    int group[3];
    int planar[3];
    int aux_b[3][3];
    int blk0[4];        // first block of each projection; blk0[nproj] = first aux block
};

__global__ void moe_la_select_kernel
(
    const int64_t* __restrict__ pred,
    const int rows,
    const int k,
    const int64_t* __restrict__ map,
    const int first,
    uint8_t* __restrict__ inflight,
    int32_t* __restrict__ pool,
    int32_t* __restrict__ pool_n,
    int32_t* __restrict__ plan,
    int32_t* __restrict__ plan_n,
    const int32_t* req,
    int32_t* __restrict__ req_seen,
    const int cap_pred,
    const int cap_req,
    const int num_experts
)
{
    int n = 0;
    int pn = *pool_n;

    // Predictions, best rank first across the rows
    for (int j = 0; j < k && n < cap_pred && pn > 0; ++j)
    {
        for (int row = 0; row < rows && n < cap_pred && pn > 0; ++row)
        {
            const int64_t r = pred[row * k + j];
            if (r < 0 || r >= num_experts) continue;
            const int64_t p = map[r];
            if (p < first) continue;
            const int c = (int) (p - first);
            if (inflight[c]) continue;
            const int g = pool[--pn];
            inflight[c] = 1;
            plan[3 * n] = (int32_t) r; plan[3 * n + 1] = c; plan[3 * n + 2] = g;
            n++;
        }
    }

    // Host requests (misses of earlier steps): [seq, count, ids...], seq written last by the host
    const volatile int32_t* vr = req;
    const int32_t seq = vr[0];
    if (seq != *req_seen)
    {
        const int cnt = vr[1];
        int taken = 0;
        for (int j = 0; j < cnt && taken < cap_req && pn > 0; ++j)
        {
            const int32_t r = vr[2 + j];
            if (r < 0 || r >= num_experts) continue;
            const int64_t p = map[r];
            if (p < first) continue;
            const int c = (int) (p - first);
            if (inflight[c]) continue;
            const int g = pool[--pn];
            inflight[c] = 1;
            plan[3 * n] = r; plan[3 * n + 1] = c; plan[3 * n + 2] = g;
            n++; taken++;
        }
        *req_seen = seq;
    }

    *pool_n = pn;
    *plan_n = n;
}

// Same restore as moe_unswizzle_kernel (moe_unswizzle.cu, runtime-layout path) for one expert, with the
// source and destination taken from the pointer tables by the planned (worker slot, GPU slot)
__global__ __launch_bounds__(LA_THREADS)
void moe_la_upload_kernel
(
    const int32_t* __restrict__ plan,
    const int32_t* __restrict__ plan_n,
    const int64_t* __restrict__ tab_cpu,
    const int64_t* __restrict__ tab_gpu,
    const LaGeom geo
)
{
    const int e = blockIdx.y;
    if (e >= *plan_n) return;
    const int c = plan[3 * e + 1];
    const int g = plan[3 * e + 2];
    const int64_t* tc = tab_cpu + (size_t) c * MOE_LA_CPU_COLS;
    const int64_t* tg = tab_gpu + (size_t) g * MOE_LA_GPU_COLS;

    // Each block walks the runs blockIdx.x, blockIdx.x + gridDim.x, ...: the grid is one block per 8-tile
    // run by default (as in moe_unswizzle_kernel), or upload_blocks blocks per expert. Fewer blocks only
    // made the upload slower in testing; they did not shorten the compute stream's stall
    for (int b = blockIdx.x; b < geo.blk0[geo.nproj] + geo.nproj; b += gridDim.x)
    {

    if (b >= geo.blk0[geo.nproj])
    {
        // Aux tensors of one projection (fp16 suh, svh, bias)
        const int p = b - geo.blk0[geo.nproj];
        for (int a = 0; a < 3; ++a)
        {
            const uint16_t* s = reinterpret_cast<const uint16_t*>(tc[1 + 3 * p + a]);
            uint16_t* d = reinterpret_cast<uint16_t*>(tg[3 + 3 * p + a]);
            if (!s || !d) continue;
            const int cnt = geo.aux_b[p][a] / 2;
            for (int i = threadIdx.x; i < cnt; i += LA_THREADS)
                d[i] = s[i];
        }
        continue;
    }

    int p = 0;
    while (p + 1 < geo.nproj && b >= geo.blk0[p + 1]) p++;
    const int lb = b - geo.blk0[p];
    const int tiles_k = geo.tiles_k[p];
    const int tiles_n = geo.tiles_n[p];
    const int tile_b = geo.tile_b[p];
    const int group = geo.group[p];
    const int planar = geo.planar[p];
    const uint8_t* src = reinterpret_cast<const uint8_t*>(tc[0]) + geo.off_b[p];
    uint8_t* dst = reinterpret_cast<uint8_t*>(tg[p]);

    const int runs = tiles_n / 8;
    const int kt = lb / runs;
    const int r = lb % runs;                    // native 8-tile run index
    const size_t run_off = ((size_t) kt * tiles_n + (size_t) r * 8) * tile_b;
    const size_t run_b = (size_t) 8 * tile_b;
    uint4* d = reinterpret_cast<uint4*>(dst + run_off);
    if (!group)
    {
        const uint4* s = reinterpret_cast<const uint4*>(src + run_off);
        for (int i = threadIdx.x; i < (int) (run_b / 16); i += LA_THREADS)
            d[i] = s[i];
        continue;
    }
    const size_t sub_b = (size_t) group * tile_b;
    for (int s = 0; s < 8 / group; ++s)
    {
        const size_t sub_src = ((size_t) (r * (8 / group) + s) * tiles_k + kt) * sub_b;
        uint4* sd = d + (size_t) s * (sub_b / 16);
        if (!planar)
        {
            const uint4* ss = reinterpret_cast<const uint4*>(src + sub_src);
            for (int i = threadIdx.x; i < (int) (sub_b / 16); i += LA_THREADS)
                sd[i] = ss[i];
            continue;
        }
        if (tile_b % 32)
        {
            // Fractional planar inverse (see moe_unswizzle_kernel)
            const int q = tile_b / 16;
            const int wd = tile_b / 4;
            const uint32_t* ss = reinterpret_cast<const uint32_t*>(src + sub_src);
            uint32_t* dd = reinterpret_cast<uint32_t*>(sd);
            for (int i = threadIdx.x; i < (int) (sub_b / 4); i += LA_THREADS)
            {
                const int w = i % wd;
                const int tile0 = i - w;
                const int hl = 2 * w + 1, hh = 2 * w;
                const int pl = 8 * (hl % q) + hl / q;
                const int ph = 8 * (hh % q) + hh / q;
                const uint32_t a = ss[tile0 + (pl >> 1)];
                const uint32_t bb = ss[tile0 + (ph >> 1)];
                const int sel = ((pl & 1) ? 0x0032 : 0x0010) | ((ph & 1) ? 0x7600 : 0x5400);
                dd[i] = __byte_perm(a, bb, sel);
            }
            continue;
        }
        // Planar inverse: native dword w of a tile is stored at 8 * (w % bits) + w / bits
        const int bits = tile_b / 32;
        const int wd = tile_b / 4;
        const uint32_t* ss = reinterpret_cast<const uint32_t*>(src + sub_src);
        uint32_t* dd = reinterpret_cast<uint32_t*>(sd);
        for (int i = threadIdx.x; i < (int) (sub_b / 4); i += LA_THREADS)
        {
            const int w = i % wd;
            dd[i] = ss[(i - w) + 8 * (w % bits) + w / bits];
        }
    }

    }   // runs of this block
}

__global__ void moe_la_commit_kernel
(
    const int32_t* __restrict__ plan,
    const int32_t* __restrict__ plan_n,
    int64_t* map,
    int32_t* log,
    int32_t* __restrict__ log_n,
    uint32_t* done,
    const int layer,
    const int log_cap
)
{
    const int n = *plan_n;
    if (n <= 0) return;
    const int ln = *log_n;
    volatile int32_t* vl = log;
    for (int i = 0; i < n; ++i)
    {
        const int32_t r = plan[3 * i], c = plan[3 * i + 1], g = plan[3 * i + 2];
        map[r] = (int64_t) g;
        const int e = (int) ((unsigned) (ln + i) % (unsigned) log_cap);
        vl[4 * e] = layer; vl[4 * e + 1] = r; vl[4 * e + 2] = c; vl[4 * e + 3] = g;
    }
    *log_n = ln + n;
    __threadfence_system();
    volatile uint32_t* vd = done;
    *vd = (uint32_t) (ln + n);
    __threadfence_system();
}

__global__ void moe_la_host_ops_kernel
(
    const int32_t* ops,
    const int n,
    int64_t* map,
    const int first,
    uint8_t* __restrict__ inflight,
    int32_t* __restrict__ pool,
    int32_t* __restrict__ pool_n,
    const int pool_cap
)
{
    const volatile int32_t* vo = ops;
    int pn = *pool_n;
    for (int i = 0; i < n; ++i)
    {
        const int32_t kind = vo[4 * i], r = vo[4 * i + 1], c = vo[4 * i + 2], g = vo[4 * i + 3];
        if (kind == 1)
        {
            map[r] = (int64_t) first + c;
            inflight[c] = 0;
        }
        if (pn < pool_cap) pool[pn++] = g;
    }
    *pool_n = pn;
}

namespace {

// Per layer, the event behind the layer's last commit: a new step must not reuse the plan buffer before it
std::mutex g_la_mutex;
std::unordered_map<int64_t, cudaEvent_t> g_la_commit_ev;
std::unordered_map<int64_t, cudaEvent_t> g_la_select_ev;

cudaEvent_t la_event(std::unordered_map<int64_t, cudaEvent_t>& m, int64_t key, bool* created)
{
    auto it = m.find(key);
    if (it != m.end()) { *created = false; return it->second; }
    cudaEvent_t ev;
    cuda_check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    m[key] = ev;
    *created = true;
    return ev;
}

} // namespace

bool moe_la_step
(
    int64_t layer,
    const c10::optional<at::Tensor>& pred,
    at::Tensor map,
    int64_t first,
    at::Tensor inflight,
    at::Tensor pool,
    at::Tensor pool_n,
    at::Tensor plan,
    at::Tensor plan_n,
    int64_t req_ptr,
    at::Tensor req_seen,
    int64_t cap_pred,
    int64_t cap_req,
    const at::Tensor& tab_cpu,
    const at::Tensor& tab_gpu,
    const std::vector<int64_t>& geom,
    int64_t log_ptr,
    int64_t log_cap,
    at::Tensor log_n,
    int64_t done_ptr,
    int64_t side_stream,
    int64_t upload_blocks
)
{
    const at::cuda::OptionalCUDAGuard device_guard(map.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaStream_t side = reinterpret_cast<cudaStream_t>(side_stream);
    TORCH_CHECK_DTYPE(map, kLong);
    TORCH_CHECK_DTYPE(inflight, kByte);
    TORCH_CHECK_DTYPE(pool, kInt);
    TORCH_CHECK_DTYPE(pool_n, kInt);
    TORCH_CHECK_DTYPE(plan, kInt);
    TORCH_CHECK_DTYPE(plan_n, kInt);
    TORCH_CHECK_DTYPE(req_seen, kInt);
    TORCH_CHECK_DTYPE(log_n, kInt);
    TORCH_CHECK_DTYPE(tab_cpu, kLong);
    TORCH_CHECK_DTYPE(tab_gpu, kLong);
    TORCH_CHECK_DTYPE_OPT(pred, kLong);
    TORCH_CHECK(tab_cpu.dim() == 2 && tab_cpu.size(1) == MOE_LA_CPU_COLS && tab_cpu.is_contiguous(), "moe_la_step: bad tab_cpu");
    TORCH_CHECK(tab_gpu.dim() == 2 && tab_gpu.size(1) == MOE_LA_GPU_COLS && tab_gpu.is_contiguous(), "moe_la_step: bad tab_gpu");
    const int nproj = (int) (geom.size() / MOE_LA_GEOM_COLS);
    TORCH_CHECK(nproj >= 1 && nproj <= 3 && geom.size() == (size_t) nproj * MOE_LA_GEOM_COLS, "moe_la_step: bad geometry");
    const int cap = (int) (plan.numel() / MOE_LA_PLAN_COLS);
    TORCH_CHECK(cap_pred >= 0 && cap_req >= 0 && cap_pred + cap_req <= cap && cap >= 1, "moe_la_step: caps exceed the plan buffer");
    int rows = 0, k = 0;
    if (pred.has_value() && cap_pred > 0)
    {
        TORCH_CHECK(pred.value().dim() == 2 && pred.value().is_contiguous(), "moe_la_step: pred must be contiguous [rows, k]");
        rows = (int) pred.value().size(0);
        k = (int) pred.value().size(1);
    }

    LaGeom geo;
    memset(&geo, 0, sizeof(geo));
    geo.nproj = nproj;
    int blk = 0;
    for (int p = 0; p < nproj; ++p)
    {
        const int64_t* gp = geom.data() + (size_t) p * MOE_LA_GEOM_COLS;
        geo.off_b[p] = (int) gp[0];
        geo.tiles_k[p] = (int) gp[1];
        geo.tiles_n[p] = (int) gp[2];
        geo.tile_b[p] = (int) gp[3];
        geo.group[p] = (int) gp[4];
        geo.planar[p] = (int) gp[5];
        geo.aux_b[p][0] = (int) gp[6];
        geo.aux_b[p][1] = (int) gp[7];
        geo.aux_b[p][2] = (int) gp[8];
        TORCH_CHECK(geo.tiles_n[p] % 8 == 0 && geo.tiles_k[p] > 0, "moe_la_step: tiles_n must be a multiple of 8");
        TORCH_CHECK(geo.group[p] == 0 || geo.group[p] == 2 || geo.group[p] == 8, "moe_la_step: group must be 0, 2 or 8");
        TORCH_CHECK(geo.off_b[p] % 16 == 0 && geo.tile_b[p] % 16 == 0, "moe_la_step: offsets must be 16-byte aligned");
        TORCH_CHECK(!geo.planar[p] || geo.group[p], "moe_la_step: planar requires a packed projection");
        TORCH_CHECK(!geo.planar[p] || geo.tile_b[p] % 32 == 0 || geo.group[p] == 2, "moe_la_step: fractional planar layout requires group 2");
        geo.blk0[p] = blk;
        blk += geo.tiles_k[p] * (geo.tiles_n[p] / 8);
    }
    geo.blk0[nproj] = blk;
    const int blocks_x = blk + nproj;

    cudaEvent_t ev_select, ev_commit;
    {
        std::lock_guard<std::mutex> lock(g_la_mutex);
        const int64_t key = (int64_t) map.get_device() * 100000 + layer;
        bool created;
        ev_commit = la_event(g_la_commit_ev, key, &created);
        // The previous step of this layer still owns the plan buffer: skip this one
        if (!created && cudaEventQuery(ev_commit) != cudaSuccess)
        {
            cudaGetLastError();
            return false;
        }
        ev_select = la_event(g_la_select_ev, key, &created);
    }

    moe_la_select_kernel<<<1, 1, 0, stream>>>
    (
        rows ? (const int64_t*) pred.value().data_ptr() : nullptr, rows, k,
        (const int64_t*) map.data_ptr(), (int) first,
        (uint8_t*) inflight.data_ptr(),
        (int32_t*) pool.data_ptr(), (int32_t*) pool_n.data_ptr(),
        (int32_t*) plan.data_ptr(), (int32_t*) plan_n.data_ptr(),
        reinterpret_cast<const int32_t*>(req_ptr), (int32_t*) req_seen.data_ptr(),
        (int) cap_pred, (int) cap_req, (int) map.numel()
    );
    cuda_check(cudaPeekAtLastError());
    cuda_check(cudaEventRecord(ev_select, stream));

    cuda_check(cudaStreamWaitEvent(side, ev_select, 0));
    const int bx = upload_blocks > 0 && upload_blocks < blocks_x ? (int) upload_blocks : blocks_x;
    dim3 grid((unsigned) bx, (unsigned) cap);
    moe_la_upload_kernel<<<grid, LA_THREADS, 0, side>>>
    (
        (const int32_t*) plan.data_ptr(), (const int32_t*) plan_n.data_ptr(),
        (const int64_t*) tab_cpu.data_ptr(), (const int64_t*) tab_gpu.data_ptr(), geo
    );
    cuda_check(cudaPeekAtLastError());
    moe_la_commit_kernel<<<1, 1, 0, side>>>
    (
        (const int32_t*) plan.data_ptr(), (const int32_t*) plan_n.data_ptr(),
        (int64_t*) map.data_ptr(),
        reinterpret_cast<int32_t*>(log_ptr), (int32_t*) log_n.data_ptr(),
        reinterpret_cast<uint32_t*>(done_ptr), (int) layer, (int) log_cap
    );
    cuda_check(cudaPeekAtLastError());
    cuda_check(cudaEventRecord(ev_commit, side));
    return true;
}

void moe_la_host_ops
(
    int64_t ops_ptr,
    int64_t n,
    at::Tensor map,
    int64_t first,
    at::Tensor inflight,
    at::Tensor pool,
    at::Tensor pool_n
)
{
    if (n <= 0) return;
    const at::cuda::OptionalCUDAGuard device_guard(map.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(map, kLong);
    TORCH_CHECK_DTYPE(inflight, kByte);
    TORCH_CHECK_DTYPE(pool, kInt);
    TORCH_CHECK_DTYPE(pool_n, kInt);
    moe_la_host_ops_kernel<<<1, 1, 0, stream>>>
    (
        reinterpret_cast<const int32_t*>(ops_ptr), (int) n,
        (int64_t*) map.data_ptr(), (int) first,
        (uint8_t*) inflight.data_ptr(),
        (int32_t*) pool.data_ptr(), (int32_t*) pool_n.data_ptr(), (int) pool.numel()
    );
    cuda_check(cudaPeekAtLastError());
}
