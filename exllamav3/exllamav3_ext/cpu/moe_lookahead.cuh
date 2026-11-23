#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <cstdint>
#include <vector>

// Local patch: expert lookahead for the CPU expert split (see moe_lookahead.cu).

// Columns of the per-layer pointer tables
#define MOE_LA_CPU_COLS 10      // arena block (device alias), then (suh, svh, bias) per projection
#define MOE_LA_GPU_COLS 12      // trellis per projection, then (suh, svh, bias) per projection
#define MOE_LA_GEOM_COLS 9      // per projection: off_b, tiles_k, tiles_n, tile_b, group, planar, suh_b, svh_b, bias_b
#define MOE_LA_PLAN_COLS 3      // router id, worker slot, GPU slot
#define MOE_LA_LOG_COLS 4       // layer, router id, worker slot, GPU slot

// One layer's lookahead step, all enqueued without a host wait. On the current (compute) stream: pick up to
// cap_pred predicted experts (pred: [rows, k] router ids, rank-major) and up to cap_req host-requested ones
// that are still CPU-resident and not in flight, give each a free GPU slot and mark its worker slot in flight.
// On `side_stream`, behind an event at that point: copy each planned expert from the pinned arena into its GPU
// slot (restoring native trellis order), then point the placement map at the slot and append
// (layer, router id, worker slot, GPU slot) to the pinned completion log.
// Returns false without enqueueing anything when this layer's previous step is still in flight on the side
// stream (its plan buffer is still in use).
bool moe_la_step
(
    int64_t layer,
    const c10::optional<at::Tensor>& pred,   // long [rows, k] on the layer's device, or None
    at::Tensor map,                          // long [experts]: router id -> physical slot
    int64_t first,                           // physical ids >= first are worker slots
    at::Tensor inflight,                     // uint8 [worker slots]
    at::Tensor pool,                         // int32 [capacity]: free GPU slots (stack)
    at::Tensor pool_n,                       // int32 [1]
    at::Tensor plan,                         // int32 [cap, 3]
    at::Tensor plan_n,                       // int32 [1]
    int64_t req_ptr,                         // device alias of pinned int32 [seq, count, ids...]
    at::Tensor req_seen,                     // int32 [1]: last request seq consumed
    int64_t cap_pred,
    int64_t cap_req,
    const at::Tensor& tab_cpu,               // int64 [worker slots, MOE_LA_CPU_COLS]
    const at::Tensor& tab_gpu,               // int64 [GPU slots, MOE_LA_GPU_COLS]
    const std::vector<int64_t>& geom,        // [projections * MOE_LA_GEOM_COLS]
    int64_t log_ptr,                         // device alias of pinned int32 [log_cap, MOE_LA_LOG_COLS]
    int64_t log_cap,
    at::Tensor log_n,                        // int32 [1] on the device: entries written so far
    int64_t done_ptr,                        // device alias of a pinned uint32: entries visible to the host
    int64_t side_stream,                     // cudaStream_t of the upload stream
    int64_t upload_blocks                    // blocks per expert in the upload kernel (0: one per 8-tile run)
);

// Host bookkeeping applied in compute-stream order: ops is a device alias of pinned int32 [n, 4] rows of
// (kind, router id, worker slot, GPU slot). kind 0: the GPU slot is free (push to the pool). kind 1: a
// demotion landed: the router id maps to the worker slot again, the worker slot leaves the in-flight set and
// the GPU slot is free.
void moe_la_host_ops
(
    int64_t ops_ptr,
    int64_t n,
    at::Tensor map,
    int64_t first,
    at::Tensor inflight,
    at::Tensor pool,
    at::Tensor pool_n
);
