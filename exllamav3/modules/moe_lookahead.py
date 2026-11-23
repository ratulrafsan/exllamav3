from __future__ import annotations
import os
import re
import torch
from ..ext import exllamav3_ext as ext

"""
Local patch: expert lookahead for the CPU expert split (EXL3_MOE_CPU_LOOKAHEAD=N, N predicted experts
uploaded per split layer per decode step; 0 or unset = off).

While split layer L computes, layer L+1's router is applied to L's router input. The experts it names
that are still CPU-resident are copied into free GPU slots of L+1 by GPU kernels on a side stream
(exllamav3_ext/cpu/moe_lookahead.cu), so L+1 mostly finds its picks in VRAM. Uploaded experts stay
resident. With the lookahead on, every promotion goes through those kernels: the host's promote step
(block_sparse_mlp_cpu._split_promote_step) only requests its miss promotions, reads the completion log
and demotes cold residents to refill the free GPU slots.

Needs the pinned arena registered as device-mapped (moe_cpu_host._pin_pending_chunk does that when the
variable is set) and an extension built from this source (moe_la_step); with a stock extension the
variable only prints a notice.

Experimental and off by default. On the system it was written for (3090 on PCIe 4.0 x16, 13700K, DDR4) it
raises the GPU hit rate (76% to 82%) but decode is slower (75 to 60 T/s): the upload kernel reads mapped host
memory, and while it runs, the CPU hand-off kernels of the compute stream, which use mapped host memory too,
stall. See the README section "Expert lookahead".
"""

LOOKAHEAD = int(os.environ.get("EXL3_MOE_CPU_LOOKAHEAD", 0))
LOOKAHEAD_ROWS = int(os.environ.get("EXL3_MOE_CPU_LOOKAHEAD_ROWS", 4))    # forwards with more rows don't predict
# Diagnostics: 1 = run the predictor but upload none of its picks (isolates the predictor's cost)
_DRY = os.environ.get("EXL3_MOE_CPU_LOOKAHEAD_DRY", "0") != "0"
# Blocks per expert in the upload kernel (0: one block per 8-tile run, the fastest upload). Fewer blocks make
# the upload slower. On the test system the compute stream stalls for as long as an upload runs (both use
# mapped host memory), so fewer blocks were slower overall: 60 T/s at 0, 58 at 128, 45 at 16, 19 at 4
_UPLOAD_BLOCKS = int(os.environ.get("EXL3_MOE_CPU_LOOKAHEAD_BLOCKS", 0))
_REQ_MAX = 8            # host requests per layer per step the request buffer holds
_CAP_REQ = 1            # ... of which a select takes this many (the promote step asks for one per layer)
_LOG_CAP = 8192         # completion log entries (a ring; the host drains it every step)
_OPS_RING, _OPS_MAX = 512, 64
_POOL_CAP = 64


def _layer_number(m):
    hit = re.search(r"layers\.(\d+)\.", m.key)
    return int(hit.group(1)) if hit else 1 << 30


class _LayerState:
    pass


class Lookahead:
    """Lookahead state over the promote step's modules (`ps` is the _PromoteState; a module's state
    index is its row there, which is also the layer id in the completion log)"""

    def __init__(self, ps, slot_linears):
        from ..model.model_tp_cuda import cuda_host_get_device_pointer
        self.ps = ps
        self.ok = False
        mods = [m for m in ps.modules if getattr(m, "cpu_component", "text") == "text"]
        if len(mods) != len(ps.modules):
            raise RuntimeError("a non-text component is split")
        devs = {torch.device(m.device) for m in mods}
        if len(devs) != 1:
            raise RuntimeError("split layers on more than one device")
        self.device = dev = devs.pop()
        if not hasattr(ext, "moe_la_step"):
            raise RuntimeError("the extension has no moe_la_step (not built from the patched source)")
        n = len(ps.modules)
        with torch.cuda.device(dev):
            self.side = torch.cuda.Stream(device = dev)
            self.log = torch.zeros((_LOG_CAP, 4), dtype = torch.int32).pin_memory()
            self.log_np = self.log.numpy()
            self.log_dev = ext.pinned_cuda_view(self.log, dev.index or 0)
            self.done = torch.zeros(1, dtype = torch.int32).pin_memory()
            self.done_np = self.done.numpy()
            self.done_dev = ext.pinned_cuda_view(self.done, dev.index or 0)
            self.log_n = torch.zeros(1, dtype = torch.int32, device = dev)
            self.seen = 0
            self.req = torch.zeros((n, 2 + _REQ_MAX), dtype = torch.int32).pin_memory()
            self.req_np = self.req.numpy()
            self.req_dev = ext.pinned_cuda_view(self.req, dev.index or 0)
            self.ops = torch.zeros((_OPS_RING, _OPS_MAX, 4), dtype = torch.int32).pin_memory()
            self.ops_np = self.ops.numpy()
            self.ops_dev = ext.pinned_cuda_view(self.ops, dev.index or 0)
            self.ops_i = 0
        self.pending_ops = [[] for _ in range(n)]
        self.pending_req = [[] for _ in range(n)]
        self.stats = dict(uploads = 0, skipped = 0, steps = 0)

        order = sorted(range(n), key = lambda i: _layer_number(ps.modules[i]))
        self.states = [None] * n
        for i in order:
            self.states[i] = self._layer_state(i, ps.modules[i], slot_linears, cuda_host_get_device_pointer)
        # Forward order: each layer predicts for the next one; the first has no predictor
        for a, b in zip(order, order[1:]):
            ma, mb = ps.modules[a], ps.modules[b]
            plain = all(getattr(ma, at, None) is None for at in ("router_pre_norm", "routed_pre_norm", "latent_in")) \
                and not ma.alt_residual_channel and ma.hidden_size == mb.hidden_size \
                and mb.routing_fn.__name__ == "routing_std" and mb.routing_gate is not None
            self.states[a].target = self.states[b] if plain else None
            self.states[b].has_predictor = plain
        for i in order:
            m, st = ps.modules[i], self.states[i]
            # The module's free GPU slots become the device-side pool; from here on the select kernel owns them
            for g in m._split_free_gpu:
                self.pending_ops[i].append((0, 0, 0, g))
            m._split_free_gpu = []
            m._la = st
        self.flush_ops()
        self.ok = True

    def _layer_state(self, i, m, slot_linears, dev_ptr):
        host, li = m.cpu_host, m.cpu_layer_idx
        spec = host.specs[li]
        dev = self.device
        if m._split_pending:
            raise RuntimeError("transfers already in flight")
        if not host.pinned or host.layer_blocks[li] is None or li not in host.aux:
            raise RuntimeError(f"{m.key}: no pinned arena")
        layout, _ = host._stream_layout(spec)
        pd = spec["proj_dims"]
        first = m.cpu_split_first
        blocks = host.layer_blocks[li]
        st = _LayerState()
        st.index, st.module, st.first = i, m, first
        st.target, st.has_predictor = None, False
        parts = host._promo_parts(li, slot_linears(m, 0))
        geom = []
        for _, name, lin, off, nb in parts:
            tk, tn = pd[name][0] // 16, pd[name][1] // 16
            group, planar = layout[("g", "u", "d").index(name)]
            tile_b = nb // (tk * tn)
            if tile_b * tk * tn != nb:
                raise RuntimeError(f"{m.key}: projection size is not a whole number of tiles")
            aux_b = [(t.numel() * t.element_size() if t is not None else 0)
                     for t in (getattr(lin, a, None) for a in ("suh", "svh", "bias"))]
            geom += [off, tk, tn, tile_b, group, planar] + aux_b
        st.geom = geom
        tab_gpu = torch.zeros((first, 12), dtype = torch.long)
        for g in range(first):
            for p, (_, name, lin, off, nb) in enumerate(host._promo_parts(li, slot_linears(m, g))):
                tr = lin.trellis
                if tr.numel() * tr.element_size() != nb or not tr.is_contiguous() or tr.data_ptr() % 16:
                    raise RuntimeError(f"{m.key}: GPU slot {g} does not have the layer's projection size")
                tab_gpu[g, p] = tr.data_ptr()
                for a, an in enumerate(("suh", "svh", "bias")):
                    t = getattr(lin, an, None)
                    if t is None:
                        continue
                    if t.dtype != torch.half or t.numel() * 2 != geom[9 * p + 6 + a]:
                        raise RuntimeError(f"{m.key}: GPU slot {g} {an} differs from slot 0")
                    tab_gpu[g, 3 + 3 * p + a] = t.data_ptr()
        ncpu = len(blocks)
        tab_cpu = torch.zeros((ncpu, 10), dtype = torch.long)
        aux = host.aux[li]
        chunk_ptr = {}
        for c in range(ncpu):
            ci, off = blocks[c]
            if ci >= len(host.arena_views) or host.arena_views[ci] is None:
                raise RuntimeError(f"{m.key}: an expert's arena chunk is not page-locked")
            if ci not in chunk_ptr:
                chunk_ptr[ci] = dev_ptr(host.arena_views[ci].data_ptr())
            ptr = chunk_ptr[ci] + off
            if ptr % 16:
                raise RuntimeError(f"{m.key}: arena block is not 16-byte aligned")
            tab_cpu[c, 0] = ptr
            for p, (_, name, lin, _, _) in enumerate(parts):
                for a, an in enumerate(("suh", "svh", "bias")):
                    lst = aux.get(f"{an}_{name}")
                    t = lst[c] if lst is not None else None
                    if t is None or not geom[9 * p + 6 + a]:
                        continue
                    if t.dtype != torch.half or t.numel() * 2 != geom[9 * p + 6 + a] or t.device != dev:
                        raise RuntimeError(f"{m.key}: worker expert {c} {an} differs from the GPU slots")
                    tab_cpu[c, 1 + 3 * p + a] = t.data_ptr()
        with torch.cuda.device(dev):
            st.tab_gpu = tab_gpu.to(dev)
            st.tab_cpu = tab_cpu.to(dev)
            st.inflight = torch.zeros(ncpu, dtype = torch.uint8, device = dev)
            st.pool = torch.zeros(_POOL_CAP, dtype = torch.int32, device = dev)
            st.pool_n = torch.zeros(1, dtype = torch.int32, device = dev)
            st.plan = torch.zeros((LOOKAHEAD + _CAP_REQ, 3), dtype = torch.int32, device = dev)
            st.plan_n = torch.zeros(1, dtype = torch.int32, device = dev)
            st.req_seen = torch.zeros(1, dtype = torch.int32, device = dev)
            st.req_ptr = self.req_dev[i].data_ptr()
            cfg = m.routing_cfg
            st.logits = torch.empty((LOOKAHEAD_ROWS, m.num_experts), dtype = cfg.logits_dtype, device = dev)
            st.sel = torch.empty((LOOKAHEAD_ROWS, m.num_experts_per_tok), dtype = torch.long, device = dev)
            st.w = torch.empty((LOOKAHEAD_ROWS, m.num_experts_per_tok), dtype = torch.half, device = dev)
        return st

    def _select(self, st, pred):
        """Select + upload + commit for the layer of `st` (pred: its predicted picks or None)"""
        m = st.module
        issued = ext.moe_la_step(
            st.index, pred, m._split_map, st.first, st.inflight, st.pool, st.pool_n, st.plan, st.plan_n,
            st.req_ptr, st.req_seen, LOOKAHEAD if pred is not None and not _DRY else 0, _CAP_REQ,
            st.tab_cpu, st.tab_gpu, st.geom, self.log_dev.data_ptr(), _LOG_CAP, self.log_n,
            self.done_dev.data_ptr(), self.side.cuda_stream, _UPLOAD_BLOCKS)
        if not issued:
            self.stats["skipped"] += 1

    def step(self, st, y, bsz):
        """Called from a split layer's submit on a decode-size forward, right after its routing: predict
        the next split layer's picks from this layer's router input and start their uploads"""
        if not st.has_predictor:
            # No layer predicts for this one: its host requests still need a select
            self._select(st, None)
        t = st.target
        if t is None:
            return
        if bsz > LOOKAHEAD_ROWS or y.dtype != torch.half and y.dtype != torch.float:
            self._select(t, None)
            return
        from .block_sparse_mlp_routing import _gate_t
        cfg = t.module.routing_cfg
        _gate_t(cfg)
        sel = t.sel[:bsz]
        ext.routing_std(y, cfg.gate_tensor, t.logits[:bsz], sel, t.w[:bsz], cfg.per_expert_scale,
                        cfg.gate_tensor_t, None, cfg.gate_i8, cfg.gate_sb)
        self._select(t, sel)

    def request(self, i, r):
        if len(self.pending_req[i]) < _REQ_MAX:
            self.pending_req[i].append(r)

    def flush_requests(self):
        for i, rs in enumerate(self.pending_req):
            if not rs:
                continue
            row = self.req_np[i]
            row[2:2 + len(rs)] = rs
            row[1] = len(rs)
            row[0] += 1                 # the sequence number goes last: the select kernel reads it first
            self.pending_req[i] = []

    def flush_ops(self):
        """Apply the queued host ops (free slots, landed demotions) in compute-stream order"""
        dev = self.device
        for i, ops in enumerate(self.pending_ops):
            while ops:
                batch, ops = ops[:_OPS_MAX], ops[_OPS_MAX:]
                k = self.ops_i % _OPS_RING
                self.ops_i += 1
                self.ops_np[k, :len(batch)] = batch
                m, st = self.ps.modules[i], self.states[i]
                with torch.cuda.device(dev):
                    ext.moe_la_host_ops(self.ops_dev[k].data_ptr(), len(batch), m._split_map, st.first,
                                        st.inflight, st.pool, st.pool_n)
            self.pending_ops[i] = []

    def drain_log(self):
        """Completed uploads since the last call, as (state index, router id, worker slot, GPU slot)"""
        done = int(self.done_np[0]) & 0xFFFFFFFF
        out = []
        while self.seen != done:
            e = self.log_np[self.seen % _LOG_CAP]
            out.append((int(e[0]), int(e[1]), int(e[2]), int(e[3])))
            self.seen = (self.seen + 1) & 0xFFFFFFFF
        self.stats["uploads"] += len(out)
        return out

    def quiesce(self):
        """Let every upload in flight land (before a prefill-size forward, whose stager reads the arena
        and whose expert kernels may read any GPU slot)"""
        self.side.synchronize()
