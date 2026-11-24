"""
Component timings for the Inkling-2 EXL3 model.
  decode   one layer's expert MLP (router + experts) replayed 48 times in a CUDA graph; the head
  prefill  sections of a prefill chunk (device sync after each), at two context lengths

    python util/inkling2/component_bench.py <EXL3 model dir>
"""
import os, sys, time
from collections import defaultdict
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

@torch.inference_mode()
def main():
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.modules import inkling2 as I2
    from exllamav3.architecture import inkling2_loop as A
    config = Config.from_directory(sys.argv[1])
    model = Model.from_config(config)
    size = 49152
    cache = Cache(model, max_num_tokens = size, max_batch_size = 1)
    model.load(reserve_per_device = [3.0, 1.0])
    tok = Tokenizer.from_config(config)
    with open(os.path.join(HERE, "..", "..", "eval", "eval_texts", "variable_man_mod.txt"), encoding = "utf8") as f:
        ids = tok.encode(f.read())
    while ids.shape[1] < size:
        ids = torch.cat([ids, ids], 1)

    # --- decode: expert MLP alone, in a graph
    for li in (3, 21):
        L = model.modules[1 + li]
        dev = L.device
        x = torch.randn(1, config.hidden_size, device = dev).half()
        L.ex.params = {}
        for _ in range(3):
            L.ex.moe(x, torch.half)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.device(dev):
            with torch.cuda.graph(g, stream = torch.cuda.Stream(device = dev)):
                for _ in range(48):
                    y = L.ex.moe(x, torch.half)
            g.replay(); torch.cuda.synchronize(dev); t0 = time.perf_counter()
            for _ in range(50):
                g.replay()
            torch.cuda.synchronize(dev)
        print(f"decode: expert MLP of layer {li} ({dev}): {(time.perf_counter() - t0) / 50 / 48 * 1e6:.0f} us per call "
              f"-> {(time.perf_counter() - t0) / 50 * 1e3:.2f} ms for 48", flush = True)

    # --- prefill sections
    names = ("layer_attention", "attention_core", "layer_mlp_input", "layer_finish", "attn_res", "short_conv",
             "dyn_short_conv", "message_gate", "rms_norm", "l2_clip")
    times = defaultdict(float)
    def timed(n, fn):
        def w(*a, **k):
            torch.cuda.synchronize(); t0 = time.perf_counter(); r = fn(*a, **k); torch.cuda.synchronize()
            times[n] += time.perf_counter() - t0; return r
        return w
    orig = {n: getattr(I2, n) for n in names}
    moe_t = [0.0]
    orig_moe = A.Exl3Experts._moe
    def moe(self, x, adt):
        torch.cuda.synchronize(); t0 = time.perf_counter(); r = orig_moe(self, x, adt); torch.cuda.synchronize()
        moe_t[0] += time.perf_counter() - t0; return r

    def step(t, past, st):
        p = {"attn_mode": "flash_attn", "cache": cache, "past_len": past, "batch_shape": (1, size), "last_tokens_only": 1}
        if st is not None: p["recurrent_states"] = st
        model.forward(t, p); return p["recurrent_states"]

    st, pos = None, 0
    for target in (0, 40960):
        while pos < target:
            st = step(ids[:, pos : pos + 4096], pos, st); pos += 4096
        torch.cuda.synchronize(); t0 = time.perf_counter()
        st = step(ids[:, pos : pos + 2048], pos, st); pos += 2048
        torch.cuda.synchronize(); plain = time.perf_counter() - t0
        for n in names: setattr(I2, n, timed(n, orig[n]))
        A.Exl3Experts._moe = moe
        for m in model.modules:
            if hasattr(m, "ex") and hasattr(m.ex, "moe"): m.ex.moe = m.ex._moe
        times.clear(); moe_t[0] = 0.0
        torch.cuda.synchronize(); t0 = time.perf_counter()
        st = step(ids[:, pos : pos + 2048], pos, st); pos += 2048
        torch.cuda.synchronize(); total = time.perf_counter() - t0
        for n in names: setattr(I2, n, orig[n])
        A.Exl3Experts._moe = orig_moe
        for m in model.modules:
            if hasattr(m, "ex") and hasattr(m.ex, "moe"): m.ex.moe = m.ex._moe
        print(f"prefill chunk of 2048 at context {target}: {plain * 1e3:.0f} ms ({2048 / plain:.0f} T/s); with timers {total * 1e3:.0f} ms", flush = True)
        print(f"   experts {moe_t[0] * 1e3:.0f} ms | layer_attention {times['layer_attention'] * 1e3:.0f} (attention_core {times['attention_core'] * 1e3:.0f}) "
              f"| mlp_input {times['layer_mlp_input'] * 1e3:.0f} | finish {times['layer_finish'] * 1e3:.0f} || inside: attn_res {times['attn_res'] * 1e3:.0f}, "
              f"conv {(times['short_conv'] + times['dyn_short_conv']) * 1e3:.0f}, gate {times['message_gate'] * 1e3:.0f}, rms {times['rms_norm'] * 1e3:.0f}, l2 {times['l2_clip'] * 1e3:.0f}")

if __name__ == "__main__":
    main()
