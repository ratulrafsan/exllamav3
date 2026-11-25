"""
Convert a llama.cpp control-vector GGUF (tensors "direction.<layer>", one f32 vector per layer)
to safetensors for the control-vector patch (EXL3_CVEC, exllamav3/modules/transformer.py).

    python -I util/cvec_gguf_to_safetensors.py in.gguf out.safetensors

Layer numbering is kept as llama.cpp's: "direction.k" applies after decoder layer k (0-based),
and llama.cpp never stores a direction for layer 0. Only reads; nothing in the file is executed.

Also reads GLP files (projective control vectors, github.com/msuiche/weightless docs/spec/GLP.md): the tensors
are kept under their names and the file's `glp.*`, `controlvector.*` and `general.name` metadata go into the
safetensors metadata, as strings. The Inkling-2 control vector reader (EXL3_I2_CVEC,
exllamav3/architecture/inkling2_loop.py) takes mode, default strength and numbering from them: in a file with
`glp.structure = per-execution-step`, "direction.N" is for layer application N - 1.
"""
import struct, sys
import numpy as np

GGUF_MAGIC = b"GGUF"
# GGUF metadata value types
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
GGML_F32, GGML_F16 = 0, 1


class Reader:
    def __init__(self, buf):
        self.b, self.o = buf, 0

    def take(self, fmt):
        v = struct.unpack_from(fmt, self.b, self.o)[0]
        self.o += struct.calcsize(fmt)
        return v

    def string(self):
        n = self.take("<Q")
        s = self.b[self.o:self.o + n].decode("utf-8", "replace")
        self.o += n
        return s

    def value(self, t):
        if t in _SCALARS:
            return self.take(_SCALARS[t])
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.take("<I"), self.take("<Q")
            return [self.value(et) for _ in range(n)]
        raise ValueError(f"unknown GGUF value type {t}")


def read_cvec(path):
    buf = open(path, "rb").read()
    r = Reader(buf)
    if r.b[:4] != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    r.o = 4
    version = r.take("<I")
    if version < 2:
        raise ValueError(f"GGUF v{version} not supported")
    n_tensors, n_kv = r.take("<Q"), r.take("<Q")
    meta = {}
    for _ in range(n_kv):
        k = r.string()
        meta[k] = r.value(r.take("<I"))
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.take("<Q") for _ in range(r.take("<I"))]
        ttype, off = r.take("<I"), r.take("<Q")
        infos.append((name, dims, ttype, off))
    align = int(meta.get("general.alignment", 32))
    data0 = (r.o + align - 1) // align * align
    out = {}
    for name, dims, ttype, off in infos:
        n = int(np.prod(dims))
        if ttype == GGML_F32:
            a = np.frombuffer(buf, np.float32, n, data0 + off)
        elif ttype == GGML_F16:
            a = np.frombuffer(buf, np.float16, n, data0 + off).astype(np.float32)
        else:
            raise ValueError(f"{name}: tensor type {ttype} not supported (f32/f16 only)")
        out[name] = a.copy()
    return meta, out


def main():
    src, dst = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    meta, tensors = read_cvec(src)
    print("metadata:", {k: v for k, v in meta.items() if not isinstance(v, list)})
    layers = sorted(int(k.split(".")[1]) for k in tensors if k.startswith("direction."))
    norms = {l: float(np.linalg.norm(tensors[f"direction.{l}"])) for l in layers}
    dim = {tensors[f"direction.{l}"].size for l in layers}
    print(f"{len(layers)} directions, layers {layers[0]}..{layers[-1]}, dim {dim}, "
          f"norm min {min(norms.values()):.4f} max {max(norms.values()):.4f}")
    other = [k for k in tensors if not k.startswith("direction.")]
    if other:
        print("other tensors (ignored):", other)
    if dst:
        from safetensors.numpy import save_file
        keep = {k: (str(v).lower() if isinstance(v, bool) else str(v)) for k, v in meta.items()
                if not isinstance(v, list) and (k.startswith(("glp.", "controlvector.")) or k == "general.name")}
        per_step = meta.get("glp.structure") == "per-execution-step"
        save_file({k: v for k, v in tensors.items() if k.startswith("direction.")}, dst,
                  metadata = {"source": src.rsplit("/", 1)[-1],
                              "convention": "direction.N = layer application N - 1" if per_step
                              else "llama.cpp direction.k = after layer k", **keep})
        print("wrote", dst)


if __name__ == "__main__":
    main()
