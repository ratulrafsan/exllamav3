"""
The control vector in this port against the transformers reference with the documented hooks
(util/inkling2/cvec_ref_dump.py): the same tokens, the same vector, the same scales.

Two implementations of this model never agree exactly: top-k expert routing amplifies rounding differences
(scale 0 below shows that floor). What has to agree is the effect of the vector: the change of the logits
from scale 0 to scale s, in direction (cosine) and size (ratio of norms), and the steered predictions.

    python util/inkling2/cvec_ref_compare.py <model dir> <vector.safetensors>
        <dir of cvec_ref_dump.py>
"""
import argparse, glob, os
import numpy as np
import torch


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("vector"); ap.add_argument("ref")
    ap.add_argument("--device")
    args = ap.parse_args()
    os.environ["EXL3_I2_CVEC"] = args.vector
    from exllamav3 import Config, Model
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    model.load(device = args.device) if args.device else model.load(reserve_per_device = [3.0, 1.0])
    cv = config.cvec
    lsm = lambda z: torch.log_softmax(z.float(), -1)
    for f in sorted(glob.glob(os.path.join(args.ref, "*.npz"))):
        d = np.load(f)
        ids = torch.from_numpy(d["ids"]).unsqueeze(0)
        keys = sorted(k for k in d.files if k.startswith("s") and not k.endswith("_argmax"))
        tail = d[keys[0]].shape[0]
        port = {}
        for s in sorted({float(k[1:].split("_")[0]) for k in keys}):
            cv.set_scale(s)
            port[s] = model.forward(ids)[0, -tail:].float().cpu()
        print(f"{os.path.basename(f)[:-4]} ({ids.shape[1]} tokens, last {tail} positions)")
        ref0 = torch.from_numpy(d["s0_f32"]).float()
        for k in keys:
            s = float(k[1:].split("_")[0])
            ref = torch.from_numpy(d[k]).float()
            p = port[s]
            kl = (lsm(ref).exp() * (lsm(ref) - lsm(p))).sum(-1).mean().item()
            agree = (ref.argmax(-1) == p.argmax(-1)).float().mean().item()
            line = f"  scale {s:<4g} hook in {k.split('_')[1]:<4}: KL(reference || port) {kl:.4f}, same top token {100 * agree:5.1f}%"
            if s > 0:
                dr, dp = (ref - ref0).flatten(), (p - port[0.0]).flatten()
                cos = torch.nn.functional.cosine_similarity(dr, dp, dim = 0).item()
                moved_r = (ref.argmax(-1) != ref0.argmax(-1)).float().mean().item()
                moved_p = (p.argmax(-1) != port[0.0].argmax(-1)).float().mean().item()
                line += (f";  change of the logits from scale 0: cosine {cos:.3f}, size port / reference "
                         f"{(dp.norm() / dr.norm()).item():.3f};  top token changed at {100 * moved_r:.0f}% (reference) "
                         f"/ {100 * moved_p:.0f}% (port) of the positions")
            print(line, flush = True)


if __name__ == "__main__":
    main()
