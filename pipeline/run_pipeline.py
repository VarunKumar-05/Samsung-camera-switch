#!/usr/bin/env python
"""Process a burst folder pulled from the phone (Download/CamLab/<session>/).

  python run_pipeline.py DIR [--preset natural|neutral|punchy] [--ml kpn.pt] [--set sharpen.amount=0.3 ...] [--out out]
Writes <out>/<name>_<preset>.png (16-bit) and <out>/<name>_single.png (the reference frame alone, for A/B).
"""
import argparse
import json
import os

import cv2
import numpy as np

from camlab.io import load_burst_dir, load_jpeg_burst_dir
from camlab.params import preset
from camlab.pipeline import process_burst, process_jpeg_burst, rotate_upright


def save16(path, rgb):
    cv2.imwrite(path, cv2.cvtColor((np.clip(rgb, 0, 1) * 65535 + 0.5).astype(np.uint16), cv2.COLOR_RGB2BGR))


def apply_overrides(P, sets):
    for s in sets or []:
        key, val = s.split("=")
        sec, name = key.split(".")
        cur = getattr(getattr(P, sec), name)
        setattr(getattr(P, sec), name, type(cur)(json.loads(val)) if not isinstance(cur, bool) else val.lower() in "1true")
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--preset", default="natural")
    ap.add_argument("--ml", help="trained KPN checkpoint (camlab.ml.train)")
    ap.add_argument("--set", nargs="*", help="override knobs, e.g. sharpen.amount=0.3 denoise.luma=1.5")
    ap.add_argument("--frames", type=int, help="use only the first N frames")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = apply_overrides(preset(a.preset), a.set)
    name = os.path.basename(os.path.normpath(a.dir))
    meta = json.load(open(os.path.join(a.dir, "meta.json")))
    if meta.get("mode") == "JPEG":
        out = process_jpeg_burst(load_jpeg_burst_dir(a.dir, a.frames), P)
        save16(os.path.join(a.out, f"{name}_{a.preset}.png"), out)
        return
    burst = load_burst_dir(a.dir, a.frames)
    model = None
    if a.ml:
        from camlab.ml.infer import load
        model = load(a.ml)
    stages = {}
    out = process_burst(burst, P, stages=stages, ml_model=model)
    save16(os.path.join(a.out, f"{name}_{a.preset}{'_ml' if model else ''}.png"), rotate_upright(out, burst.orientation))
    # single-frame baseline through the *same* tone/sharpen settings: isolates what burst merging buys you
    one = type(burst)(**{**burst.__dict__, "frames": burst.frames[stages["ref_idx"]:stages["ref_idx"] + 1]})
    save16(os.path.join(a.out, f"{name}_single.png"), rotate_upright(process_burst(one, P), burst.orientation))
    with open(os.path.join(a.out, f"{name}_{a.preset}.params.json"), "w") as f:
        f.write(P.to_json())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
