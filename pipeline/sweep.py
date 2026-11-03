#!/usr/bin/env python
"""Contact sheet of one knob swept across values, same burst: find the look you like.

  python sweep.py DIR sharpen.amount 0 0.4 0.8 1.2 [--crop 0.5 0.5 400]
"""
import argparse
import os

import cv2
import numpy as np

from camlab.io import load_burst_dir
from camlab.params import preset
from camlab.pipeline import process_burst, rotate_upright
from run_pipeline import apply_overrides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("knob")
    ap.add_argument("values", nargs="+")
    ap.add_argument("--preset", default="natural")
    ap.add_argument("--crop", nargs=3, type=float, default=[0.5, 0.5, 400], help="cx cy size_px (fractions, px)")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    burst = load_burst_dir(a.dir)
    tiles = []
    for v in a.values:
        img = rotate_upright(process_burst(burst, apply_overrides(preset(a.preset), [f"{a.knob}={v}"])), burst.orientation)
        h, w = img.shape[:2]
        s = int(a.crop[2]); cx, cy = int(a.crop[0] * w), int(a.crop[1] * h)
        c = img[max(cy - s // 2, 0):cy + s // 2, max(cx - s // 2, 0):cx + s // 2]
        c = np.ascontiguousarray((c * 255 + .5).astype(np.uint8)[..., ::-1])
        cv2.putText(c, f"{a.knob}={v}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(c)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"sweep_{a.knob}.png")
    cv2.imwrite(path, np.hstack(tiles))
    print("wrote", path)


if __name__ == "__main__":
    main()
