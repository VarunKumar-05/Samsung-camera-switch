#!/usr/bin/env python
"""Compare the stock Samsung JPEG with our output of the same scene.

  python compare.py stock.jpg ours.png [--crops 0.3,0.4 0.7,0.6] [--size 300]
Prints no-reference metrics and writes <out>/compare.png: full frames side by side + 100% crops.
Metrics (higher sharpness = more detail OR more oversharpening; look at the crops, not just numbers):
  sharp  : variance of Laplacian of luma, on both images resized to the same width
  noise  : std of high-pass residual in the flattest 5% of the image (lower = cleaner)
  detail : high-frequency energy in textured areas / noise  (higher = more real detail per unit noise)
"""
import argparse
import os

import cv2
import numpy as np


def load(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = (im / 257).astype(np.uint8)
    return im


def metrics(bgr):
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    hp = y - cv2.GaussianBlur(y, (0, 0), 1.5)
    local = cv2.GaussianBlur(np.abs(hp), (0, 0), 8)
    flat = local < np.percentile(local, 5)
    busy = local > np.percentile(local, 80)
    noise = float(hp[flat].std())
    return dict(sharp=float(cv2.Laplacian(y, cv2.CV_32F).var() * 1e3), noise=noise * 1e3,
                detail=float(np.abs(hp[busy]).mean() / (noise + 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stock")
    ap.add_argument("ours")
    ap.add_argument("--crops", nargs="*", default=["0.3,0.4", "0.7,0.6"])
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--width", type=int, default=2000, help="common analysis width")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    imgs = []
    for p in (a.stock, a.ours):
        im = load(p)
        h, w = im.shape[:2]
        if h > w and True:  # keep orientation as-is; just resize
            pass
        s = a.width / w
        imgs.append(cv2.resize(im, (a.width, int(h * s)), interpolation=cv2.INTER_AREA))
    names = ["stock", "ours"]
    for n, im in zip(names, imgs):
        m = metrics(im)
        print(f"{n:6s} sharp={m['sharp']:8.2f}  noise={m['noise']:6.3f}  detail/noise={m['detail']:6.2f}")
    rows = []
    for im, n in zip(imgs, names):
        cv2.putText(im, n, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    h = min(i.shape[0] for i in imgs)
    full = np.hstack([cv2.resize(i[:h], (a.width // 2, int(h * 0.5))) for i in imgs])
    crops = []
    for c in a.crops:
        fx, fy = map(float, c.split(","))
        row = []
        for im in imgs:
            H, W = im.shape[:2]
            cx, cy, s = int(fx * W), int(fy * H), a.size
            row.append(cv2.resize(im[max(cy - s // 2, 0):cy + s // 2, max(cx - s // 2, 0):cx + s // 2], (s, s), interpolation=cv2.INTER_NEAREST))
        crops.append(np.hstack(row))
    crop_img = np.vstack(crops)
    cw = full.shape[1]
    pad = np.zeros((crop_img.shape[0], max(cw - crop_img.shape[1], 0), 3), np.uint8)
    sheet = np.vstack([full, np.hstack([crop_img, pad])[:, :max(cw, crop_img.shape[1])]]) if crop_img.shape[1] <= cw else full
    os.makedirs(a.out, exist_ok=True)
    cv2.imwrite(os.path.join(a.out, "compare.png"), sheet)
    print("wrote", os.path.join(a.out, "compare.png"))


if __name__ == "__main__":
    main()
