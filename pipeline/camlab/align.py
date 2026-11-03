"""Coarse-to-fine tile alignment (HDR+-style), fully vectorised with numpy/OpenCV.

Convention: flow[y,x] = (dx,dy) such that  alt(x+dx, y+dy) ~= ref(x, y).
"""
from typing import Sequence

import cv2
import numpy as np


def _remap(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp (h,w) or (C,h,w) image by a full-resolution flow field."""
    if img.ndim == 2:
        return _remap(img, flow)
    return np.stack([_remap(c, flow) for c in img])


def _cost_volume(ref, warped, r, tile):
    H, W = ref.shape
    th, tw = H // tile, W // tile
    pad = np.pad(warped, r, mode="edge")
    n = 2 * r + 1
    costs = np.empty((n, n, th, tw), np.float32)
    for iy, dy in enumerate(range(-r, r + 1)):
        for ix, dx in enumerate(range(-r, r + 1)):
            d = ref - pad[r + dy:r + dy + H, r + dx:r + dx + W]
            d = (d * d)[:th * tile, :tw * tile]
            costs[iy, ix] = d.reshape(th, tile, tw, tile).sum(axis=(1, 3))
    return costs


def _subpixel(cm, c, cp, interior):
    den = cm - 2 * c + cp
    ok = interior & (den > 1e-12)
    return np.where(ok, np.clip(0.5 * (cm - cp) / np.where(ok, den, 1), -0.5, 0.5), 0)


def estimate_flow(ref: np.ndarray, alt: np.ndarray, tile: int = 16, levels: int = 4,
                  radius: Sequence[int] = (4, 3, 2, 2)) -> np.ndarray:
    """Per-pixel flow (H,W,2) from per-tile estimates, refined over a Gaussian pyramid, with sub-pixel accuracy.

    radius[k] is the +-search radius at the k-th pyramid level counted from the coarsest.
    """
    pr, pa = [ref.astype(np.float32)], [alt.astype(np.float32)]
    for _ in range(levels - 1):
        pr.append(cv2.pyrDown(pr[-1]))
        pa.append(cv2.pyrDown(pa[-1]))
    flow = None
    for k, lvl in enumerate(reversed(range(levels))):
        R, A = pr[lvl], pa[lvl]
        H, W = R.shape
        th, tw = H // tile, W // tile
        if th < 1 or tw < 1:
            continue
        if flow is None:
            flow = np.zeros((th, tw, 2), np.float32)
        else:
            flow = cv2.resize(flow, (tw, th), interpolation=cv2.INTER_LINEAR) * 2.0
        warped = _remap(A, cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR))
        r = radius[min(k, len(radius) - 1)]
        costs = _cost_volume(R, warped, r, tile)
        n = 2 * r + 1
        best = costs.reshape(n * n, th, tw).argmin(0)
        iy, ix = best // n, best % n
        yy, xx = np.mgrid[0:th, 0:tw]
        c0 = costs[iy, ix, yy, xx]
        cl, cr = costs[iy, np.clip(ix - 1, 0, n - 1), yy, xx], costs[iy, np.clip(ix + 1, 0, n - 1), yy, xx]
        cu, cd = costs[np.clip(iy - 1, 0, n - 1), ix, yy, xx], costs[np.clip(iy + 1, 0, n - 1), ix, yy, xx]
        ox = _subpixel(cl, c0, cr, (ix > 0) & (ix < n - 1))
        oy = _subpixel(cu, c0, cd, (iy > 0) & (iy < n - 1))
        flow = flow + np.stack([ix - r + ox, iy - r + oy], -1).astype(np.float32)
    h, w = ref.shape
    return cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
