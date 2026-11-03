import cv2
import numpy as np

from .align import estimate_flow, warp


def pick_reference(guides: np.ndarray) -> int:
    """Sharpest frame (variance of Laplacian) is the reference; it fixes the output geometry."""
    return int(np.argmax([cv2.Laplacian(g, cv2.CV_32F).var() for g in guides]))


def align_burst(planes: np.ndarray, ref_idx: int, tile: int = 16, levels: int = 4):
    """planes: (N,C,h,w). Returns a list of flows (None for the reference)."""
    guides = planes.mean(1)
    return [None if i == ref_idx else estimate_flow(guides[ref_idx], guides[i], tile=tile, levels=levels)
            for i in range(len(planes))]


def merge_burst(planes: np.ndarray, flows, ref_idx: int, S: np.ndarray, O: np.ndarray,
                strictness: float = 1.0, window: int = 5, sat: float = 0.98):
    """Robust temporal merge in the normalised linear raw domain.

    planes: (N,C,h,w); S,O: (C,) noise params (var = S*x+O). Returns (merged (C,h,w), var_out (C,h,w)).
    Per-pixel weights fall to 0 where the aligned frame disagrees with the reference by more than noise
    explains (moving objects, misalignment), so ghosts degrade gracefully to the single reference frame.
    """
    ref = planes[ref_idx]
    S3, O3 = S[:, None, None].astype(np.float32), O[:, None, None].astype(np.float32)
    var = S3 * np.maximum(ref, 0) + O3
    num, den, w2 = ref.copy(), np.ones_like(ref), np.ones_like(ref)
    for i, fl in enumerate(flows):
        if i == ref_idx:
            continue
        w_img = warp(planes[i], fl)
        d = np.stack([cv2.blur(c, (window, window)) for c in (ref - w_img) ** 2])
        w = np.exp(-np.maximum(d - 2 * var, 0) / (strictness * var + 1e-12)).astype(np.float32)
        w *= (w_img < sat)
        num += w * w_img
        den += w
        w2 += w * w
    return num / den, var * (w2 / den ** 2)
