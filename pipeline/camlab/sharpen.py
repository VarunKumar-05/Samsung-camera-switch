import cv2
import numpy as np


def sharpen_luma(rgb: np.ndarray, amount: float, radius: float, threshold: float, halo: float) -> np.ndarray:
    """Noise-gated, halo-limited unsharp mask on luma of gamma-encoded RGB in [0,1]."""
    if amount <= 0:
        return rgb
    ycc = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    y = ycc[..., 0]
    high = y - cv2.GaussianBlur(y, (0, 0), radius)
    # smooth gate: ~0 for |high| << threshold, ~1 for |high| >> threshold
    a = np.abs(high)
    gate = a * a / (a * a + threshold * threshold)
    y2 = y + amount * high * gate
    k = np.ones((3, 3), np.uint8)
    lo, hi = cv2.erode(y, k) - halo, cv2.dilate(y, k) + halo
    ycc[..., 0] = np.clip(y2, lo, hi)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
