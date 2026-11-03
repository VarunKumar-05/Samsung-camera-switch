import cv2
import numpy as np


def denoise_planes(planes: np.ndarray, var: np.ndarray, strength: float, window: int = 5) -> np.ndarray:
    """Edge-aware denoise of merged Bayer planes in a variance-stabilised domain.

    Noise var = S*x+O is made unit-variance with y = 2*sqrt(max(var,eps)) / S-ish; here we use the per-pixel
    sigma directly: normalise by sigma, bilateral-filter with sigma_color ~ strength, then un-normalise.
    planes/var: (C,h,w). strength=0 disables.
    """
    if strength <= 0:
        return planes
    out = np.empty_like(planes)
    for c in range(planes.shape[0]):
        sig = np.sqrt(np.maximum(var[c], 1e-12)).astype(np.float32)
        # VST by dividing by local sigma gives locally unit-noise image; coarse but effective for smooth sigma
        sig_s = cv2.GaussianBlur(sig, (0, 0), 3.0)
        z = planes[c] / sig_s
        f = cv2.bilateralFilter(z.astype(np.float32), window, float(strength) * 1.5, window / 2.0)
        out[c] = f * sig_s
    return out


def denoise_chroma(rgb: np.ndarray, blur_px: float) -> np.ndarray:
    """Blur Cb/Cr only (gamma-encoded RGB in [0,1]); luma detail is untouched."""
    if blur_px <= 0:
        return rgb
    ycc = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    for ch in (1, 2):
        ycc[..., ch] = cv2.GaussianBlur(ycc[..., ch], (0, 0), blur_px)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
