import cv2
import numpy as np


def estimate_noise_model(plane: np.ndarray, bins: int = 24):
    """Fit var = S*x + O from a single normalised image using robust residuals in flat areas. Returns (S,O)."""
    p = plane.astype(np.float32)
    resid = p - cv2.GaussianBlur(p, (0, 0), 1.0)
    # a Gaussian high-pass residual of white noise has variance var*(1 + sum(g^2) - 2*g0)
    k = cv2.getGaussianKernel(9, 1.0)
    g = k @ k.T
    factor = 1.0 + float((g ** 2).sum()) - 2 * float(g[4, 4])
    base = cv2.GaussianBlur(p, (0, 0), 2.0)
    grad = np.hypot(cv2.Sobel(base, cv2.CV_32F, 1, 0), cv2.Sobel(base, cv2.CV_32F, 0, 1))
    flat = grad < np.percentile(grad, 40)
    xs, vs = [], []
    lo, hi = np.percentile(base, [1, 99])
    edges = np.linspace(lo, hi, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = flat & (base >= a) & (base < b)
        if m.sum() < 500:
            continue
        mad = np.median(np.abs(resid[m] - np.median(resid[m]))) * 1.4826
        xs.append(base[m].mean())
        vs.append(mad ** 2 / factor)
    if len(xs) < 3:
        return 1e-3, 1e-5
    A = np.stack([xs, np.ones(len(xs))], 1)
    S, O = np.linalg.lstsq(A, np.array(vs), rcond=None)[0]
    return float(max(S, 1e-7)), float(max(O, 1e-9))
