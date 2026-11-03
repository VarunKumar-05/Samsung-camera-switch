import cv2
import numpy as np

from .params import ToneParams

_LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def tone_map(lin: np.ndarray, p: ToneParams) -> np.ndarray:
    """Linear sRGB-primaries RGB -> display-referred linear RGB in [0,1] (gamma applied by caller)."""
    eps = 1e-5
    lin = np.maximum(lin, 0)
    L = lin @ _LUM + eps
    if p.auto_expose:
        lin = lin * (0.18 / float(np.exp(np.mean(np.log(L)))))
        L = lin @ _LUM + eps
    gain = 2.0 ** p.exposure_ev
    lin, L = lin * gain, L * gain

    logL = np.log2(L)
    sigma = max(2.0, p.base_sigma_frac * max(L.shape))
    small = cv2.resize(logL, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    base = cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / 4), (L.shape[1], L.shape[0]), interpolation=cv2.INTER_LINEAR)
    detail = logL - base
    pivot = np.log2(0.18)
    base2 = (base - pivot) * p.compress + pivot
    if p.shadows_ev:
        base2 = base2 + p.shadows_ev * np.clip((pivot - base2) / 4.0, 0, 1)
    L2 = np.exp2(base2 + detail * p.detail)
    ratio = L2 / L
    rgb = lin * ratio[..., None]
    if p.saturation != 1.0:
        l = rgb @ _LUM
        rgb = l[..., None] + (rgb - l[..., None]) * p.saturation
        rgb = np.maximum(rgb, 0)
    w2 = p.white ** 2
    rgb = rgb * (1 + rgb / w2) / (1 + rgb)  # extended Reinhard shoulder: protects highlights
    return np.clip(rgb, 0, 1)


def s_curve(srgb: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return srgb
    s = srgb * srgb * (3 - 2 * srgb)  # smoothstep
    return srgb * (1 - strength) + s * strength
