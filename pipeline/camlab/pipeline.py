from typing import Dict, Optional

import numpy as np

from .align import warp
from .bayer import demosaic, pack, shading_channel, unpack
from .denoise import denoise_chroma, denoise_planes
from .merge import align_burst, merge_burst, pick_reference
from .noise import estimate_noise_model
from .params import Params, preset
from .sharpen import sharpen_luma
from .tonemap import linear_to_srgb, s_curve, tone_map
from .types import RawBurst

XYZ_D50_TO_SRGB = np.array([[3.1338561, -1.6168667, -0.4906146],
                            [-0.9787684, 1.9161415, 0.0334540],
                            [0.0719453, -0.2289914, 1.4052427]], np.float32)
XYZ_D65_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                            [-0.9692660, 1.8760108, 0.0415560],
                            [0.0556434, -0.2040259, 1.0572252]], np.float32)


def normalize(burst: RawBurst) -> np.ndarray:
    """(N,4,h,w) planes, black-subtracted and scaled so white=1 (negatives kept: they carry noise info)."""
    planes = np.stack([pack(f) for f in burst.frames])
    black = np.asarray(burst.black, np.float32).reshape(1, 4, 1, 1)
    return (planes - black) / max(burst.white - float(black.mean()), 1.0)


def noise_params(burst: RawBurst, planes: np.ndarray):
    if burst.noise is not None:
        n = np.asarray(burst.noise, np.float64)
        return n[:, 0].astype(np.float32), n[:, 1].astype(np.float32)
    ref = planes[0]
    est = [estimate_noise_model(ref[c]) for c in range(ref.shape[0])]
    return np.array([e[0] for e in est], np.float32), np.array([e[1] for e in est], np.float32)


def _shading_gain(burst: RawBurst, h: int, w: int) -> Optional[np.ndarray]:
    if burst.shading is None:
        return None
    import cv2
    ch = shading_channel(burst.cfa)
    maps = [cv2.resize(burst.shading[..., c].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) for c in ch]
    return np.stack(maps)


def cam_to_linear_srgb(cam: np.ndarray, burst: RawBurst) -> np.ndarray:
    """cam (H,W,3) un-white-balanced camera RGB -> linear sRGB."""
    neutral = np.asarray(burst.neutral, np.float32)
    wb = cam / neutral.reshape(1, 1, 3)
    if burst.forward_matrix is not None:
        M = XYZ_D50_TO_SRGB @ burst.forward_matrix
        return np.einsum("ij,hwj->hwi", M, wb)
    if burst.color_matrix is not None:
        M = XYZ_D65_TO_SRGB @ np.linalg.inv(burst.color_matrix)
        return np.einsum("ij,hwj->hwi", M, wb)
    return wb


def process_burst(burst: RawBurst, params: Optional[Params] = None, ref_idx: Optional[int] = None,
                  stages: Optional[Dict[str, np.ndarray]] = None, ml_model=None) -> np.ndarray:
    """RawBurst -> float32 RGB in [0,1] (sRGB gamma). If `stages` is a dict it receives intermediates."""
    P = params or preset("natural")
    planes = normalize(burst)
    S, O = noise_params(burst, planes)
    ref_idx = pick_reference(planes.mean(1)) if ref_idx is None else ref_idx
    flows = align_burst(planes, ref_idx, P.merge.tile, P.merge.levels)
    if ml_model is not None:
        from .ml.infer import kpn_merge
        merged, var = kpn_merge(ml_model, planes, flows, ref_idx, S, O)
    else:
        merged, var = merge_burst(planes, flows, ref_idx, S, O, P.merge.strictness, P.merge.window)

    merged = denoise_planes(merged, var, P.denoise.luma, P.denoise.spatial)
    gain = _shading_gain(burst, *merged.shape[1:])
    if gain is not None:
        merged = merged * gain
    cam = demosaic(unpack(merged), burst.cfa)
    lin = np.maximum(cam_to_linear_srgb(cam, burst), 0)
    if stages is not None:
        stages["ref_idx"] = ref_idx
        stages["single_frame_cam"] = demosaic(unpack(planes[ref_idx] * (gain if gain is not None else 1)), burst.cfa)
        stages["linear"] = lin
    srgb = linear_to_srgb(tone_map(lin, P.tone))
    srgb = s_curve(srgb, P.tone.contrast)
    srgb = denoise_chroma(srgb, P.denoise.chroma)
    srgb = sharpen_luma(srgb, P.sharpen.amount, P.sharpen.radius, P.sharpen.threshold, P.sharpen.halo)
    return np.clip(srgb, 0, 1).astype(np.float32)


def rotate_upright(img: np.ndarray, degrees: int) -> np.ndarray:
    k = (-degrees // 90) % 4  # np.rot90 is counter-clockwise
    return np.rot90(img, k)


def process_jpeg_burst(frames: np.ndarray, params: Optional[Params] = None) -> np.ndarray:
    """Fallback when the phone will not give RAW: merge a burst of high-quality JPEGs in linear light.

    frames: (N,H,W,3) sRGB in [0,1]. Quality is bounded by the phone's own in-camera processing.
    """
    from .tonemap import srgb_to_linear
    P = params or preset("natural")
    lin = srgb_to_linear(frames)
    planes = lin.transpose(0, 3, 1, 2).copy()              # (N,3,H,W)
    ref = pick_reference(planes.mean(1))
    S, O = noise_params(RawBurst(frames=frames), planes)
    flows = align_burst(planes, ref, P.merge.tile, P.merge.levels)
    merged, _ = merge_burst(planes, flows, ref, S, O, P.merge.strictness, P.merge.window)
    srgb = linear_to_srgb(merged.transpose(1, 2, 0))
    srgb = denoise_chroma(srgb, P.denoise.chroma)
    return np.clip(sharpen_luma(srgb, P.sharpen.amount, P.sharpen.radius, P.sharpen.threshold, P.sharpen.halo), 0, 1)
