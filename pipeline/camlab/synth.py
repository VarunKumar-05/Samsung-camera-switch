"""Synthetic bursts with known ground truth, for tests and for pre-training the ML merge."""
import cv2
import numpy as np

from .bayer import unpack
from .types import RawBurst


def make_scene(h=256, w=256, seed=0) -> np.ndarray:
    """Linear RGB scene in [0.02,0.9] with edges, texture and gradients."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([0.2 + 0.5 * xx / w, 0.2 + 0.5 * yy / h, 0.4 + 0.2 * np.sin(xx / 17)], -1)
    for _ in range(14):
        x0, y0 = rng.integers(0, w - 40), rng.integers(0, h - 40)
        s = rng.integers(15, 40)
        img[y0:y0 + s, x0:x0 + s] = rng.uniform(0.05, 0.85, 3)
    tex = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 1.2) * 0.05
    return np.clip(img + tex[..., None], 0.02, 0.9).astype(np.float32)


def mosaic_rggb(rgb: np.ndarray) -> np.ndarray:
    h, w, _ = rgb.shape
    m = np.empty((h, w), np.float32)
    m[0::2, 0::2], m[0::2, 1::2], m[1::2, 0::2], m[1::2, 1::2] = rgb[0::2, 0::2, 0], rgb[0::2, 1::2, 1], rgb[1::2, 0::2, 1], rgb[1::2, 1::2, 2]
    return m


def make_burst(n=8, h=256, w=256, shift=2.5, S=0.004, O=2e-5, seed=0, black=64.0, white=1023.0,
               moving_square=False, scene=None):
    """Returns (RawBurst, clean_scene, true_shifts). Frame i samples the scene shifted by true_shifts[i] (dx,dy)."""
    rng = np.random.default_rng(seed + 1)
    scene = make_scene(h, w, seed) if scene is None else scene
    shifts = [(0.0, 0.0)] + [tuple(rng.uniform(-shift, shift, 2)) for _ in range(n - 1)]
    frames = []
    for i, (dx, dy) in enumerate(shifts):
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])  # frame(x) = scene(x + d)
        rgb = cv2.warpAffine(scene, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        if moving_square and i > 0:
            x0 = 60 + 12 * i
            rgb[100:130, x0:x0 + 30] = 0.9
        clean = mosaic_rggb(np.clip(rgb, 0, 1))
        sigma = np.sqrt(S * clean + O)
        noisy = clean + rng.standard_normal(clean.shape).astype(np.float32) * sigma
        frames.append(noisy * (white - black) + black)
    burst = RawBurst(frames=np.stack(frames).astype(np.float32), cfa="RGGB", black=np.full(4, black, np.float32),
                     white=white, neutral=np.ones(3, np.float32), noise=np.tile([S, O], (4, 1)))
    return burst, scene, shifts
