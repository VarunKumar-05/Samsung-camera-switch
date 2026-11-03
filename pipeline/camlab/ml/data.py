"""Training data. Two sources:
  * synthetic bursts with exact ground truth (bootstraps the model, no phone needed)
  * 'burst distillation' from YOUR real bursts: target = classical merge of all frames, input = a short subset.
    This teaches the net to approach an N-frame result from K frames, using your sensor's real noise.
"""
from typing import List

import numpy as np

from .. import synth
from ..merge import align_burst, merge_burst, pick_reference
from ..align import warp
from ..pipeline import noise_params, normalize


def aligned_stack(planes: np.ndarray, flows, ref_idx: int, idx: List[int]) -> np.ndarray:
    """(K,4,h,w): reference first, then the chosen frames warped onto it."""
    out = [planes[ref_idx]]
    for i in idx:
        out.append(warp(planes[i], flows[i]))
    return np.stack(out).astype(np.float32)


def synthetic_samples(n: int, frames: int = 4, size: int = 96, seed: int = 0):
    from ..bayer import pack
    rng = np.random.default_rng(seed)
    X, SIG, Y = [], [], []
    for i in range(n):
        S = float(rng.uniform(0.002, 0.02))
        O = float(rng.uniform(1e-5, 2e-4))
        burst, scene, _ = synth.make_burst(n=frames, h=size, w=size, shift=2.0, S=S, O=O, seed=int(rng.integers(1 << 30)),
                                           moving_square=bool(rng.random() < 0.3))
        planes = normalize(burst)
        flows = align_burst(planes, 0, tile=16, levels=3)
        stack = aligned_stack(planes, flows, 0, list(range(1, frames)))
        X.append(stack)
        SIG.append(np.sqrt(S * np.maximum(planes[0], 0) + O).astype(np.float32))
        Y.append(pack(synth.mosaic_rggb(scene)))
    return np.stack(X), np.stack(SIG), np.stack(Y)


def distill_samples(burst, frames: int = 4, crops: int = 40, size: int = 128, seed: int = 0):
    """Real RawBurst -> (X, SIG, Y). Target is the classical merge of the *whole* burst."""
    rng = np.random.default_rng(seed)
    planes = normalize(burst)
    S, O = noise_params(burst, planes)
    ref = pick_reference(planes.mean(1))
    flows = align_burst(planes, ref)
    target, _ = merge_burst(planes, flows, ref, S, O)
    others = [i for i in range(len(planes)) if i != ref]
    _, _, h, w = planes.shape
    X, SIG, Y = [], [], []
    for _ in range(crops):
        pick = list(rng.choice(others, size=min(frames - 1, len(others)), replace=False))
        stack = aligned_stack(planes, flows, ref, pick)
        y0, x0 = rng.integers(0, h - size), rng.integers(0, w - size)
        sl = (slice(None), slice(None), slice(y0, y0 + size), slice(x0, x0 + size))
        X.append(stack[sl])
        SIG.append(np.sqrt(S[:, None, None] * np.maximum(planes[ref][:, y0:y0 + size, x0:x0 + size], 0) + O[:, None, None]))
        Y.append(target[:, y0:y0 + size, x0:x0 + size])
    return np.stack(X).astype(np.float32), np.stack(SIG).astype(np.float32), np.stack(Y).astype(np.float32)
