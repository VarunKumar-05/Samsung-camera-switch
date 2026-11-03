import numpy as np
import torch

from ..merge import align_burst
from .data import aligned_stack
from .kpn import KPN


def load(path: str) -> KPN:
    ck = torch.load(path, map_location="cpu")
    m = KPN(frames=ck["frames"])
    m.load_state_dict(ck["state"])
    return m.eval()


@torch.no_grad()
def kpn_merge(model: KPN, planes: np.ndarray, flows, ref_idx: int, S: np.ndarray, O: np.ndarray):
    """Merge using the K-1 frames closest to the reference (lowest aligned difference). Returns (merged, var_out)."""
    K = model.frames
    ref = planes[ref_idx]
    from ..align import warp
    cost = {i: float(np.mean((warp(planes[i], flows[i]) - ref) ** 2)) for i in range(len(planes)) if i != ref_idx}
    pick = sorted(cost, key=cost.get)[:K - 1]
    while len(pick) < K - 1:
        pick.append(pick[-1])
    stack = aligned_stack(planes, flows, ref_idx, pick)
    sigma = np.sqrt(S[:, None, None] * np.maximum(ref, 0) + O[:, None, None]).astype(np.float32)
    out = model(torch.from_numpy(stack)[None], torch.from_numpy(sigma)[None])[0].numpy()
    return out, (sigma ** 2) / K
