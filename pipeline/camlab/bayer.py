"""Bayer helpers. Planes are packed in *position* order: (even,even) (even,odd) (odd,even) (odd,odd)."""
import cv2
import numpy as np

CFA_NAMES = ["RGGB", "GRBG", "GBRG", "BGGR"]  # Android SENSOR_INFO_COLOR_FILTER_ARRANGEMENT 0..3


def cfa_str(c) -> str:
    return CFA_NAMES[c] if isinstance(c, (int, np.integer)) else str(c)


def pack(mosaic: np.ndarray) -> np.ndarray:
    H, W = mosaic.shape
    H -= H % 2
    W -= W % 2
    return np.stack([mosaic[0:H:2, 0:W:2], mosaic[0:H:2, 1:W:2],
                     mosaic[1:H:2, 0:W:2], mosaic[1:H:2, 1:W:2]]).astype(np.float32)


def unpack(planes: np.ndarray) -> np.ndarray:
    _, h, w = planes.shape
    m = np.empty((h * 2, w * 2), planes.dtype)
    m[0::2, 0::2], m[0::2, 1::2], m[1::2, 0::2], m[1::2, 1::2] = planes
    return m


def shading_channel(cfa: str) -> list:
    """Android lens-shading maps are ordered [R, G_even, G_odd, B]; map each CFA position to that index."""
    out, seen_g = [], 0
    for ch in cfa:
        if ch == "R":
            out.append(0)
        elif ch == "B":
            out.append(3)
        else:
            out.append(1 + seen_g)
            seen_g += 1
    return out


_KG = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], np.float32) / 8
_KH = np.array([[0, 0, .5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, .5, 0, 0]], np.float32) / 8
_KV = _KH.T.copy()
_KD = np.array([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]], np.float32) / 8


def demosaic(mosaic: np.ndarray, cfa: str = "RGGB") -> np.ndarray:
    """Malvar-He-Cutler demosaic -> (H,W,3) RGB, same units as input."""
    H, W = mosaic.shape
    m = mosaic.astype(np.float32)
    f = lambda k: cv2.filter2D(m, -1, k, borderType=cv2.BORDER_REFLECT)
    pos = {ch: [] for ch in "RGB"}
    for i, ch in enumerate(cfa):
        pos[ch].append(divmod(i, 2))

    def mask(cells):
        a = np.zeros((H, W), bool)
        for ry, rx in cells:
            a[ry::2, rx::2] = True
        return a

    (ry, rx), (by, bx) = pos["R"][0], pos["B"][0]
    g_in_r_row = mask([(ry, 1 - rx)])   # G sharing a row with R (so also a column with B)
    g_in_b_row = mask([(by, 1 - bx)])   # G sharing a row with B (so a column with R)
    is_r, is_b, is_g = mask(pos["R"]), mask(pos["B"]), mask(pos["G"])

    gk, hk, vk, dk = f(_KG), f(_KH), f(_KV), f(_KD)
    G = np.where(is_g, m, gk)
    R = np.where(is_r, m, np.where(g_in_r_row, hk, np.where(g_in_b_row, vk, dk)))
    B = np.where(is_b, m, np.where(g_in_b_row, hk, np.where(g_in_r_row, vk, dk)))
    return np.stack([R, G, B], -1)
