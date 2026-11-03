"""Load a burst folder saved by the CamLab Android app (frame_XXX.dng + meta.json)."""
import json
import os
from typing import Optional

import numpy as np

from .bayer import cfa_str
from .types import RawBurst


def _mat(x) -> Optional[np.ndarray]:
    return np.array(x, np.float32).reshape(3, 3) if x else None


def load_burst_dir(path: str, max_frames: Optional[int] = None) -> RawBurst:
    import rawpy  # imported lazily so synthetic tests do not need it

    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    if meta.get("mode") != "RAW":
        raise ValueError("not a RAW burst; use load_jpeg_burst_dir for JPEG bursts")
    st, frames_meta = meta["static"], meta["frames"]
    if max_frames:
        frames_meta = frames_meta[:max_frames]

    frames = []
    for fm in frames_meta:
        with rawpy.imread(os.path.join(path, fm["file"])) as raw:
            frames.append(raw.raw_image_visible.astype(np.float32).copy())
    frames = np.stack(frames)

    f0 = frames_meta[0]
    dyn_black = [fm.get("dynamic_black") for fm in frames_meta]
    black = np.mean([b for b in dyn_black if b], 0) if any(dyn_black) else np.array(st["black_level_pattern"], np.float32)
    whites = [fm.get("dynamic_white") for fm in frames_meta if fm.get("dynamic_white")]
    white = float(np.mean(whites)) if whites else float(st["white_level"])

    noise = None
    if f0.get("noise_profile"):
        noise = np.array(f0["noise_profile"], np.float64)
        if len(noise) == 1:
            noise = np.repeat(noise, 4, 0)

    shading = None
    if f0.get("shading"):
        s = f0["shading"]
        shading = np.array(s["gains_row_major_rggb"], np.float32).reshape(s["rows"], s["cols"], 4)

    # Prefer the illuminant-2 (usually D65) matrices; they are the daylight-ish calibration.
    fwd = _mat(st.get("forward_matrix2")) if st.get("forward_matrix2") else _mat(st.get("forward_matrix1"))
    cm = _mat(st.get("color_matrix2")) if st.get("color_matrix2") else _mat(st.get("color_matrix1"))
    return RawBurst(
        frames=frames, cfa=cfa_str(st["cfa"]), black=np.asarray(black, np.float32), white=white,
        neutral=np.array(f0["neutral"], np.float32), noise=noise, forward_matrix=fwd, color_matrix=cm,
        shading=shading, orientation=int(st.get("sensor_orientation") or 0),
        names=[fm["file"] for fm in frames_meta],
    )


def load_jpeg_burst_dir(path: str, max_frames: Optional[int] = None) -> np.ndarray:
    """JPEG fallback: returns (N,H,W,3) float32 sRGB in [0,1]."""
    import cv2
    files = sorted(f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg")))
    if max_frames:
        files = files[:max_frames]
    return np.stack([cv2.cvtColor(cv2.imread(os.path.join(path, f)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                     for f in files])
