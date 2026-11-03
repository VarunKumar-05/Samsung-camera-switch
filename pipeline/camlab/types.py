from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class RawBurst:
    """A burst of Bayer frames (raw DN, un-normalized) plus the metadata needed to render them."""
    frames: np.ndarray                  # (N,H,W) float32, raw digital numbers
    cfa: str = "RGGB"
    black: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))   # per CFA position
    white: float = 1023.0
    neutral: np.ndarray = field(default_factory=lambda: np.ones(3, np.float32))  # as-shot neutral (R,G,B)
    noise: Optional[np.ndarray] = None  # (4,2) (S,O) per CFA position, signal normalized to [0,1]; None -> estimate
    forward_matrix: Optional[np.ndarray] = None  # 3x3 white-balanced camera -> XYZ(D50)
    color_matrix: Optional[np.ndarray] = None    # 3x3 XYZ(D65) -> camera
    shading: Optional[np.ndarray] = None         # (rows,cols,4) gain map [R,Ge,Go,B]
    orientation: int = 0                         # degrees clockwise to display upright
    names: List[str] = field(default_factory=list)
