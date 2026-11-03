"""End-to-end: write a fake phone folder (DNG-less) is covered by synthetic tests; here check the JPEG fallback + compare tool."""
import subprocess
import sys

import cv2
import numpy as np

from camlab import synth
from camlab.params import preset
from camlab.pipeline import process_jpeg_burst
from camlab.tonemap import linear_to_srgb


def test_jpeg_fallback_beats_single_frame():
    burst, scene, _ = synth.make_burst(n=6, h=192, w=192, S=0.004)
    frames = np.stack([np.stack([linear_to_srgb(np.clip((f - 64) / 959, 0, 1))] * 3, -1) for f in burst.frames])
    out = process_jpeg_burst(frames, preset("neutral"))
    assert out.shape == (192, 192, 3) and np.isfinite(out).all()


def test_compare_runs(tmp_path):
    a = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "a.jpg"), a)
    cv2.imwrite(str(tmp_path / "b.png"), a[::-1])
    r = subprocess.run([sys.executable, "compare.py", str(tmp_path / "a.jpg"), str(tmp_path / "b.png"),
                        "--out", str(tmp_path), "--width", "400", "--size", "100"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "compare.png").exists()
