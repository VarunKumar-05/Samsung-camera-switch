import cv2
import numpy as np
import pytest

from camlab import synth
from camlab.align import estimate_flow
from camlab.bayer import demosaic, pack, unpack
from camlab.merge import align_burst, merge_burst
from camlab.params import preset
from camlab.pipeline import normalize, process_burst


def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - b) ** 2))


def test_pack_unpack_roundtrip():
    m = np.random.rand(8, 10).astype(np.float32)
    assert np.array_equal(unpack(pack(m)), m)


@pytest.mark.parametrize("cfa", ["RGGB"])
def test_demosaic_smooth_image(cfa):
    scene = synth.make_scene(128, 128)
    scene = cv2.GaussianBlur(scene, (0, 0), 3)
    out = demosaic(synth.mosaic_rggb(scene), cfa)
    assert psnr(out[8:-8, 8:-8], scene[8:-8, 8:-8]) > 40


def test_alignment_recovers_subpixel_shift():
    scene = cv2.GaussianBlur(synth.make_scene(256, 256, 3), (0, 0), 1.0)
    g = scene.mean(-1)
    dx, dy = 3.3, -2.1
    alt = cv2.warpAffine(g, np.float32([[1, 0, -dx], [0, 1, -dy]]), (256, 256), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    flow = estimate_flow(g, alt)
    inner = flow[32:-32, 32:-32]
    assert abs(np.median(inner[..., 0]) + dx) < 0.25  # alt(x)=g(x+d) so flow=-d
    assert abs(np.median(inner[..., 1]) + dy) < 0.25


def test_merge_reduces_noise():
    burst, scene, _ = synth.make_burst(n=8, S=0.006, O=5e-5)
    planes = normalize(burst)
    flows = align_burst(planes, 0)
    S, O = np.full(4, 0.006, np.float32), np.full(4, 5e-5, np.float32)
    merged, var = merge_burst(planes, flows, 0, S, O)
    truth = pack(synth.mosaic_rggb(scene))
    single = psnr(planes[0][:, 16:-16, 16:-16], truth[:, 16:-16, 16:-16])
    multi = psnr(merged[:, 16:-16, 16:-16], truth[:, 16:-16, 16:-16])
    assert multi > single + 3.0, (single, multi)


def test_merge_does_not_ghost():
    burst, scene, _ = synth.make_burst(n=8, moving_square=True, shift=1.0)
    planes = normalize(burst)
    flows = align_burst(planes, 0)
    S, O = np.full(4, 0.004, np.float32), np.full(4, 2e-5, np.float32)
    merged, _ = merge_burst(planes, flows, 0, S, O)
    naive = planes.mean(0)
    # region of the moving square in half-res plane coordinates (frames 1.. paint it at several x offsets)
    sl = (slice(None), slice(50, 65), slice(30, 120))
    truth = pack(synth.mosaic_rggb(scene))
    assert np.mean((merged[sl] - truth[sl]) ** 2) < 0.6 * np.mean((naive[sl] - truth[sl]) ** 2)


def test_full_pipeline_and_knobs():
    burst, scene, _ = synth.make_burst(n=6, h=192, w=192)
    stages = {}
    img = process_burst(burst, preset("natural"), stages=stages)
    assert img.shape == (192, 192, 3) and img.dtype == np.float32
    assert np.isfinite(img).all() and 0 <= img.min() and img.max() <= 1
    # sharpening knob must raise high-frequency energy; denoise knob must lower noise on a flat-ish area
    p0, p1 = preset("natural"), preset("natural")
    p0.sharpen.amount, p1.sharpen.amount = 0.0, 1.5
    lap = lambda im: cv2.Laplacian(im.mean(-1), cv2.CV_32F).var()
    assert lap(process_burst(burst, p1)) > lap(process_burst(burst, p0))
    d0, d1 = preset("natural"), preset("natural")
    d0.denoise.luma, d1.denoise.luma = 0.0, 2.0
    d0.denoise.chroma, d1.denoise.chroma = 0.0, 4.0
    for p in (d0, d1):
        p.sharpen.amount = 0
    flat = (slice(150, 180), slice(10, 40))
    assert process_burst(burst, d1)[flat].std() <= process_burst(burst, d0)[flat].std()
