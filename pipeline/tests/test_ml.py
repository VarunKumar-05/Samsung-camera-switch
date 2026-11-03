import numpy as np
import pytest

torch = pytest.importorskip("torch")

from camlab.ml.data import synthetic_samples
from camlab.ml.kpn import KPN
from camlab.ml.train import train


def test_kpn_shapes_and_convexity():
    m = KPN(frames=3)
    x = torch.rand(2, 3, 4, 33, 35)
    out = m(x, torch.full((2, 4, 33, 35), 0.01))
    assert out.shape == (2, 4, 33, 35)
    assert out.min() >= x.min() - 1e-5 and out.max() <= x.max() + 1e-5  # convex combination of inputs


def test_kpn_learns_to_beat_reference_frame():
    X, SIG, Y = synthetic_samples(24, frames=3, size=64, seed=1)
    model, hist = train(X, SIG, Y, frames=3, steps=150, bs=8, log=lambda *_: None)
    assert np.mean(hist[-20:]) < np.mean(hist[:20])
    with torch.no_grad():
        pred = model(torch.from_numpy(X[:8]), torch.from_numpy(SIG[:8])).numpy()
    mse = lambda a: float(np.mean((a - Y[:8]) ** 2))
    assert mse(pred) < mse(X[:8, 0])  # better than just using the reference frame
