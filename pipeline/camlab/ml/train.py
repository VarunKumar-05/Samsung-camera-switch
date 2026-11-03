"""python -m camlab.ml.train --out kpn.pt [--bursts DIR1 DIR2 ...] [--steps 2000]"""
import argparse

import numpy as np
import torch

from .data import distill_samples, synthetic_samples
from .kpn import KPN


def train(X, SIG, Y, frames=4, steps=1000, bs=8, lr=2e-3, log=print, seed=0):
    torch.manual_seed(seed)
    model = KPN(frames=frames)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    X, SIG, Y = map(torch.from_numpy, (X, SIG, Y))
    n = len(X)
    hist = []
    for step in range(steps):
        idx = torch.randint(0, n, (bs,))
        pred = model(X[idx], SIG[idx])
        # L1 in a sqrt domain ~ variance-stabilised, so dark and bright areas count comparably
        loss = (torch.sqrt(pred.clamp(min=0) + 1e-3) - torch.sqrt(Y[idx].clamp(min=0) + 1e-3)).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(loss.item())
        if step % 50 == 0:
            log(f"step {step:5d}  loss {np.mean(hist[-50:]):.5f}")
    return model, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kpn.pt")
    ap.add_argument("--bursts", nargs="*", default=[], help="real RAW burst folders for distillation")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--synthetic", type=int, default=150)
    a = ap.parse_args()
    parts = [synthetic_samples(a.synthetic, a.frames)]
    if a.bursts:
        from ..io import load_burst_dir
        for d in a.bursts:
            parts.append(distill_samples(load_burst_dir(d), a.frames))
    X, SIG, Y = (np.concatenate([p[i] for p in parts]) for i in range(3))
    model, _ = train(X, SIG, Y, a.frames, a.steps)
    torch.save({"frames": a.frames, "state": model.state_dict()}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
