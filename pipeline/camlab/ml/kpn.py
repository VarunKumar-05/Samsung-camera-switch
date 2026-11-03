"""Kernel-prediction burst merger (Mildenhall et al. 2018, simplified), working on packed Bayer planes.

Input : K classically-aligned frames (B,K,4,h,w), frame 0 = reference, plus the reference noise sigma (B,4,h,w).
Output: (B,4,h,w) merged planes = sum over frames & k*k taps of softmax-normalised per-pixel weights.
The network can down-weight misaligned/moving content and spatially denoise at the same time, because the
kernels span a k*k neighbourhood of every frame. Because output is a convex combination of input pixels it
cannot hallucinate texture, which keeps results "natural".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True))


class KPN(nn.Module):
    def __init__(self, frames: int = 4, k: int = 3, base: int = 24):
        super().__init__()
        self.frames, self.k = frames, k
        cin = frames * 4 + 4 + frames  # frames + sigma + per-frame |diff to ref| summary
        self.e1, self.e2, self.e3 = _block(cin, base), _block(base, base * 2), _block(base * 2, base * 4)
        self.d2, self.d1 = _block(base * 4 + base * 2, base * 2), _block(base * 2 + base, base)
        self.head = nn.Conv2d(base, frames * k * k, 1)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        B, K, C, h, w = x.shape
        diff = (x - x[:, :1]).abs().mean(2) / (sigma.mean(1, keepdim=True) + 1e-4)  # (B,K,h,w), noise-normalised
        inp = torch.cat([x.reshape(B, K * C, h, w), sigma, diff], 1)
        H, W = h + (-h) % 4, w + (-w) % 4  # pad to a multiple of 4 for two pooling steps
        inp = F.pad(inp, (0, W - w, 0, H - h), mode="replicate")
        e1 = self.e1(inp)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        logits = self.head(d1)[..., :h, :w]                                   # (B,K*k*k,h,w)
        wts = torch.softmax(logits, dim=1).view(B, K, 1, self.k * self.k, h, w)
        patches = F.unfold(x.reshape(B, K * C, h, w), self.k, padding=self.k // 2)
        patches = patches.view(B, K, C, self.k * self.k, h, w)
        return (wts * patches).sum(dim=(1, 3))
