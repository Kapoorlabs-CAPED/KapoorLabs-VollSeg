"""Augmentations for StarDist training data.

All transforms operate on ``(raw, label)`` pairs and return
``(raw, label)``. Geometric transforms apply identically to both — and
because the dataset re-derives targets from the augmented label, no
ray-channel permutation is needed: any flip / rotation works in any
ndim.

Transforms compose via :class:`Compose`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


# ----------------------------------------------------------------- pipeline


class Compose(nn.Module):
    """Apply a list of ``(raw, label) -> (raw, label)`` transforms in order."""

    def __init__(self, transforms: Sequence[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(list(transforms))

    def forward(self, raw, label):
        for t in self.transforms:
            raw, label = t(raw, label)
        return raw, label


# ----------------------------------------------------------------- geometric


class RandomFlip(nn.Module):
    """Independently flip each spatial axis with probability ``p``.

    Works for any ``ndim``. Both ``raw`` and ``label`` are flipped
    identically — the label-derived targets will then describe the
    flipped scene correctly.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = float(p)

    def forward(self, raw, label):
        for axis in range(raw.dim()):
            if torch.rand(1).item() < self.p:
                raw = torch.flip(raw, dims=[axis])
                label = torch.flip(label, dims=[axis])
        return raw, label


class RandomRot90(nn.Module):
    """Random 0 / 90 / 180 / 270° rotation in the last two axes (the YX plane).

    Skipped for tensors with fewer than two dims. Identical rotation
    applied to ``raw`` and ``label``.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = float(p)

    def forward(self, raw, label):
        if torch.rand(1).item() >= self.p or raw.dim() < 2:
            return raw, label
        k = int(torch.randint(0, 4, (1,)).item())
        if k == 0:
            return raw, label
        raw = torch.rot90(raw, k=k, dims=[-2, -1])
        label = torch.rot90(label, k=k, dims=[-2, -1])
        return raw, label


# ----------------------------------------------------------------- intensity (raw-only)


class InputGaussianNoise(nn.Module):
    """Add zero-mean Gaussian noise to ``raw`` only."""

    def __init__(self, std: float = 0.01, p: float = 0.5):
        super().__init__()
        self.std = float(std)
        self.p = float(p)

    def forward(self, raw, label):
        if torch.rand(1).item() < self.p:
            raw = raw + torch.randn_like(raw) * self.std
        return raw, label


class InputPercentileNormalize(nn.Module):
    """Percentile-normalize ``raw`` to [0, 1]; leave ``label`` alone."""

    def __init__(self, pmin: float = 0.1, pmax: float = 99.9, eps: float = 1e-8):
        super().__init__()
        self.pmin = float(pmin)
        self.pmax = float(pmax)
        self.eps = float(eps)

    def forward(self, raw, label):
        flat = raw.flatten()
        lo = torch.quantile(flat, self.pmin / 100.0)
        hi = torch.quantile(flat, self.pmax / 100.0)
        raw = (raw - lo) / (hi - lo + self.eps)
        return raw, label
