"""Minimal tensor transforms for CARE-style prediction.

A 5-line :class:`PercentileNormalize` and a cast to ``float32`` —
deliberately tiny so we don't need to import the oneat package for them.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PercentileNormalize(nn.Module):
    """Clip to ``[pmin, pmax]`` percentiles and rescale to [0, 1]."""

    def __init__(self, pmin: float = 0.1, pmax: float = 99.9, eps: float = 1e-8):
        super().__init__()
        self.pmin = float(pmin)
        self.pmax = float(pmax)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        lo = torch.quantile(x.flatten(), self.pmin / 100.0)
        hi = torch.quantile(x.flatten(), self.pmax / 100.0)
        return (x - lo) / (hi - lo + self.eps)


class ToFloat32(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)
