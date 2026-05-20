"""
Paired transforms for CARE (Content-Aware image REstoration).

Geometric transforms (flip, rotation) are applied identically to both
low SNR input and high SNR target. Intensity augmentations are applied
only to the input.

``PercentileNormalize`` and ``ToFloat32`` were originally in
``kapoorlabs_lightning.oneat_transforms`` — they're generic image
preprocessing utilities with nothing to do with oneat, so they're
inlined here next to the CARE-specific transforms that use them.
"""

import torch
import torch.nn as nn


class PercentileNormalize(nn.Module):
    def __init__(self, pmin: float = 1.0, pmax: float = 99.8, eps: float = 1e-8):
        super().__init__()
        self.pmin = pmin
        self.pmax = pmax
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mi = torch.quantile(x.flatten(), self.pmin / 100.0)
        ma = torch.quantile(x.flatten(), self.pmax / 100.0)
        x = (x - mi) / (ma - mi + self.eps)
        return torch.clamp(x, 0, 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pmin={self.pmin}, pmax={self.pmax})"


class ToFloat32(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PairedRandomSpatialFlip(nn.Module):
    """Apply identical random flips to a pair of 3D volumes (ZYX)."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        if low.dim() == 3:
            # Z flip
            if torch.rand(1).item() < self.p:
                low = torch.flip(low, dims=[0])
                high = torch.flip(high, dims=[0])
            # Y flip
            if torch.rand(1).item() < self.p:
                low = torch.flip(low, dims=[1])
                high = torch.flip(high, dims=[1])
            # X flip
            if torch.rand(1).item() < self.p:
                low = torch.flip(low, dims=[2])
                high = torch.flip(high, dims=[2])
        return low, high


class PairedRandomRotation90(nn.Module):
    """Apply identical random 90-degree rotation in YX plane to a pair."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        if torch.rand(1).item() < self.p:
            k = torch.randint(0, 4, (1,)).item()
            if low.dim() == 3:
                low = torch.rot90(low, k, dims=[1, 2])
                high = torch.rot90(high, k, dims=[1, 2])
        return low, high


class InputGaussianNoise(nn.Module):
    """Add Gaussian noise to input only (not target)."""

    def __init__(self, std: float = 0.01, p: float = 0.5):
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(low) * self.std
            low = low + noise
        return low, high


class PairedPercentileNormalize(nn.Module):
    """Apply percentile normalization independently to each volume."""

    def __init__(self, pmin: float = 0.1, pmax: float = 99.9, eps: float = 1e-8):
        super().__init__()
        self.norm = PercentileNormalize(pmin=pmin, pmax=pmax, eps=eps)

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        return self.norm(low), self.norm(high)


class PairedToFloat32(nn.Module):
    """Convert both volumes to float32."""

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        return low.to(torch.float32), high.to(torch.float32)


__all__ = [
    "PairedRandomSpatialFlip",
    "PairedRandomRotation90",
    "InputGaussianNoise",
    "PairedPercentileNormalize",
    "PairedToFloat32",
]
