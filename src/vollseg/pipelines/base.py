"""The Pipeline protocol and the Result dataclass that flows through it.

Every Layer 1 singleton and every Layer 2 composite implements ``predict``
with the same shape: ``image -> Result``. That uniform contract is what
lets composites wrap singletons (and other composites) without knowing
which one they got.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class Result:
    """Output bundle from any pipeline.

    Fields are all optional — each pipeline fills only what it produces, and
    composites pass everything else through. ``labels`` is the canonical
    instance segmentation; ``semantic`` is the raw binary U-Net mask;
    ``denoised`` is the CARE output; ``roi`` is the gating mask.
    """

    labels: Optional[np.ndarray] = None
    semantic: Optional[np.ndarray] = None
    denoised: Optional[np.ndarray] = None
    roi: Optional[np.ndarray] = None
    probability: Optional[np.ndarray] = None
    polys: Optional[Any] = None
    extra: dict = field(default_factory=dict)

    def merge(self, **updates) -> "Result":
        """Return a copy of this Result with the given fields overwritten."""
        return replace(self, **updates)


@runtime_checkable
class Pipeline(Protocol):
    """Anything that turns an image into a Result."""

    def predict(self, image: np.ndarray, **kwargs) -> Result: ...


def infer_axes(image: np.ndarray) -> str:
    """Best-effort axes string for csbdeep/stardist from ``image.ndim``."""
    if image.ndim == 2:
        return "YX"
    if image.ndim == 3:
        return "ZYX"
    if image.ndim == 4:
        return "ZYXC"
    raise ValueError(f"Cannot infer axes for ndim={image.ndim}")
