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

    Every field is optional — each pipeline fills only what it produces and
    composites pass everything else through.

    Stage outputs (each is populated only when the corresponding model
    runs; ``None`` otherwise):

    - ``denoised``        — CARE denoised image (CARE only).
    - ``roi``             — ROI gating mask (Mask-UNet only).
    - ``unet_labels``     — connected-component labels of the U-Net
                            semantic mask (U-Net only).
    - ``stardist_labels`` — raw StarDist instance labels (StarDist only).
    - ``vollseg_labels``  — watershed-fused VollSeg labels (seed-pool
                            branch only).

    Convenience aliases — point at one of the stage outputs above:

    - ``labels``      — canonical instance labels = ``vollseg_labels`` if
                        the seed-pool branch ran, else ``stardist_labels``,
                        else ``unet_labels``.
    - ``semantic``    — the binary mask actually used in the segmentation
                        stage (U-Net mask when supplied, or the
                        Otsu-threshold fallback in the no-U-Net seed-pool
                        path).
    - ``probability`` — probability map from whichever network emitted one.
    - ``polys``       — StarDist polyhedron metadata (vertices, faces).
    - ``extra``       — escape hatch for pipeline-specific outputs.
    """

    labels: Optional[np.ndarray] = None
    stardist_labels: Optional[np.ndarray] = None
    vollseg_labels: Optional[np.ndarray] = None
    unet_labels: Optional[np.ndarray] = None
    semantic: Optional[np.ndarray] = None
    denoised: Optional[np.ndarray] = None
    roi: Optional[np.ndarray] = None
    probability: Optional[np.ndarray] = None
    polys: Optional[Any] = None
    extra: dict = field(default_factory=dict)

    def merge(self, **updates) -> Result:
        """Return a copy of this Result with the given fields overwritten."""
        return replace(self, **updates)


@runtime_checkable
class Pipeline(Protocol):
    """Anything that turns an image into a Result."""

    def predict(self, image: np.ndarray, **kwargs) -> Result:
        ...


def infer_axes(image: np.ndarray) -> str:
    """Best-effort axes string for csbdeep/stardist from ``image.ndim``."""
    if image.ndim == 2:
        return "YX"
    if image.ndim == 3:
        return "ZYX"
    if image.ndim == 4:
        return "ZYXC"
    raise ValueError(f"Cannot infer axes for ndim={image.ndim}")
