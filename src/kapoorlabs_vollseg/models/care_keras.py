"""CARE denoising singleton (keras / csbdeep) — legacy.

New code should use :class:`kapoorlabs_vollseg.CAREDenoiser` (PyTorch + Lightning).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .._backbones.care_keras import CAREBackboneKeras
from ..pipelines.base import Result, infer_axes


class CAREDenoiserKeras:
    """Run a csbdeep-CARE denoising network on an image."""

    def __init__(self, backbone: CAREBackboneKeras):
        self.backbone = backbone

    @classmethod
    def from_pretrained(cls, name_or_alias: str) -> CAREDenoiserKeras:
        from ..pretrained import get_model_instance

        return cls(get_model_instance(CAREBackboneKeras, name_or_alias))

    def predict(
        self,
        image: np.ndarray,
        *,
        axes: Optional[str] = None,
        n_tiles: Optional[tuple] = None,
        **_ignored,
    ) -> Result:
        if axes is None:
            axes = infer_axes(image)
        denoised = self.backbone.predict(
            image.astype("float32"), axes=axes, n_tiles=n_tiles
        )
        return Result(denoised=denoised)
