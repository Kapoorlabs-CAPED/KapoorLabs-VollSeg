"""Decorator pipeline: denoise with CARE, then run a downstream pipeline."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..models.care import CAREDenoiser
from .base import Pipeline, Result


class DenoisedPipeline:
    """Wrap any downstream pipeline with a CARE denoising pre-step.

    The downstream pipeline receives the *denoised* image, not the original.
    The denoised image is also attached to the returned ``Result`` so callers
    can inspect or save it.
    """

    def __init__(self, care: CAREDenoiser, downstream: Pipeline):
        if not isinstance(care, Pipeline):
            raise TypeError(f"care must be a Pipeline, got {type(care).__name__}")
        if not isinstance(downstream, Pipeline):
            raise TypeError(f"downstream must be a Pipeline, got {type(downstream).__name__}")
        self.care = care
        self.downstream = downstream

    def predict(
        self,
        image: np.ndarray,
        *,
        axes: Optional[str] = None,
        n_tiles: Optional[tuple] = None,
        **kwargs,
    ) -> Result:
        denoised = self.care.predict(image, axes=axes, n_tiles=n_tiles, **kwargs).denoised
        result = self.downstream.predict(denoised, axes=axes, n_tiles=n_tiles, **kwargs)
        return result.merge(denoised=denoised)
