"""Decorator pipeline: gate a downstream pipeline by an ROI mask."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Pipeline, Result


class ROIPipeline:
    """Restrict a downstream pipeline's output to a U-Net-defined ROI.

    A region-of-interest U-Net runs on the original image to produce a
    binary mask. The downstream pipeline still sees the full image (so its
    receptive field is not truncated), but its output ``labels`` /
    ``semantic`` are zeroed outside the ROI.
    """

    def __init__(self, roi_unet: Pipeline, downstream: Pipeline):
        if not isinstance(roi_unet, Pipeline):
            raise TypeError(
                f"roi_unet must be a Pipeline, got {type(roi_unet).__name__}"
            )
        if not isinstance(downstream, Pipeline):
            raise TypeError(
                f"downstream must be a Pipeline, got {type(downstream).__name__}"
            )
        self.roi_unet = roi_unet
        self.downstream = downstream

    def predict(
        self,
        image: np.ndarray,
        *,
        axes: Optional[str] = None,
        n_tiles: Optional[tuple] = None,
        **kwargs,
    ) -> Result:
        roi = self.roi_unet.predict(
            image, axes=axes, n_tiles=n_tiles, **kwargs
        ).semantic
        roi = roi.astype(bool)

        result = self.downstream.predict(image, axes=axes, n_tiles=n_tiles, **kwargs)

        labels = None if result.labels is None else np.where(roi, result.labels, 0)
        semantic = (
            None if result.semantic is None else (result.semantic.astype(bool) & roi)
        )

        return result.merge(labels=labels, semantic=semantic, roi=roi)
