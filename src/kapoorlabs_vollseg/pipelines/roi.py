"""Decorator pipeline: crop image to a ROI, run downstream, paste back.

Matches the original ``VollSeg.utils.VollSeg2D`` / ``VollSeg_unet`` flow:
the ROI Mask-UNet predicts a binary mask on the full image, we take the
mask's bounding box in the spatial plane (YX for both 2D-on-3D and 2D
inputs; ZYX when the ROI itself is 3D), crop the image to that bbox,
hand the **cropped patch** to the downstream pipeline, then paste its
labels back into a full-shape array at the bbox position. Labels
outside the ROI are 0 by construction.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Pipeline, Result


def _spatial_bbox(roi: np.ndarray) -> Optional[tuple[slice, ...]]:
    """Tightest axis-aligned bbox of ``roi`` as a tuple of slices, or
    ``None`` if the mask is empty. For a 3D mask that's actually a 2D-
    broadcast (constant in Z), Z is returned as the full slice — Z stays
    untouched, only YX gets cropped."""
    if not roi.any():
        return None
    bbox = []
    for axis in range(roi.ndim):
        idx = np.where(roi.any(axis=tuple(j for j in range(roi.ndim) if j != axis)))[0]
        # If every slice along this axis carries ROI, we don't need to
        # crop it — keep the axis full so we don't pay a copy.
        if idx.size == roi.shape[axis]:
            bbox.append(slice(None))
        else:
            bbox.append(slice(int(idx.min()), int(idx.max()) + 1))
    return tuple(bbox)


class ROIPipeline:
    """Crop to a Mask-UNet ROI, run downstream on the patch, restore.

    The ROI Mask-UNet runs once on the full image to produce its binary
    mask (typically 2D-broadcast-to-3D for the Xenopus ROI model — see
    :class:`MaskUNetSegmenter`). We then compute the spatial bbox of
    that mask, crop the image, dispatch the cropped patch to the
    downstream pipeline, and paste the result back into a full-shape
    array. Labels outside the ROI are 0 by construction. Empty ROI →
    everything-zero result.
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
        roi_res = self.roi_unet.predict(image, axes=axes, n_tiles=n_tiles, **kwargs)
        roi = roi_res.semantic
        if roi is None:  # MaskUNet returns labels only — fall back.
            roi = roi_res.labels
        if roi is None:
            raise ValueError(
                "ROI model produced no mask (semantic / labels both None)."
            )
        roi = np.asarray(roi) > 0

        bbox = _spatial_bbox(roi)
        if bbox is None:
            # Empty ROI — return a zero-everywhere result.
            return Result(
                labels=np.zeros(image.shape, dtype=np.uint32),
                semantic=np.zeros(image.shape, dtype=bool),
                roi=roi,
            )

        patch = image[bbox]
        result = self.downstream.predict(patch, axes=axes, n_tiles=n_tiles, **kwargs)

        def _restore(field, dtype):
            if field is None:
                return None
            full = np.zeros(image.shape, dtype=dtype)
            full[bbox] = field.astype(dtype)
            # Mask off anything the downstream wrote *outside* the
            # bbox-cropped ROI (defensive — should be a no-op).
            full[~roi] = 0
            return full

        labels = _restore(result.labels, np.uint32)
        semantic = _restore(
            (result.semantic.astype(bool) if result.semantic is not None else None),
            bool,
        )
        probability = (
            None
            if result.probability is None
            else _restore(result.probability, np.float32)
        )

        return result.merge(
            labels=labels,
            semantic=semantic,
            probability=probability,
            roi=roi,
        )
