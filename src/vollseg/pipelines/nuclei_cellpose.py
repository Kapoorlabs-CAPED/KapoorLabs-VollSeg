"""StarDist (or any nuclei pipeline) + CellPose composite.

Mirrors the role of :class:`UNetStarDistPipeline` but for nuclei + cell
membranes. Operates on a 4D ``(C, Z, Y, X)`` or 3D ``(Z, Y, X)`` membrane
image; the nuclei pipeline provides the seeds and the CellPose model
provides the constraining mask.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..fusion import cellpose_watershed_fuse
from .base import Pipeline, Result


class NucleiSeededCellPosePipeline:
    """Nuclei-seeded CellPose membrane segmentation.

    Composes:

    1. ``nuclei_pipeline`` → instance labels for nuclei (markers)
    2. ``cellpose`` → CellPose membrane mask (constraint)
    3. :func:`cellpose_watershed_fuse` → final cell labels

    Parameters
    ----------
    nuclei_pipeline
        Any pipeline returning ``Result.labels`` for the nuclei channel.
    cellpose
        :class:`CellPoseSegmenter` for the membrane channel.
    nuclei_channel, membrane_channel
        Channel indices to slice from a multi-channel input. Set both to
        ``None`` if you'll pass nuclei and membrane volumes separately
        via :meth:`predict_split`.
    """

    def __init__(
        self,
        nuclei_pipeline: Pipeline,
        cellpose: Pipeline,
        *,
        nuclei_channel: Optional[int] = None,
        membrane_channel: Optional[int] = None,
    ):
        if not isinstance(nuclei_pipeline, Pipeline):
            raise TypeError(
                f"nuclei_pipeline must be a Pipeline, got {type(nuclei_pipeline).__name__}"
            )
        if not isinstance(cellpose, Pipeline):
            raise TypeError(
                f"cellpose must be a Pipeline, got {type(cellpose).__name__}"
            )
        self.nuclei_pipeline = nuclei_pipeline
        self.cellpose = cellpose
        self.nuclei_channel = nuclei_channel
        self.membrane_channel = membrane_channel

    # -------------------------------------------------------------- API

    def predict(
        self,
        image: np.ndarray,
        *,
        axes: Optional[str] = None,
        n_tiles: Optional[tuple] = None,
        **kwargs,
    ) -> Result:
        nuclei_image, membrane_image = self._split(image)
        return self.predict_split(
            nuclei_image=nuclei_image,
            membrane_image=membrane_image,
            axes=axes,
            n_tiles=n_tiles,
            **kwargs,
        )

    def predict_split(
        self,
        *,
        nuclei_image: np.ndarray,
        membrane_image: np.ndarray,
        axes: Optional[str] = None,
        n_tiles: Optional[tuple] = None,
        **kwargs,
    ) -> Result:
        """Same as :meth:`predict` but takes the two channels as separate volumes."""
        nuclei_res = self.nuclei_pipeline.predict(
            nuclei_image, axes=axes, n_tiles=n_tiles, **kwargs
        )
        cellpose_res = self.cellpose.predict(membrane_image, axes=axes, **kwargs)

        cell_labels = cellpose_watershed_fuse(
            membrane_image,
            nuclei_labels=nuclei_res.labels,
            cellpose_mask=cellpose_res.labels > 0,
        )
        return Result(
            labels=cell_labels,
            semantic=cellpose_res.labels.astype(bool),
            polys=nuclei_res.polys,
            extra={"nuclei_labels": nuclei_res.labels},
        )

    # ----------------------------------------------------------- helpers

    def _split(self, image: np.ndarray):
        if self.nuclei_channel is None or self.membrane_channel is None:
            raise ValueError(
                "predict() needs both nuclei_channel and membrane_channel set on "
                "the pipeline; otherwise call predict_split(nuclei_image=, "
                "membrane_image=) directly."
            )
        # Channel axis is conventionally axis 1 for CZYX / TCZYX layouts.
        if image.ndim == 4:
            return image[self.nuclei_channel], image[self.membrane_channel]
        if image.ndim == 5:
            return (
                image[:, self.nuclei_channel],
                image[:, self.membrane_channel],
            )
        raise ValueError(f"Cannot split channels from ndim={image.ndim}")
