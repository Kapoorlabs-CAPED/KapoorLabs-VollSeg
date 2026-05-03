"""CellPose Layer 1 singleton — wraps a CellPose backbone for inference.

Dispatches over ``image.ndim`` and an optional ``has_time`` hint to handle
3D volumes, 4D ``TZYX`` timelapses, and 3D ``TYX`` 2D-timelapses with the
same API. No parallel ``CellPose2DSegmenter`` / ``CellPose3DSegmenter``
class tree.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .._backbones.cellpose import CellPoseBackbone
from ..pipelines.base import Result


class CellPoseSegmenter:
    """Run a CellPose model. Returns instance labels in ``Result.labels``.

    Parameters
    ----------
    backbone
        A constructed :class:`CellPoseBackbone`.
    diameter
        CellPose diameter in pixels for the trained scale.
    flow_threshold, cellprob_threshold, stitch_threshold
        Standard CellPose thresholds; defaults match the original VollSeg.
    anisotropy
        Per-axis voxel anisotropy for 3D inference; ``None`` disables.
    channels
        ``[cyto, nuclei]`` channel selection forwarded to CellPose.
    bsize
        Tile size for CellPose's internal tiling.
    """

    def __init__(
        self,
        backbone: CellPoseBackbone,
        *,
        diameter: float = 34.6,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        stitch_threshold: float = 0.5,
        anisotropy: Optional[float] = None,
        channels: Sequence[int] = (0, 0),
        bsize: int = 224,
    ):
        self.backbone = backbone
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.stitch_threshold = stitch_threshold
        self.anisotropy = anisotropy
        self.channels = list(channels)
        self.bsize = bsize

    def predict(
        self,
        image: np.ndarray,
        *,
        axes: Optional[str] = None,
        has_time: bool = False,
        do_3D: bool = False,
        **_ignored,
    ) -> Result:
        """Run CellPose on ``image``.

        Set ``has_time=True`` (or include ``T`` in ``axes``) for a TZYX or
        TYX timelapse — each frame is segmented independently.
        """
        if axes is not None and "T" in axes:
            has_time = True

        if has_time:
            labels_per_frame = [
                self._eval_one(frame, do_3D=do_3D)
                for frame in tqdm(image, desc="cellpose timelapse")
            ]
            labels = np.stack(labels_per_frame, axis=0)
        else:
            labels = self._eval_one(image, do_3D=do_3D)

        return Result(labels=labels.astype(np.uint16))

    # --------------------------------------------------------- internals

    def _eval_one(self, frame: np.ndarray, *, do_3D: bool) -> np.ndarray:
        kwargs = dict(
            diameter=self.diameter,
            channels=self.channels,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            stitch_threshold=self.stitch_threshold,
            bsize=self.bsize,
            do_3D=do_3D,
        )
        if self.anisotropy is not None:
            kwargs["anisotropy"] = self.anisotropy

        labels, *_ = self.backbone.model.eval(frame, **kwargs)
        return np.asarray(labels)
