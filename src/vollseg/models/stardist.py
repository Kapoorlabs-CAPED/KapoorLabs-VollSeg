"""StarDist instance segmentation singleton.

Dispatches to a 2D or 3D backbone based on ``image.ndim`` — there is no
parallel ``StarDist2DSegmenter`` / ``StarDist3DSegmenter`` class tree.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .._backbones import StarDist2DBackbone, StarDist3DBackbone
from ..pipelines.base import Result, infer_axes


_StarDistAny = Union[StarDist2DBackbone, StarDist3DBackbone]


class StarDistSegmenter:
    """Run a StarDist model and return instance labels + polygon details.

    Parameters
    ----------
    backbone
        A trained :class:`StarDist2DBackbone` or :class:`StarDist3DBackbone`.
        The class is inspected at construction to pick the right ``predict_*``
        call later — there is no runtime dispatch cost.
    prob_thresh, nms_thresh
        Defaults forwarded to ``predict_instances`` (per-call kwargs override).
    """

    def __init__(
        self,
        backbone: _StarDistAny,
        *,
        prob_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
    ):
        self.backbone = backbone
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh

    @classmethod
    def from_pretrained(cls, name_or_alias: str, *, ndim: int = 3, **kwargs) -> "StarDistSegmenter":
        from ..pretrained import get_model_instance
        bb_cls = StarDist3DBackbone if ndim == 3 else StarDist2DBackbone
        return cls(get_model_instance(bb_cls, name_or_alias), **kwargs)

    def predict(
        self,
        image: np.ndarray,
        *,
        axes: Optional[str] = None,
        n_tiles: Optional[tuple] = None,
        prob_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
        **_ignored,
    ) -> Result:
        if axes is None:
            axes = infer_axes(image)
        labels, details = self.backbone.predict_instances(
            image.astype("float32"),
            axes=axes,
            n_tiles=n_tiles,
            prob_thresh=prob_thresh if prob_thresh is not None else self.prob_thresh,
            nms_thresh=nms_thresh if nms_thresh is not None else self.nms_thresh,
        )
        return Result(labels=labels, polys=details)
