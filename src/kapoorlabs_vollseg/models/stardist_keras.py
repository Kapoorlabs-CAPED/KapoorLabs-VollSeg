"""StarDist singleton (keras / stardist) — current first-class for instance seg.

A PyTorch StarDist is planned; until then this remains the canonical
StarDist class. The ``Keras`` suffix is kept for parity with the other
keras backbones, but no PyTorch ``StarDistSegmenter`` exists yet — code
that needs StarDist should import :class:`StarDistSegmenterKeras`
directly.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .._backbones.stardist_keras import StarDist2DBackboneKeras, StarDist3DBackboneKeras
from ..pipelines.base import Result, infer_axes


_StarDistAny = Union[StarDist2DBackboneKeras, StarDist3DBackboneKeras]


class StarDistSegmenterKeras:
    """Run a StarDist model and return instance labels + polygon details."""

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
    def from_pretrained(
        cls, name_or_alias: str, *, ndim: int = 3, **kwargs
    ) -> StarDistSegmenterKeras:
        from ..pretrained import get_model_instance

        bb_cls = StarDist3DBackboneKeras if ndim == 3 else StarDist2DBackboneKeras
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
