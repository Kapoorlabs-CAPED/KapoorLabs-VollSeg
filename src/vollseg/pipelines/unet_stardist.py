"""StarDist + U-Net composite, with optional seedpool fusion."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..fusion import watershed_fuse
from ..models.stardist import StarDistSegmenter
from ..models.unet import UNetSegmenter
from .base import Pipeline, Result


class UNetStarDistPipeline:
    """Run a U-Net and a StarDist model together.

    With ``seedpool=False`` the two segmentations are simply returned side
    by side: ``result.labels`` is the StarDist instances, ``result.semantic``
    is the U-Net binary mask.

    With ``seedpool=True`` the two are fused via a marker-controlled
    watershed (see :func:`vollseg.fusion.watershed_fuse`): StarDist
    centroids and U-Net seeds outside every StarDist box become markers,
    constrained by the U-Net mask.
    """

    def __init__(
        self,
        unet: UNetSegmenter,
        stardist: StarDistSegmenter,
        *,
        seedpool: bool = False,
    ):
        if not isinstance(unet, Pipeline):
            raise TypeError(f"unet must be a Pipeline, got {type(unet).__name__}")
        if not isinstance(stardist, Pipeline):
            raise TypeError(f"stardist must be a Pipeline, got {type(stardist).__name__}")
        self.unet = unet
        self.stardist = stardist
        self.seedpool = seedpool

    def predict(
        self,
        image: np.ndarray,
        *,
        axes: Optional[str] = None,
        n_tiles: Optional[tuple] = None,
        **kwargs,
    ) -> Result:
        unet_res = self.unet.predict(image, axes=axes, n_tiles=n_tiles, **kwargs)
        star_res = self.stardist.predict(image, axes=axes, n_tiles=n_tiles, **kwargs)

        if self.seedpool:
            fused = watershed_fuse(
                image,
                stardist_labels=star_res.labels,
                unet_mask=unet_res.semantic,
                seedpool=True,
            )
            labels = fused
        else:
            labels = star_res.labels

        return Result(
            labels=labels,
            semantic=unet_res.semantic,
            probability=unet_res.probability,
            polys=star_res.polys,
        )
