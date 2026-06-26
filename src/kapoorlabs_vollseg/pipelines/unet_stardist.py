"""StarDist + (optional) U-Net composite, with optional seedpool fusion.

Three behaviour modes, matching the original VollSeg pipeline shape:

1. ``unet`` + ``stardist`` + ``seedpool=True`` — classic VollSeg.
   Run both networks; fuse StarDist instances with the U-Net semantic
   mask via :func:`kapoorlabs_vollseg.fusion.watershed_fuse`.

2. ``stardist`` only + ``seedpool=True`` — no U-Net supplied.
   Auto-derive the seed-pool mask: Otsu-threshold the input, take it
   as the semantic mask (``watershed_fuse`` runs connected components
   internally to extract seed centroids). Same fusion as case 1.

3. ``unet`` + ``stardist`` + ``seedpool=False`` — no fusion.
   Run both side by side. ``Result.labels`` = StarDist instances,
   ``Result.semantic`` = U-Net mask.

The single-model branches (``stardist`` only with ``seedpool=False``,
``unet`` only) are routed to the bare singleton by the
:class:`VollSeg.from_models` factory; this class is only used when at
least one of the multi-model branches is needed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import label as cc_label
from skimage.filters import threshold_otsu

from ..fusion import watershed_fuse
from .base import Pipeline, Result


class UNetStarDistPipeline:
    """Composite of StarDist + optional U-Net with optional seedpool fusion.

    Parameters
    ----------
    unet
        Optional U-Net singleton. When ``None`` and ``seedpool=True``,
        the semantic mask is derived from an Otsu threshold of the
        input image (which is already denoised + ROI-cropped at this
        point in the composition).
    stardist
        StarDist singleton. Required — at least the instance detector
        must be present for either seedpool or side-by-side mode.
    seedpool
        When ``True``, fuse StarDist instances with the semantic mask
        via :func:`watershed_fuse`. When ``False``, return both
        outputs side by side without fusion.
    """

    def __init__(
        self,
        unet: Optional[Pipeline] = None,
        stardist: Optional[Pipeline] = None,
        *,
        seedpool: bool = False,
    ):
        if stardist is None:
            raise TypeError(
                "UNetStarDistPipeline requires `stardist`. "
                "Use the bare U-Net singleton directly for the U-Net-only case."
            )
        if not isinstance(stardist, Pipeline):
            raise TypeError(
                f"stardist must be a Pipeline, got {type(stardist).__name__}"
            )
        if unet is not None and not isinstance(unet, Pipeline):
            raise TypeError(f"unet must be a Pipeline, got {type(unet).__name__}")
        if unet is None and not seedpool:
            raise ValueError(
                "UNetStarDistPipeline with no `unet` is only meaningful when "
                "`seedpool=True` (the auto-derived threshold mask is the "
                "seed-pool source). Use the StarDist singleton directly."
            )
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
        # ── 1. StarDist instances on the (already-denoised, ROI-cropped)
        # input. This is the dense instance detector — its centroids are
        # the primary seeds for the watershed.
        star_res = self.stardist.predict(image, axes=axes, n_tiles=n_tiles, **kwargs)

        # ── 2. Semantic mask source. With a U-Net it's the U-Net binary
        # output. Without a U-Net (and only when ``seedpool=True``) it's
        # an Otsu-threshold of the input; ``watershed_fuse`` runs
        # ``label(mask)`` internally so connected-component extraction
        # is implicit. ``image`` is the denoised + ROI-cropped tensor at
        # this point in the composition (see VollSeg.from_models).
        # ``unet_labels`` is the CC-labelled U-Net semantic mask. We
        # only populate it when a U-Net actually ran — the no-U-Net
        # seed-pool path leaves it at ``None`` per the contract that
        # each stage output is surfaced only when its model is supplied.
        unet_labels = None
        if self.unet is not None:
            unet_res = self.unet.predict(image, axes=axes, n_tiles=n_tiles, **kwargs)
            semantic = unet_res.semantic
            probability = unet_res.probability
            if semantic is not None:
                cc, _ = cc_label(np.asarray(semantic).astype(bool))
                unet_labels = cc.astype(np.uint32)
        else:
            semantic = _threshold_mask(image)
            probability = semantic.astype(np.float32)

        # ── 3a. No seedpool: return both segmentations side by side.
        # StarDist labels are primary; U-Net mask + CC labels travel
        # along. ``stardist_labels`` and ``unet_labels`` are surfaced
        # explicitly so downstream code never has to guess what
        # ``labels`` came from.
        if not self.seedpool:
            return Result(
                labels=star_res.labels,
                stardist_labels=star_res.labels,
                unet_labels=unet_labels,
                semantic=semantic,
                probability=probability,
                polys=star_res.polys,
            )

        # ── 3b. Seedpool: watershed-fuse StarDist instances with the
        # semantic mask. Fused labels go into BOTH ``vollseg_labels``
        # (the named VollSeg output) and ``labels`` (the canonical
        # "use this" field). ``stardist_labels`` keeps the raw StarDist
        # instances; ``unet_labels`` is populated when (and only when)
        # a U-Net actually ran upstream.
        fused = watershed_fuse(
            image,
            stardist_labels=star_res.labels,
            unet_mask=semantic,
            seedpool=True,
        )
        return Result(
            labels=fused,
            vollseg_labels=fused,
            stardist_labels=star_res.labels,
            unet_labels=unet_labels,
            semantic=semantic,
            probability=probability,
            polys=star_res.polys,
        )


def _threshold_mask(image: np.ndarray) -> np.ndarray:
    """Otsu threshold for the no-U-Net seed-pool path.

    Returns a boolean mask the same shape as ``image``. Connected
    components are extracted later by ``watershed_fuse`` via ``label(mask)``
    when it walks the seed-pool centroids. Returns an all-False mask on
    degenerate (e.g. all-zero) input.
    """
    try:
        t = float(threshold_otsu(image))
    except ValueError:
        return np.zeros(image.shape, dtype=bool)
    return image > t
