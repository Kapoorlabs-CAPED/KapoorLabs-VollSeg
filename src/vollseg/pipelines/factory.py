"""Layer 3 — the smart factory that assembles a pipeline from supplied models.

This is the spiritual successor to the original ``utils.VollSeg`` if/else
router, but with one rule: the pipeline shape is fixed at construction
time, not chosen mid-prediction. Invalid combinations raise here, not deep
inside ``.predict``.
"""

from __future__ import annotations

from typing import Optional, Tuple

from ..models.care import CAREDenoiser
from ..models.stardist import StarDistSegmenter
from ..models.unet import UNetSegmenter
from .base import Pipeline
from .chunked import Chunked
from .denoised import DenoisedPipeline
from .roi import ROIPipeline
from .unet_stardist import UNetStarDistPipeline


class VollSeg:
    """Factory namespace — not instantiated."""

    @staticmethod
    def from_models(
        *,
        care: Optional[CAREDenoiser] = None,
        unet: Optional[UNetSegmenter] = None,
        stardist: Optional[StarDistSegmenter] = None,
        roi_unet: Optional[UNetSegmenter] = None,
        seedpool: bool = False,
        chunk: Optional[Tuple[int, int, int]] = None,
        overlap: Tuple[int, int, int] = (0, 0, 0),
    ) -> Pipeline:
        """Assemble a pipeline matching the supplied model set.

        Parameters
        ----------
        care
            Optional CARE denoiser; wraps the rest in :class:`DenoisedPipeline`.
        unet, stardist
            Provide either, both, or neither (must provide at least one of
            these or ``care``). Both → :class:`UNetStarDistPipeline`.
        roi_unet
            Optional ROI U-Net; wraps the rest in :class:`ROIPipeline`.
        seedpool
            Only meaningful when both ``unet`` and ``stardist`` are given;
            enables watershed fusion.
        chunk, overlap
            If ``chunk`` is set, wrap the whole thing in :class:`Chunked`
            with this chunk shape and overlap.

        Returns
        -------
        Pipeline
            A composed pipeline ready for ``.predict(image)``.
        """
        if seedpool and not (unet is not None and stardist is not None):
            raise ValueError("seedpool=True requires both `unet` and `stardist`.")
        if all(m is None for m in (care, unet, stardist)):
            raise ValueError("Provide at least one of `care`, `unet`, or `stardist`.")

        # Inner segmenter
        if unet is not None and stardist is not None:
            inner: Pipeline = UNetStarDistPipeline(unet, stardist, seedpool=seedpool)
        elif stardist is not None:
            inner = stardist
        elif unet is not None:
            inner = unet
        else:
            inner = care  # care-only: degenerate "denoise as the whole pipeline"
            care = None   # don't double-wrap below

        # Decorators, outer-most last so that user-visible order matches the
        # README diagram: chunk(roi(denoised(inner)))
        if care is not None:
            inner = DenoisedPipeline(care, inner)
        if roi_unet is not None:
            inner = ROIPipeline(roi_unet, inner)
        if chunk is not None:
            inner = Chunked(inner, chunk=chunk, overlap=overlap)

        return inner
