"""Layer 3 — the smart factory that assembles a pipeline from supplied models.

This is the spiritual successor to the original ``utils.VollSeg`` if/else
router, but with one rule: the pipeline shape is fixed at construction
time, not chosen mid-prediction. Invalid combinations raise here, not deep
inside ``.predict``.
"""

from __future__ import annotations

from typing import Optional

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
        care: Optional[Pipeline] = None,
        unet: Optional[Pipeline] = None,
        stardist: Optional[Pipeline] = None,
        roi_unet: Optional[Pipeline] = None,
        seedpool: bool = False,
        chunk: Optional[tuple[int, int, int]] = None,
        overlap: tuple[int, int, int] = (0, 0, 0),
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
            When passed alone (no ``care`` / ``unet`` / ``stardist``)
            the factory returns the ROI singleton directly — the
            output is the ROI mask itself, useful for inspection.
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
        if all(m is None for m in (care, unet, stardist, roi_unet)):
            raise ValueError(
                "Provide at least one of `care`, `unet`, `stardist`, or `roi_unet`."
            )

        # Inner segmenter
        if unet is not None and stardist is not None:
            inner: Pipeline = UNetStarDistPipeline(unet, stardist, seedpool=seedpool)
        elif stardist is not None:
            inner = stardist
        elif unet is not None:
            inner = unet
        elif care is not None:
            inner = care  # care-only: degenerate "denoise as the whole pipeline"
            care = None  # don't double-wrap below
        else:
            # ROI-only: just return the ROI mask itself. No downstream
            # to gate, so don't wrap in ROIPipeline.
            inner = roi_unet
            roi_unet = None

        # Decorators, outer-most last so that user-visible order matches the
        # README diagram: chunk(roi(denoised(inner)))
        if care is not None:
            inner = DenoisedPipeline(care, inner)
        if roi_unet is not None:
            inner = ROIPipeline(roi_unet, inner)
        if chunk is not None:
            inner = Chunked(inner, chunk=chunk, overlap=overlap)

        return inner
