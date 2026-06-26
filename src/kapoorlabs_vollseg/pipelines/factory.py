"""Layer 3 — the smart factory that assembles a pipeline from supplied models.

Pipeline shape is fixed at construction; invalid combinations raise here,
not deep inside ``.predict``.
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

        Composition order (innermost out):

        ::

            chunk( denoised( roi( segmentation ) ) )

        — i.e. at predict time: chunk the input, then denoise each chunk
        in full, then run the ROI Mask-UNet on the **denoised** chunk,
        then crop to the ROI bbox and segment that crop.

        Segmentation core (the innermost stage) is chosen as follows:

        - ``seedpool=True`` + ``stardist`` + ``unet`` →
          :class:`UNetStarDistPipeline`: run both networks, fuse via
          watershed seed-pool.
        - ``seedpool=True`` + ``stardist`` + ``care`` (no ``unet``) →
          :class:`UNetStarDistPipeline` with no U-Net: the seed-pool
          mask is auto-derived from an Otsu threshold of the denoised
          image; same fusion.
        - ``seedpool=False`` + ``unet`` + ``stardist`` →
          :class:`UNetStarDistPipeline`: run both, no fusion.
        - Single-model: bare singleton.
        - ``care`` only / ``roi_unet`` only: bare singleton too.

        Permissive composition rules (the only exception is "no model
        at all"):

        - ``seedpool=True`` is **silently ignored** when its
          prerequisites aren't met: no ``stardist`` (nothing to fuse),
          or no ``unet`` AND no ``care`` (no semantic mask source).
          In those cases the factory falls back to the next-best
          shape — usually the bare StarDist singleton.
        - Any single-model configuration returns the bare singleton.
          Composition only kicks in when there's actually something
          to compose.

        Parameters
        ----------
        care
            CARE denoiser. Optional. When present, wraps the rest in
            :class:`DenoisedPipeline`.
        unet, stardist
            Segmentation singletons. Provide at least one; either or
            both compose with ``seedpool``.
        roi_unet
            ROI Mask-UNet. When present, wraps the segmentation in
            :class:`ROIPipeline` so percentile-normalisation +
            segmentation happen on the ROI bbox crop, not the full
            (mostly-empty) volume.
        seedpool
            Enables the VollSeg watershed seed-pool fusion. Requires
            ``stardist``; ``unet`` is optional and replaced by an Otsu
            threshold of the input when absent.
        chunk, overlap
            When ``chunk`` is set, wrap the whole composition in
            :class:`Chunked` with this chunk shape and overlap.

        Returns
        -------
        Pipeline
            A composed pipeline ready for ``.predict(image)``.
        """
        # Only failure mode: literally no model. Everything else
        # composes whatever is available.
        if all(m is None for m in (care, unet, stardist, roi_unet)):
            raise ValueError(
                "Provide at least one of `care`, `unet`, `stardist`, or `roi_unet`."
            )

        # Silently ignore ``seedpool`` when its prerequisites aren't
        # met. Two cases:
        #   1. No ``stardist`` — nothing to fuse (the watershed needs
        #      instance markers from StarDist).
        #   2. No ``unet`` AND no ``care`` — no semantic mask source.
        #      ``care`` enables the Otsu-threshold-of-the-denoised-
        #      image fallback; without either, there's no meaningful
        #      mask to drive the seed pool.
        if seedpool and (stardist is None or (unet is None and care is None)):
            seedpool = False

        # ── Segmentation core ───────────────────────────────────────
        # Two-model fusion or side-by-side composite.
        if seedpool or (unet is not None and stardist is not None):
            inner: Pipeline = UNetStarDistPipeline(
                unet=unet,
                stardist=stardist,
                seedpool=seedpool,
            )
        # Single-model branches.
        elif stardist is not None:
            inner = stardist
        elif unet is not None:
            inner = unet
        # Degenerate single-stage modes: only care, or only roi_unet.
        elif care is not None:
            inner = care
            care = None  # don't double-wrap below
        else:
            inner = roi_unet
            roi_unet = None

        # ── Decorators, outer-most last ─────────────────────────────
        # Effective predict-time order: chunk → denoise full input →
        # ROI on the denoised input → crop denoised → segment crop.
        # That matches the original VollSeg flow: denoising happens
        # first so the ROI Mask-UNet and the downstream segmenters
        # all see the same cleaned-up image.
        if roi_unet is not None:
            inner = ROIPipeline(roi_unet, inner)
        if care is not None:
            inner = DenoisedPipeline(care, inner)
        if chunk is not None:
            inner = Chunked(inner, chunk=chunk, overlap=overlap)

        return inner
