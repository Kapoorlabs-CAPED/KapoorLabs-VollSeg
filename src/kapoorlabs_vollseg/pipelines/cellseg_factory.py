"""Layer 3 factory for CellPose-based membrane pipelines.

Sibling of :class:`VollSeg`. Kept separate because the membrane pipeline
needs a *nuclei pipeline* as one of its inputs — that input is itself
typically built via ``VollSeg.from_models(...)`` — and lumping both
factories together would obscure the fact that membrane work *consumes*
nuclei work.
"""

from __future__ import annotations

from typing import Optional

from .base import Pipeline
from .chunked import Chunked
from .denoised import DenoisedPipeline
from .nuclei_cellpose import NucleiSeededCellPosePipeline


class VollCellSeg:
    """Factory namespace — not instantiated."""

    @staticmethod
    def from_models(
        *,
        nuclei_pipeline: Optional[Pipeline] = None,
        cellpose: Optional[Pipeline] = None,
        care: Optional[Pipeline] = None,
        nuclei_channel: Optional[int] = None,
        membrane_channel: Optional[int] = None,
        chunk: Optional[tuple[int, int, int]] = None,
        overlap: tuple[int, int, int] = (0, 0, 0),
    ) -> Pipeline:
        """Assemble a membrane pipeline from supplied models.

        Three valid shapes:

        - **CellPose only** (``cellpose=...`` alone) → returns the segmenter
          directly. Use this for plain membrane segmentation without
          nuclei seeding.
        - **CellPose + nuclei pipeline** → :class:`NucleiSeededCellPosePipeline`
          fuses the two via :func:`cellpose_watershed_fuse`.
        - Either of the above wrapped by ``care`` (denoise membrane first)
          and/or ``chunk`` (tile a huge volume).

        Parameters
        ----------
        nuclei_pipeline
            Any pipeline returning nuclei labels (typically built via
            :func:`VollSeg.from_models`).
        cellpose
            Required. The CellPose segmenter for the membrane channel.
        care
            Optional CARE denoiser, applied to the membrane channel before
            CellPose runs.
        nuclei_channel, membrane_channel
            Channel indices to slice from a multi-channel input. Required
            if the user calls ``pipeline.predict(image)`` on a CZYX/TCZYX
            volume; can be omitted if calling
            ``pipeline.predict_split(nuclei_image=, membrane_image=)``.
        chunk, overlap
            If ``chunk`` is set, wrap the result in :class:`Chunked`.
        """
        if cellpose is None:
            raise ValueError("VollCellSeg.from_models requires `cellpose=...`")

        if nuclei_pipeline is not None:
            inner: Pipeline = NucleiSeededCellPosePipeline(
                nuclei_pipeline,
                cellpose,
                nuclei_channel=nuclei_channel,
                membrane_channel=membrane_channel,
            )
        else:
            inner = cellpose

        if care is not None:
            inner = DenoisedPipeline(care, inner)
        if chunk is not None:
            inner = Chunked(inner, chunk=chunk, overlap=overlap)

        return inner
