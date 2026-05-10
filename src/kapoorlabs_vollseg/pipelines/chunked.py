"""Decorator pipeline: run any pipeline on overlapping chunks of a 3D volume.

Stitches instance label outputs by remapping each chunk's label IDs into a
running global namespace, and writes only into still-empty regions of the
output volume (so the central, full-context part of each chunk wins over
the half-overlap margins of its neighbors).
"""

from __future__ import annotations

import gc
from typing import Optional

import numpy as np
from tqdm import tqdm

from .base import Pipeline, Result


class Chunked:
    """Run a downstream pipeline tile-by-tile on a large 3D volume.

    Parameters
    ----------
    downstream
        Any pipeline that accepts and returns 3D arrays.
    chunk
        Per-axis chunk shape ``(Z, Y, X)``.
    overlap
        Per-axis overlap between adjacent chunks ``(Z, Y, X)``. Half of this
        margin is cropped from each interior chunk before stitching to mute
        boundary artefacts.
    """

    def __init__(
        self,
        downstream: Pipeline,
        *,
        chunk: tuple[int, int, int],
        overlap: tuple[int, int, int] = (0, 0, 0),
    ):
        if not isinstance(downstream, Pipeline):
            raise TypeError(
                f"downstream must be a Pipeline, got {type(downstream).__name__}"
            )
        if any(o >= c for c, o in zip(chunk, overlap)):
            raise ValueError(
                f"overlap {overlap} must be smaller than chunk {chunk} on every axis"
            )
        self.downstream = downstream
        self.chunk = chunk
        self.overlap = overlap

    def predict(
        self,
        image: np.ndarray,
        *,
        axes: Optional[str] = None,
        n_tiles: Optional[tuple] = None,
        **kwargs,
    ) -> Result:
        if image.ndim != 3:
            raise ValueError(f"Chunked expects a 3D volume, got ndim={image.ndim}")

        slices = list(_iter_chunk_slices(image.shape, self.chunk, self.overlap))
        stitched = np.zeros(image.shape, dtype=np.uint32)
        max_label = 0

        for sl in tqdm(slices, desc="Chunked predict"):
            sub = np.asarray(image[sl])
            res = self.downstream.predict(sub, axes=axes, n_tiles=n_tiles, **kwargs)
            if res.labels is None:
                continue
            max_label = _stitch(stitched, res.labels, sl, self.overlap, max_label)
            del sub
            gc.collect()

        return Result(labels=stitched)


def _iter_chunk_slices(shape, chunk, overlap):
    steps = [c - o for c, o in zip(chunk, overlap)]
    starts = [list(range(0, s, st)) for s, st in zip(shape, steps)]
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                yield (
                    slice(z, min(z + chunk[0], shape[0])),
                    slice(y, min(y + chunk[1], shape[1])),
                    slice(x, min(x + chunk[2], shape[2])),
                )


def _stitch(stitched, chunk_labels, slices, overlap, max_label):
    if chunk_labels.max() == 0:
        return max_label

    renumbered = chunk_labels.copy()
    mask = chunk_labels > 0
    renumbered[mask] = chunk_labels[mask] + max_label

    crops = []
    targets = []
    for axis, (sl, ov, dim, full_dim) in enumerate(
        zip(slices, overlap, chunk_labels.shape, stitched.shape)
    ):
        lo = ov // 2 if sl.start > 0 else 0
        hi = dim - ov // 2 if sl.stop < full_dim else dim
        crops.append(slice(lo, hi))
        targets.append(slice(sl.start + lo, sl.start + hi))

    cropped = renumbered[tuple(crops)]
    target_region = stitched[tuple(targets)]
    placement = (target_region == 0) & (cropped > 0)
    target_region[placement] = cropped[placement]
    stitched[tuple(targets)] = target_region

    return int(stitched.max())
