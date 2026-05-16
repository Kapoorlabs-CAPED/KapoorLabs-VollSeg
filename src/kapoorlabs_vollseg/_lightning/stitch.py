"""Linear-blend tile stitcher — port of ``kapoorlabs_lightning.stitch_tiles``."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def stitch_tiles(
    predictions: Iterable[tuple],
    volume_shape: tuple[int, ...],
    overlap_fraction: float = 0.125,
) -> np.ndarray:
    """Blend predicted tiles back into a full volume of ``volume_shape``.

    Works for any spatial dimensionality — ``coord_batch`` is a
    ``(B, 2 * ndim)`` tensor laid out as
    ``[start_0, …, start_n, size_0, …, size_n]`` (the schema written by
    :class:`CarePredictionDataset`). Each tile in ``pred_batch`` has a
    leading channel axis (or none) plus ``ndim`` spatial axes; if a
    channel axis is present the *first* channel is taken.
    """
    output = np.zeros(volume_shape, dtype=np.float32)
    weight = np.zeros(volume_shape, dtype=np.float32)
    ndim = len(volume_shape)

    for pred_batch, coord_batch in predictions:
        for i in range(pred_batch.shape[0]):
            tile = (
                pred_batch[i].cpu().numpy()
                if hasattr(pred_batch[i], "cpu")
                else np.asarray(pred_batch[i])
            )
            # If the model emitted a leading channel axis, take channel 0.
            if tile.ndim == ndim + 1:
                tile = tile[0]
            coords = [int(v) for v in coord_batch[i].tolist()]
            starts = coords[:ndim]
            sizes = coords[ndim:]
            sl = tuple(slice(s, s + sz) for s, sz in zip(starts, sizes))
            w = _make_blend_weight(tuple(sizes), overlap_fraction)
            output[sl] += tile * w
            weight[sl] += w

    mask = weight > 0
    output[mask] /= weight[mask]
    return output


def _make_blend_weight(
    tile_shape: tuple[int, ...], overlap_fraction: float
) -> np.ndarray:
    weight = np.ones(tile_shape, dtype=np.float32)
    for axis, size in enumerate(tile_shape):
        overlap_px = max(1, int(size * overlap_fraction))
        ramp = np.linspace(0, 1, overlap_px, dtype=np.float32)
        w1d = np.ones(size, dtype=np.float32)
        w1d[:overlap_px] = ramp
        w1d[-overlap_px:] = ramp[::-1]
        shape = [1] * len(tile_shape)
        shape[axis] = size
        weight *= w1d.reshape(shape)
    return weight
