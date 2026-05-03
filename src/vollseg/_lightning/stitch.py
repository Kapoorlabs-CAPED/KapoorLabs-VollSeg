"""Linear-blend tile stitcher — port of ``kapoorlabs_lightning.stitch_tiles``."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def stitch_tiles(
    predictions: Iterable[Tuple],
    volume_shape: Tuple[int, int, int],
    overlap_fraction: float = 0.125,
) -> np.ndarray:
    """Blend predicted tiles back into a full ``(Z, Y, X)`` volume.

    ``predictions`` is whatever ``CareModule.predict_step`` produces — an
    iterable of ``(tile_batch, coord_batch)`` pairs where ``tile_batch``
    has shape ``(B, Z, Y, X)`` and ``coord_batch`` has shape ``(B, 6)``
    holding ``[zs, ys, xs, tz, ty, tx]`` per tile.
    """
    output = np.zeros(volume_shape, dtype=np.float32)
    weight = np.zeros(volume_shape, dtype=np.float32)

    for pred_batch, coord_batch in predictions:
        for i in range(pred_batch.shape[0]):
            tile = pred_batch[i].cpu().numpy() if hasattr(pred_batch[i], "cpu") else np.asarray(pred_batch[i])
            zs, ys, xs, tz, ty, tx = (int(v) for v in coord_batch[i].tolist())
            w = _make_blend_weight((tz, ty, tx), overlap_fraction)
            output[zs:zs + tz, ys:ys + ty, xs:xs + tx] += tile * w
            weight[zs:zs + tz, ys:ys + ty, xs:xs + tx] += w

    mask = weight > 0
    output[mask] /= weight[mask]
    return output


def _make_blend_weight(tile_shape: Tuple[int, ...], overlap_fraction: float) -> np.ndarray:
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
