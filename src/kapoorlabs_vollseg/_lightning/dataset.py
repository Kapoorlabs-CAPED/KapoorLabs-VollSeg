"""Tile a 3D volume for batched prediction (PyTorch Dataset).

Port of ``kapoorlabs_lightning.care_dataset.CarePredictionDataset`` — same
tile/overlap math, same coord schema, so stitching produced by
:func:`kapoorlabs_vollseg._lightning.stitch_tiles` matches what the upstream package
would have produced from the same inputs.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class CarePredictionDataset(Dataset):
    """Yield ``(tile_tensor, coords)`` for each tile of a 2D or 3D volume.

    ``coords`` is a ``2 * ndim``-long tensor laid out as
    ``[start_axis_0, …, start_axis_n, size_axis_0, …, size_axis_n]`` —
    the same schema :func:`stitch_tiles` consumes for both 2D and 3D.
    """

    def __init__(
        self,
        volume: np.ndarray,
        tile_shape: tuple[int, ...],
        overlap: float = 0.125,
        normalizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        if volume.ndim not in (2, 3):
            raise ValueError(
                f"CarePredictionDataset expects a 2D or 3D volume, "
                f"got ndim={volume.ndim}"
            )
        if len(tile_shape) != volume.ndim:
            raise ValueError(
                f"tile_shape {tile_shape} has {len(tile_shape)} entries "
                f"but volume.ndim={volume.ndim}"
            )
        self.volume = volume
        self.tile_shape = tuple(tile_shape)
        self.overlap = float(overlap)
        self.normalizer = normalizer
        self.tiles = self._compute_tile_grid()

    def _compute_tile_grid(self):
        tiles: list[tuple[int, ...]] = []
        for axis_idx, (vol_size, tile_size) in enumerate(
            zip(self.volume.shape, self.tile_shape)
        ):
            overlap_px = int(tile_size * self.overlap)
            stride = max(1, tile_size - overlap_px)
            starts: list[int] = []
            pos = 0
            while pos + tile_size <= vol_size:
                starts.append(pos)
                pos += stride
            if not starts or starts[-1] + tile_size < vol_size:
                starts.append(max(0, vol_size - tile_size))
            tiles = (
                [(s,) for s in starts]
                if axis_idx == 0
                else [t + (s,) for t in tiles for s in starts]
            )
        return tiles

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int):
        starts = self.tiles[idx]
        slices = tuple(slice(s, s + sz) for s, sz in zip(starts, self.tile_shape))
        tile = self.volume[slices]
        tile = torch.from_numpy(np.ascontiguousarray(tile, dtype=np.float32))
        if self.normalizer is not None:
            tile = self.normalizer(tile)
        coords = torch.tensor(
            list(starts) + list(self.tile_shape),
            dtype=torch.long,
        )
        return tile, coords


def compute_tile_shape(
    volume_shape: tuple[int, ...], n_tiles: tuple[int, ...]
) -> tuple[int, ...]:
    """Pick a per-axis tile size that splits ``volume_shape`` into ~``n_tiles`` chunks.

    Works for any matching number of axes (2D, 3D, …).
    """
    if len(volume_shape) != len(n_tiles):
        raise ValueError(
            f"volume_shape {volume_shape} and n_tiles {n_tiles} must have "
            f"the same length"
        )
    return tuple(int(np.ceil(v / n)) for v, n in zip(volume_shape, n_tiles))
