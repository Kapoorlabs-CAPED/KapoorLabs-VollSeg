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
    """Yield ``(tile_tensor, coords)`` for each tile of a 3D volume."""

    def __init__(
        self,
        volume: np.ndarray,
        tile_shape: tuple[int, int, int],
        overlap: float = 0.125,
        normalizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        if volume.ndim != 3:
            raise ValueError(
                f"CarePredictionDataset expects 3D volume, got ndim={volume.ndim}"
            )
        self.volume = volume
        self.tile_shape = tuple(tile_shape)
        self.overlap = float(overlap)
        self.normalizer = normalizer
        self.tiles = self._compute_tile_grid()

    def _compute_tile_grid(self):
        tiles = []
        for axis_idx, (vol_size, tile_size) in enumerate(
            zip(self.volume.shape, self.tile_shape)
        ):
            overlap_px = int(tile_size * self.overlap)
            stride = max(1, tile_size - overlap_px)
            starts = []
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
        zs, ys, xs = self.tiles[idx]
        tz, ty, tx = self.tile_shape
        tile = self.volume[zs : zs + tz, ys : ys + ty, xs : xs + tx]
        tile = torch.from_numpy(np.ascontiguousarray(tile, dtype=np.float32))
        if self.normalizer is not None:
            tile = self.normalizer(tile)
        coords = torch.tensor([zs, ys, xs, tz, ty, tx], dtype=torch.long)
        return tile, coords


def compute_tile_shape(
    volume_shape: tuple[int, int, int], n_tiles: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Pick a per-axis tile size that splits ``volume_shape`` into ~``n_tiles`` chunks."""
    return tuple(int(np.ceil(v / n)) for v, n in zip(volume_shape, n_tiles))
