"""Tests for kapoorlabs_vollseg._lightning.dataset.CarePredictionDataset and stitch_tiles.

Skipped if torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# importorskip must run first; downstream imports may need torch transitively.
from kapoorlabs_vollseg._lightning.dataset import (  # noqa: E402
    CarePredictionDataset,
    compute_tile_shape,
)
from kapoorlabs_vollseg._lightning.stitch import stitch_tiles  # noqa: E402


class TestComputeTileShape:
    def test_basic(self):
        # 100 / 4 = 25 → ceil(100/4) = 25
        assert compute_tile_shape((100, 200, 300), (4, 4, 4)) == (25, 50, 75)


class TestCarePredictionDataset:
    def test_tile_count_no_overlap(self):
        vol = np.zeros((20, 20, 20), dtype=np.float32)
        ds = CarePredictionDataset(vol, tile_shape=(10, 10, 10), overlap=0.0)
        # 20/10 = 2 tiles per axis → 8 tiles total.
        assert len(ds) == 8

    def test_returns_tile_and_coords(self):
        vol = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
        ds = CarePredictionDataset(vol, tile_shape=(4, 4, 4), overlap=0.0)
        tile, coords = ds[0]
        assert tile.shape == (4, 4, 4)
        assert coords.shape == (6,)

    def test_tile_covers_volume(self):
        # Confirm every voxel of the volume is covered by at least one tile.
        vol = np.arange(16 * 16, dtype=np.float32).reshape(1, 16, 16)
        # Force 3D dataset — pad the leading axis up to 4 for tile_shape=(4,8,8).
        vol3 = np.broadcast_to(vol, (4, 16, 16)).copy()
        ds = CarePredictionDataset(vol3, tile_shape=(4, 8, 8), overlap=0.25)
        coverage = np.zeros(vol3.shape, dtype=bool)
        for i in range(len(ds)):
            _, coords = ds[i]
            zs, ys, xs, tz, ty, tx = (int(v) for v in coords.tolist())
            coverage[zs : zs + tz, ys : ys + ty, xs : xs + tx] = True
        assert coverage.all()


class TestStitchTiles:
    def test_round_trip_constant_volume(self):
        # Predict a constant tile per location → stitching gives the same constant.
        vol = np.zeros((4, 16, 16), dtype=np.float32)
        ds = CarePredictionDataset(vol, tile_shape=(4, 8, 8), overlap=0.0)
        # Build a single batch with all tiles.
        tiles, coords = [], []
        for i in range(len(ds)):
            t, c = ds[i]
            tiles.append(torch.full_like(t, fill_value=3.0))
            coords.append(c)
        pred = (torch.stack(tiles, dim=0), torch.stack(coords, dim=0))
        out = stitch_tiles([pred], vol.shape, overlap_fraction=0.0)
        np.testing.assert_allclose(out, np.full_like(vol, 3.0), rtol=1e-5)

    def test_shape_preserved(self):
        vol = np.zeros((4, 16, 16), dtype=np.float32)
        ds = CarePredictionDataset(vol, tile_shape=(4, 8, 8), overlap=0.125)
        tiles, coords = [], []
        for i in range(len(ds)):
            t, c = ds[i]
            tiles.append(t * 0.0 + i)  # gradient by tile index
            coords.append(c)
        pred = (torch.stack(tiles, dim=0), torch.stack(coords, dim=0))
        out = stitch_tiles([pred], vol.shape, overlap_fraction=0.125)
        assert out.shape == vol.shape
        assert out.dtype == np.float32
