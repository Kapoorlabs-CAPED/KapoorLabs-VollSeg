"""Tests for vollseg.seedpool.SeedPool / UnetStarMask geometry primitives."""

from __future__ import annotations

import pytest

from vollseg import SeedPool, UnetStarMask


class TestSeedPool2D:
    def test_point_inside_box(self):
        # bbox (ymin, xmin, ymax, xmax)
        box = (10, 10, 20, 20)
        # SeedPool.pooling() returns True if point is OUTSIDE the box on
        # at least one axis — i.e. NOT contained.
        assert SeedPool(box, (15, 15)).pooling() is False

    def test_point_outside_box(self):
        box = (10, 10, 20, 20)
        assert SeedPool(box, (5, 15)).pooling() is True
        assert SeedPool(box, (15, 25)).pooling() is True


class TestSeedPool3D:
    def test_point_inside_box(self):
        # bbox (zmin, ymin, xmin, zmax, ymax, xmax)
        box = (5, 10, 10, 15, 20, 20)
        assert SeedPool(box, (10, 15, 15)).pooling() is False

    def test_point_outside_z(self):
        box = (5, 10, 10, 15, 20, 20)
        assert SeedPool(box, (0, 15, 15)).pooling() is True


class TestUnetStarMask:
    def test_masking_inside(self):
        box = (10, 10, 20, 20)
        # Point inside box → masking() returns False (don't include).
        assert UnetStarMask(box, (15, 15)).masking() is False

    def test_masking_outside(self):
        box = (10, 10, 20, 20)
        assert UnetStarMask(box, (5, 5)).masking() is True

    def test_semi_masking_skips_axis_0(self):
        # 3D box; semi_masking only checks axes 1, 2 (not axis 0 / Z).
        box = (5, 10, 10, 15, 20, 20)
        # Point with bad Z but good YX should still semi-include.
        assert UnetStarMask(box, (0, 15, 15)).semi_masking() is False
        # Point with bad XY → semi-include.
        assert UnetStarMask(box, (10, 0, 0)).semi_masking() is True
