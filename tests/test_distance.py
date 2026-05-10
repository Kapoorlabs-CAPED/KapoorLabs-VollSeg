"""Tests for kapoorlabs_vollseg.stardist.distance — prob target + ray-march distance map."""

from __future__ import annotations

import numpy as np
import pytest

from kapoorlabs_vollseg.stardist import (
    compute_distance_map,
    foreground_probability_map,
    rays_2d,
    rays_3d_golden_spiral,
)


class TestForegroundProbabilityMap:
    def test_background_only(self):
        empty = np.zeros((20, 20), dtype=np.int32)
        prob = foreground_probability_map(empty)
        assert prob.shape == empty.shape
        assert prob.dtype == np.float32
        assert prob.max() == 0.0

    def test_peak_is_one(self, labels_2d_two_blobs):
        prob = foreground_probability_map(labels_2d_two_blobs)
        # Each object's interior peaks at exactly 1.0 (EDT divided by per-object max).
        assert pytest.approx(prob.max(), abs=1e-6) == 1.0

    def test_background_is_zero(self, labels_2d_two_blobs):
        prob = foreground_probability_map(labels_2d_two_blobs)
        bg = labels_2d_two_blobs == 0
        assert prob[bg].max() == 0.0

    def test_per_object_normalization(self, labels_2d_two_blobs):
        # Both objects should reach 1.0 at their respective centers.
        prob = foreground_probability_map(labels_2d_two_blobs)
        for lbl in (1, 2):
            assert (
                pytest.approx(prob[labels_2d_two_blobs == lbl].max(), abs=1e-6) == 1.0
            )


class TestComputeDistanceMap:
    def test_2d_shape_and_dtype(self, labels_2d_two_blobs):
        rays = rays_2d(8)
        dist = compute_distance_map(labels_2d_two_blobs, rays)
        assert dist.shape == (8,) + labels_2d_two_blobs.shape
        assert dist.dtype == np.float32

    def test_2d_background_zero(self, labels_2d_two_blobs):
        rays = rays_2d(8)
        dist = compute_distance_map(labels_2d_two_blobs, rays)
        bg = labels_2d_two_blobs == 0
        # Sum across rays at background pixels should be zero.
        assert dist[:, bg].sum() == 0.0

    def test_2d_disk_center_radius(self):
        """At a disk's center, the distance along any ray should ≈ disk radius."""
        # 81×81 disk, radius 20 at center.
        img = np.zeros((81, 81), dtype=np.int32)
        yy, xx = np.mgrid[0:81, 0:81]
        img[(yy - 40) ** 2 + (xx - 40) ** 2 <= 20**2] = 1

        rays = rays_2d(16)
        dist = compute_distance_map(img, rays)
        center_dists = dist[:, 40, 40]
        # Allow ±2 px tolerance for the discrete ray march.
        assert np.all(np.abs(center_dists - 20.0) <= 2.0)

    def test_3d_shape(self, labels_3d_two_blobs):
        rays = rays_3d_golden_spiral(16)
        dist = compute_distance_map(labels_3d_two_blobs, rays)
        assert dist.shape == (16,) + labels_3d_two_blobs.shape

    def test_3d_background_zero(self, labels_3d_two_blobs):
        rays = rays_3d_golden_spiral(16)
        dist = compute_distance_map(labels_3d_two_blobs, rays)
        bg = labels_3d_two_blobs == 0
        assert dist[:, bg].sum() == 0.0

    def test_rays_dim_must_match_labels(self):
        img2d = np.zeros((10, 10), dtype=np.int32)
        with pytest.raises(ValueError):
            compute_distance_map(
                img2d, rays_3d_golden_spiral(8)
            )  # 3D rays on 2D labels
