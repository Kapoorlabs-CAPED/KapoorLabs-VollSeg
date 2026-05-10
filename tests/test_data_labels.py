"""Tests for kapoorlabs_vollseg.data.labels — binary↔instance, erosion, scaling, hole-fill."""

from __future__ import annotations

import numpy as np
import pytest

from kapoorlabs_vollseg.data import (
    binary_to_labels,
    erode_labels,
    fill_label_holes,
    labels_to_binary,
    scale_labels,
    upscale_labels,
)


class TestBinaryLabel:
    def test_binary_to_labels_two_blobs(self, labels_2d_two_blobs):
        binary = labels_2d_two_blobs > 0
        labels = binary_to_labels(binary)
        assert labels.dtype == np.uint16
        assert labels.max() == 2

    def test_labels_to_binary(self, labels_2d_two_blobs):
        b = labels_to_binary(labels_2d_two_blobs)
        assert b.dtype == bool
        assert b.sum() == (labels_2d_two_blobs > 0).sum()


class TestErodeLabels:
    def test_zero_iterations_passthrough(self, labels_2d_two_blobs):
        out = erode_labels(labels_2d_two_blobs, iterations=0)
        np.testing.assert_array_equal(out, labels_2d_two_blobs)

    def test_erosion_preserves_label_ids(self, labels_2d_two_blobs):
        out = erode_labels(labels_2d_two_blobs, iterations=1)
        # Label IDs unchanged; eroded version has fewer pixels per id.
        for lid in (1, 2):
            assert (out == lid).sum() < (labels_2d_two_blobs == lid).sum()
            assert (out == lid).sum() > 0


class TestFillLabelHoles:
    def test_hole_filled(self):
        # Label image with a one-pixel hole in the middle.
        img = np.zeros((20, 20), dtype=np.int32)
        img[5:15, 5:15] = 1
        img[10, 10] = 0
        filled = fill_label_holes(img)
        assert filled[10, 10] == 1


class TestScaleLabels:
    def test_unit_scaling_passthrough(self, labels_2d_two_blobs):
        out = scale_labels(labels_2d_two_blobs, (1.0, 1.0))
        np.testing.assert_array_equal(out, labels_2d_two_blobs)

    def test_dtype_preserved(self, labels_3d_two_blobs):
        scaled = scale_labels(labels_3d_two_blobs, 0.5)
        assert scaled.dtype == labels_3d_two_blobs.dtype

    def test_label_ids_preserved(self, labels_3d_two_blobs):
        # Order-0 (NN) interpolation must not introduce new labels.
        scaled = scale_labels(labels_3d_two_blobs, 0.5)
        new_labels = set(np.unique(scaled).tolist())
        old_labels = set(np.unique(labels_3d_two_blobs).tolist())
        assert new_labels.issubset(old_labels)

    def test_negative_scale_rejected(self, labels_3d_two_blobs):
        with pytest.raises(ValueError):
            scale_labels(labels_3d_two_blobs, -1.0)

    def test_non_integer_rejected(self):
        with pytest.raises(TypeError):
            scale_labels(np.zeros((10, 10), dtype=np.float32), 2.0)


class TestUpscaleLabels:
    def test_target_shape_exact(self):
        img = np.zeros((10, 10), dtype=np.uint16)
        img[2:5, 2:5] = 7
        out = upscale_labels(img, target_shape=(20, 20))
        assert out.shape == (20, 20)
        assert 7 in np.unique(out)
