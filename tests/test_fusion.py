"""Tests for kapoorlabs_vollseg.fusion — watershed_fuse and cellpose_watershed_fuse.

These fuse already-segmented inputs; we use synthetic labels + masks
that don't require any model.
"""

from __future__ import annotations

import numpy as np
import pytest

from kapoorlabs_vollseg import cellpose_watershed_fuse, watershed_fuse


class TestWatershedFuse:
    def test_3d_smoke(self, labels_3d_two_blobs):
        """The seedpool watershed should produce a label image of the same shape."""
        # Use the labels themselves as the "stardist labels" and as the mask.
        image = (labels_3d_two_blobs > 0).astype(np.float32)
        mask = labels_3d_two_blobs > 0
        out = watershed_fuse(
            image, stardist_labels=labels_3d_two_blobs, unet_mask=mask, seedpool=True
        )
        assert out.shape == labels_3d_two_blobs.shape
        # At least the two original objects should survive in some form.
        assert len(set(np.unique(out)) - {0}) >= 2

    def test_2d_smoke(self, labels_2d_two_blobs):
        image = (labels_2d_two_blobs > 0).astype(np.float32)
        mask = labels_2d_two_blobs > 0
        out = watershed_fuse(
            image, stardist_labels=labels_2d_two_blobs, unet_mask=mask, seedpool=True
        )
        assert out.shape == labels_2d_two_blobs.shape

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError):
            watershed_fuse(
                np.zeros((10,)),
                stardist_labels=np.zeros((10,), dtype=np.int32),
                unet_mask=np.zeros((10,), dtype=bool),
            )


class TestCellposeWatershedFuse:
    def test_3d_smoke(self, labels_3d_two_blobs):
        # Use blobs as nuclei seeds, and the same blobs (dilated) as the
        # cellpose mask.
        membrane = np.random.rand(*labels_3d_two_blobs.shape).astype(np.float32)
        mask = labels_3d_two_blobs > 0
        out = cellpose_watershed_fuse(
            membrane, nuclei_labels=labels_3d_two_blobs, cellpose_mask=mask
        )
        assert out.shape == labels_3d_two_blobs.shape
