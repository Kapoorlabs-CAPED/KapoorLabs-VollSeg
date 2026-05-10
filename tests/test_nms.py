"""Tests for kapoorlabs_vollseg.eval.nms.NMSLabel — bounding-box NMS on label images."""

from __future__ import annotations

import numpy as np

from kapoorlabs_vollseg.eval import NMSLabel


class TestSuppressOverlapping2D:
    def test_no_overlap_keeps_all(self):
        img = np.zeros((30, 30), dtype=np.int32)
        img[5:10, 5:10] = 1
        img[15:20, 15:20] = 2
        out = NMSLabel(img, nms_thresh=0.5).suppress_overlapping()
        # Both labels survive.
        assert set(np.unique(out)) - {0} == {1, 2}

    def test_perfect_overlap_collapses(self):
        # Two labels occupying the same bbox — one should be remapped to the other.
        img = np.zeros((20, 20), dtype=np.int32)
        img[5:15, 5:15] = 1
        img[5:15, 5:15] = 2  # overwrites — single label remains
        out = NMSLabel(img, nms_thresh=0.5).suppress_overlapping()
        # Result is unchanged when only one label is present.
        assert set(np.unique(out)) - {0} == {2}

    def test_one_inside_another(self):
        # Inner box fully contained inside outer → outer absorbs inner.
        img = np.zeros((30, 30), dtype=np.int32)
        img[5:25, 5:25] = 1
        img[10:15, 10:15] = 2
        out = NMSLabel(img, nms_thresh=0.5).suppress_overlapping()
        # The "inside" label (2) gets remapped — only one should remain.
        assert len(set(np.unique(out)) - {0}) == 1


class TestSuppressThinZ:
    def test_drops_thin(self):
        # 3D label; one object spans 1 z-slice, one spans 5.
        img = np.zeros((10, 20, 20), dtype=np.int32)
        img[3, 5:10, 5:10] = 1  # thin: 1 slice
        img[2:7, 12:17, 12:17] = 2  # thick: 5 slices
        out = NMSLabel(img, nms_thresh=0.5, z_thresh=2).suppress_thin_z()
        assert (out == 1).sum() == 0  # thin dropped
        assert (out == 2).sum() > 0  # thick kept

    def test_2d_passthrough(self):
        img = np.zeros((20, 20), dtype=np.int32)
        img[5:10, 5:10] = 1
        out = NMSLabel(img, nms_thresh=0.5, z_thresh=2).suppress_thin_z()
        np.testing.assert_array_equal(out, img)
