"""Tests for vollseg.eval.matching — IoU / F1 / precision / recall."""

from __future__ import annotations

import numpy as np
import pytest

from vollseg.eval import (
    accuracy,
    f1,
    intersection_over_union,
    matching,
    matching_dataset,
    precision,
    recall,
)


class TestMatching:
    def test_perfect_match(self, labels_2d_two_blobs):
        m = matching(labels_2d_two_blobs, labels_2d_two_blobs)
        assert m.tp == 2
        assert m.fp == 0
        assert m.fn == 0
        assert m.f1 == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0

    def test_completely_disjoint(self):
        # GT has one object on the left, pred has one on the right — no overlap.
        gt = np.zeros((20, 40), dtype=np.int32)
        pred = np.zeros((20, 40), dtype=np.int32)
        gt[5:15, 5:15] = 1
        pred[5:15, 25:35] = 1
        m = matching(gt, pred, thresh=0.5)
        assert m.tp == 0
        assert m.fp == 1
        assert m.fn == 1

    def test_partial_overlap_threshold(self):
        # 10×10 GT, 10×10 pred shifted by 5 — IoU = 5/15 ≈ 0.33.
        gt = np.zeros((20, 20), dtype=np.int32)
        pred = np.zeros((20, 20), dtype=np.int32)
        gt[5:15, 5:15] = 1
        pred[5:15, 10:20] = 1
        # Below threshold → no match.
        assert matching(gt, pred, thresh=0.5).tp == 0
        # Above threshold → match.
        assert matching(gt, pred, thresh=0.2).tp == 1

    def test_threshold_iterable(self, labels_2d_two_blobs):
        results = matching(labels_2d_two_blobs, labels_2d_two_blobs, thresh=(0.3, 0.5, 0.7))
        assert len(results) == 3
        for r in results:
            assert r.f1 == 1.0


class TestMatchingDataset:
    def test_perfect_dataset(self, labels_2d_two_blobs):
        results = matching_dataset(
            [labels_2d_two_blobs, labels_2d_two_blobs],
            [labels_2d_two_blobs, labels_2d_two_blobs],
            thresh=0.5,
            show_progress=False,
        )
        assert results.f1 == 1.0
        assert results.precision == 1.0


class TestMetricFunctions:
    def test_precision(self):
        assert precision(8, 2, 0) == 0.8
        assert precision(0, 0, 5) == 0

    def test_recall(self):
        assert recall(8, 0, 2) == 0.8
        assert recall(0, 5, 0) == 0

    def test_f1(self):
        # F1 of perfect = 1
        assert f1(10, 0, 0) == 1.0
        assert f1(0, 5, 5) == 0

    def test_accuracy(self):
        assert accuracy(8, 1, 1) == pytest.approx(0.8)


class TestIoUMatrix:
    def test_iou_self(self):
        overlap = np.array([
            [0, 0, 0],
            [0, 5, 0],
            [0, 0, 5],
        ], dtype=np.int32)
        iou = intersection_over_union(overlap)
        # Diagonal: each label perfectly matches itself.
        assert pytest.approx(iou[1, 1]) == 1.0
        assert pytest.approx(iou[2, 2]) == 1.0
