"""Tests for vollseg.pipelines.base — Result dataclass, infer_axes."""

from __future__ import annotations

import numpy as np
import pytest

from vollseg import Result
from vollseg.pipelines.base import infer_axes


class TestResult:
    def test_default_all_none(self):
        r = Result()
        assert r.labels is None
        assert r.semantic is None
        assert r.denoised is None
        assert r.probability is None
        assert r.extra == {}

    def test_merge_overrides_field(self):
        r = Result(denoised=np.array([1, 2]))
        out = r.merge(labels=np.array([3, 4]))
        np.testing.assert_array_equal(out.denoised, [1, 2])
        np.testing.assert_array_equal(out.labels, [3, 4])
        # Original unchanged.
        assert r.labels is None

    def test_merge_returns_copy(self):
        r = Result(denoised=np.array([1, 2]))
        out = r.merge(denoised=np.array([5, 6]))
        assert out is not r


class TestInferAxes:
    def test_2d(self):
        assert infer_axes(np.zeros((10, 10))) == "YX"

    def test_3d(self):
        assert infer_axes(np.zeros((5, 10, 10))) == "ZYX"

    def test_4d(self):
        assert infer_axes(np.zeros((5, 10, 10, 3))) == "ZYXC"

    def test_invalid(self):
        with pytest.raises(ValueError):
            infer_axes(np.zeros((10,)))
        with pytest.raises(ValueError):
            infer_axes(np.zeros((1, 1, 1, 1, 1)))
