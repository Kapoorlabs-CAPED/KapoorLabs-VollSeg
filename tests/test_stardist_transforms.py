"""Tests for vollseg.stardist.transforms — flips and rotations.

These transforms are deterministic only with p=1, so each test forces
the augmentation to fire and then checks the geometric invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vollseg.stardist import (
    Compose,
    InputGaussianNoise,
    InputPercentileNormalize,
    RandomFlip,
    RandomRot90,
)


@pytest.fixture
def raw_label_2d():
    raw = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    label = torch.zeros(4, 4, dtype=torch.int32)
    label[1:3, 1:3] = 7
    return raw, label


@pytest.fixture
def raw_label_3d():
    raw = torch.arange(64, dtype=torch.float32).reshape(4, 4, 4)
    label = torch.zeros(4, 4, 4, dtype=torch.int32)
    label[1:3, 1:3, 1:3] = 7
    return raw, label


class TestRandomFlip:
    def test_p_zero_identity(self, raw_label_2d):
        raw, label = raw_label_2d
        flip = RandomFlip(p=0.0)
        out_raw, out_label = flip(raw, label)
        torch.testing.assert_close(out_raw, raw)
        torch.testing.assert_close(out_label, label)

    def test_p_one_flips_every_axis(self, raw_label_2d):
        torch.manual_seed(0)
        raw, label = raw_label_2d
        flip = RandomFlip(p=1.0)
        out_raw, out_label = flip(raw, label)
        # All axes flipped → equivalent to flip(raw, dims=[0,1])
        expected_raw = torch.flip(raw, dims=[0, 1])
        expected_label = torch.flip(label, dims=[0, 1])
        torch.testing.assert_close(out_raw, expected_raw)
        torch.testing.assert_close(out_label, expected_label)

    def test_3d_works(self, raw_label_3d):
        torch.manual_seed(0)
        raw, label = raw_label_3d
        out_raw, out_label = RandomFlip(p=1.0)(raw, label)
        # Shape preserved, content flipped.
        assert out_raw.shape == raw.shape
        torch.testing.assert_close(out_label.sum(), label.sum())


class TestRandomRot90:
    def test_preserves_label_count(self, raw_label_2d):
        # All four rotations preserve the *count* of foreground voxels.
        torch.manual_seed(1)
        raw, label = raw_label_2d
        out_raw, out_label = RandomRot90(p=1.0)(raw, label)
        assert (out_label > 0).sum() == (label > 0).sum()

    def test_p_zero_identity(self, raw_label_2d):
        raw, label = raw_label_2d
        out_raw, out_label = RandomRot90(p=0.0)(raw, label)
        torch.testing.assert_close(out_raw, raw)


class TestInputOnlyTransforms:
    def test_gaussian_noise_doesnt_touch_label(self, raw_label_2d):
        torch.manual_seed(0)
        raw, label = raw_label_2d
        out_raw, out_label = InputGaussianNoise(std=0.1, p=1.0)(raw, label)
        torch.testing.assert_close(out_label, label)
        # Raw was perturbed.
        assert not torch.allclose(out_raw, raw)

    def test_percentile_normalize_to_unit_range(self):
        raw = torch.arange(100, dtype=torch.float32).reshape(10, 10)
        label = torch.zeros(10, 10, dtype=torch.int32)
        out_raw, _ = InputPercentileNormalize(pmin=0, pmax=100)(raw, label)
        # With pmin=0/pmax=100, output spans roughly [0, 1].
        assert pytest.approx(out_raw.min().item(), abs=1e-3) == 0.0
        assert pytest.approx(out_raw.max().item(), abs=1e-3) == 1.0


class TestCompose:
    def test_chain_application(self, raw_label_2d):
        raw, label = raw_label_2d
        pipe = Compose([
            InputPercentileNormalize(pmin=0, pmax=100),
            RandomFlip(p=0.0),
            RandomRot90(p=0.0),
        ])
        out_raw, out_label = pipe(raw, label)
        # Only normalize fired (others identity at p=0).
        assert out_raw.min() >= 0.0
        torch.testing.assert_close(out_label, label)
