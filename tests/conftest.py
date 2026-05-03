"""Test-suite shared fixtures and helpers.

The deps map varies across the codebase — keras tests need TF, PyTorch
tests need torch + lightning + careamics, cellpose tests need cellpose,
HF tests would need huggingface_hub. Each test module gates with
``pytest.importorskip`` so the suite degrades gracefully on machines
that only have a subset installed.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def labels_2d_two_blobs():
    """A 64×64 label image with two well-separated solid disks."""
    img = np.zeros((64, 64), dtype=np.int32)
    yy, xx = np.mgrid[0:64, 0:64]
    img[(yy - 16) ** 2 + (xx - 16) ** 2 <= 49] = 1     # disk 1, r=7 at (16,16)
    img[(yy - 48) ** 2 + (xx - 48) ** 2 <= 36] = 2     # disk 2, r=6 at (48,48)
    return img


@pytest.fixture
def labels_3d_two_blobs():
    """A 16×32×32 label image with two non-overlapping spheres."""
    img = np.zeros((16, 32, 32), dtype=np.int32)
    zz, yy, xx = np.mgrid[0:16, 0:32, 0:32]
    img[(zz - 5) ** 2 + (yy - 8) ** 2 + (xx - 8) ** 2 <= 9] = 1
    img[(zz - 11) ** 2 + (yy - 22) ** 2 + (xx - 22) ** 2 <= 9] = 2
    return img


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)
