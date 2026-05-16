"""Import + sampling smoke tests — no Qt event loop required."""

from __future__ import annotations

import numpy as np


def test_imports():
    from kapoorlabs_curvature_napari import CurvatureWidget  # noqa: F401
    from kapoorlabs_curvature_napari._profile import (  # noqa: F401
        LineProfile,
        record_timelapse_kymograph,
        sample_polyline,
    )


def test_sample_polyline_simple_line():
    """Straight horizontal line → profile equals image row."""
    from kapoorlabs_curvature_napari._profile import sample_polyline

    img = np.arange(100, dtype=np.float32).reshape(10, 10)
    prof = sample_polyline(
        img,
        points=np.array([[5, 0], [5, 9]], dtype=np.float64),
        num_samples=10,
    )
    # Resampled to 10 — should be approximately row 5 (50..59).
    assert prof.shape == (10,)
    assert np.allclose(prof, img[5], atol=0.5)


def test_record_timelapse_kymograph_shapes():
    """Build a tiny 3-frame timelapse and confirm kymograph shapes."""
    from kapoorlabs_curvature_napari._profile import (
        LineProfile,
        record_timelapse_kymograph,
    )

    curv = np.zeros((3, 8, 8), dtype=np.float32)
    for t in range(3):
        curv[t, 3, :] = float(t + 1)  # vary along T
    line = LineProfile(line_id=0, points=np.array([[3, 0], [3, 7]], dtype=np.float64))

    rec = record_timelapse_kymograph(
        {"curvature": curv},
        [line],
        n_frames=3,
        samples_per_line=16,
    )
    ky = rec.kymographs["curvature"][0]
    assert ky.shape == (3, 16)
    # Each frame's mean along the line should equal the constant value used above.
    means = ky.mean(axis=1)
    assert np.allclose(means, [1, 2, 3], atol=0.2)
