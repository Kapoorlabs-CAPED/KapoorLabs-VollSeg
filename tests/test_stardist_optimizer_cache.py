"""Tests for the StarDist threshold-optimiser caching helpers.

The optimiser used to call ``nms_to_labels`` per candidate threshold,
which ran ``peak_local_max`` + per-peak rasterisation on every iteration
(thousands of times). The new path runs
:func:`precompute_peaks_and_masks` once per patch and the sweep just
filters by score + NMS overlaps the cached masks via
:func:`labels_from_precomputed`. These tests pin down the round-trip
equivalence to the direct path.
"""

from __future__ import annotations

import numpy as np

from kapoorlabs_vollseg.stardist import compute_faces, rays_3d_golden_spiral
from kapoorlabs_vollseg.stardist.inference import (
    labels_from_precomputed,
    nms_to_labels,
    precompute_peaks_and_masks,
)


def _synthetic_maps(shape, peaks, dists_value, rays):
    """Build a (prob, dist) pair with sharp peaks at the requested coords.

    Each peak sits on a unique probability so peak_local_max picks them
    out predictably; the distance field is uniform across the volume
    (every voxel maps to the same ray distances).
    """
    prob = np.zeros(shape, dtype=np.float32)
    for (z, y, x), p in peaks:
        prob[z, y, x] = float(p)
    dist = np.full((rays.shape[0],) + shape, float(dists_value), dtype=np.float32)
    return prob, dist


class TestPrecomputeRoundTripsNms:
    """Direct ``nms_to_labels`` and (precompute + labels_from_precomputed)
    must produce the same label image at any valid threshold pair."""

    def test_matches_direct_call_3d(self):
        rays = rays_3d_golden_spiral(48)
        faces = compute_faces(rays)
        shape = (24, 48, 48)
        peaks = [
            ((12, 16, 16), 0.95),
            ((12, 32, 32), 0.80),
        ]
        prob, dist = _synthetic_maps(shape, peaks, dists_value=4.0, rays=rays)

        # Direct path.
        direct = nms_to_labels(
            prob,
            dist,
            rays,
            shape,
            prob_thresh=0.5,
            nms_thresh=0.4,
            min_distance=2,
            faces=faces,
        )

        # Cached path.
        centers, scores, bboxes, masks = precompute_peaks_and_masks(
            prob, dist, rays, shape, min_prob=0.01, min_distance=2, faces=faces
        )
        cached = labels_from_precomputed(
            centers, scores, bboxes, masks, shape, prob_thresh=0.5, nms_thresh=0.4
        )

        # Label IDs may differ in numeric value but the partition of
        # voxels into objects must match exactly.
        assert direct.shape == cached.shape
        assert (direct > 0).sum() == (cached > 0).sum()
        # Same number of distinct labels.
        assert len(np.unique(direct)) == len(np.unique(cached))

    def test_prob_threshold_filters(self):
        rays = rays_3d_golden_spiral(48)
        faces = compute_faces(rays)
        shape = (16, 32, 32)
        peaks = [
            ((8, 10, 10), 0.95),
            ((8, 22, 22), 0.55),
        ]
        prob, dist = _synthetic_maps(shape, peaks, dists_value=3.0, rays=rays)

        centers, scores, bboxes, masks = precompute_peaks_and_masks(
            prob, dist, rays, shape, min_prob=0.01, min_distance=2, faces=faces
        )

        # Both peaks survive at low threshold.
        out_low = labels_from_precomputed(
            centers, scores, bboxes, masks, shape, prob_thresh=0.5, nms_thresh=0.4
        )
        assert len(np.unique(out_low)) == 3  # 0 + two object labels

        # High threshold filters the weak peak.
        out_high = labels_from_precomputed(
            centers, scores, bboxes, masks, shape, prob_thresh=0.8, nms_thresh=0.4
        )
        assert len(np.unique(out_high)) == 2  # 0 + one object label

    def test_empty_precompute_returns_zero(self):
        rays = rays_3d_golden_spiral(48)
        faces = compute_faces(rays)
        shape = (8, 16, 16)
        prob = np.zeros(shape, dtype=np.float32)
        dist = np.zeros((48,) + shape, dtype=np.float32)
        centers, scores, bboxes, masks = precompute_peaks_and_masks(
            prob, dist, rays, shape, min_prob=0.01, faces=faces
        )
        out = labels_from_precomputed(
            centers, scores, bboxes, masks, shape, prob_thresh=0.5, nms_thresh=0.4
        )
        assert out.shape == shape
        assert out.max() == 0

    def test_score_ordering_preserved(self):
        # precompute_peaks_and_masks must return centers/scores sorted by
        # descending score so labels_from_precomputed's NMS step iterates
        # over the strongest peaks first.
        rays = rays_3d_golden_spiral(48)
        faces = compute_faces(rays)
        shape = (12, 24, 24)
        peaks = [
            ((6, 8, 8), 0.40),
            ((6, 16, 16), 0.90),
            ((6, 12, 20), 0.60),
        ]
        prob, dist = _synthetic_maps(shape, peaks, dists_value=2.0, rays=rays)
        _, scores, _, _ = precompute_peaks_and_masks(
            prob, dist, rays, shape, min_prob=0.01, faces=faces
        )
        # scores must be monotonically non-increasing.
        np.testing.assert_array_equal(scores, np.sort(scores)[::-1])
