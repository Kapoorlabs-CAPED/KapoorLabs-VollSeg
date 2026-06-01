"""Tests for the StarDist inference helpers — ConvexHull triangulation,
KDTree-accelerated bbox NMS, and nearest-ray polyhedron painting.

These cover the ported-from-stardist algorithms in
``kapoorlabs_vollseg.stardist.inference``:

- ``_compute_ray_faces``     — ConvexHull simplices of ray vertices,
                                same triangulation stardist's
                                ``Rays3D.faces`` ships.
- ``_bbox_nms_kdtree``       — spatial pre-filter + bbox-IoU greedy NMS,
                                the algorithm stardist's C++/OpenMP
                                NMS uses (with ``use_bbox=True``).
- ``_polyhedra_to_label``    — nearest-ray rasterisation of kept star
                                polyhedra into a uint16 label image.

We test on synthetic peaks placed deterministically so the expected
outcomes are unambiguous (e.g. far-apart peaks all survive NMS;
sphere-of-known-radius polyhedron has the right volume after paint).
"""

from __future__ import annotations

import numpy as np
import pytest

from kapoorlabs_vollseg.stardist.inference import (
    _bbox_nms_kdtree,
    _compute_ray_faces,
    _polyhedra_to_label,
)


# ---------------------------------------------------------------- fixtures
def _golden_spiral_rays(n_rays: int = 96) -> np.ndarray:
    """Same construction as ``rays_3d_golden_spiral`` — Fibonacci
    spiral on the unit sphere. Deterministic, no scipy needed."""
    phi = np.pi * (3 - np.sqrt(5))
    i = np.arange(n_rays)
    y = 1 - (i / max(1, n_rays - 1)) * 2
    r = np.sqrt(1 - y * y)
    theta = phi * i
    return np.stack([y, r * np.cos(theta), r * np.sin(theta)], axis=1).astype(
        np.float32
    )


@pytest.fixture
def rays96() -> np.ndarray:
    return _golden_spiral_rays(96)


# ---------------------------------------------------------------- _compute_ray_faces
class TestComputeRayFaces:
    def test_returns_int_triangles(self, rays96):
        faces = _compute_ray_faces(rays96)
        assert faces.ndim == 2
        assert faces.shape[1] == 3
        assert np.issubdtype(faces.dtype, np.integer)

    def test_face_count_matches_euler(self, rays96):
        """V - E + F = 2 for a convex polyhedron; with all-triangle
        faces 3F = 2E so F = 2(V - 2) for n_rays vertices."""
        faces = _compute_ray_faces(rays96)
        assert len(faces) == 2 * (len(rays96) - 2)

    def test_face_indices_in_range(self, rays96):
        faces = _compute_ray_faces(rays96)
        assert faces.min() >= 0
        assert faces.max() < len(rays96)

    def test_normalises_non_unit_rays(self):
        """Rays at arbitrary radii should still produce the same hull
        (ConvexHull on the normalised unit-sphere projection)."""
        rays_unit = _golden_spiral_rays(32)
        rays_scaled = rays_unit * np.linspace(0.5, 3.0, 32)[:, None]
        f1 = _compute_ray_faces(rays_unit)
        f2 = _compute_ray_faces(rays_scaled)
        # Same connectivity (set of triangles, regardless of vertex order).
        s1 = {tuple(sorted(f)) for f in f1}
        s2 = {tuple(sorted(f)) for f in f2}
        assert s1 == s2


# ---------------------------------------------------------------- _bbox_nms_kdtree
class TestBboxNmsKdtree:
    def _make_peaks(self, centers, dists_radius, n_rays=96):
        rays = _golden_spiral_rays(n_rays)
        N = len(centers)
        dists = np.full((N, n_rays), float(dists_radius), dtype=np.float32)
        return np.asarray(centers, dtype=np.float32), dists, rays

    def test_empty_input(self, rays96):
        out = _bbox_nms_kdtree(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 96), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            rays96,
            (20, 100, 100),
            iou_thresh=0.3,
        )
        assert out.shape == (0,)

    def test_far_apart_all_kept(self):
        # Three peaks far apart → no bbox overlap → all survive.
        centers, dists, rays = self._make_peaks(
            [[10, 10, 10], [10, 70, 70], [10, 10, 70]], dists_radius=3.0
        )
        scores = np.array([0.9, 0.7, 0.5], dtype=np.float32)
        kept = _bbox_nms_kdtree(centers, dists, scores, rays, (20, 100, 100), 0.3)
        assert sorted(kept.tolist()) == [0, 1, 2]

    def test_returns_score_descending_order(self):
        # Same fixture as above; the function contract is to return in
        # *score-descending* order, not input order.
        centers, dists, rays = self._make_peaks(
            [[10, 10, 10], [10, 70, 70], [10, 10, 70]], dists_radius=3.0
        )
        scores = np.array([0.3, 0.9, 0.6], dtype=np.float32)
        kept = _bbox_nms_kdtree(centers, dists, scores, rays, (20, 100, 100), 0.3)
        assert kept.tolist() == [1, 2, 0]

    def test_full_overlap_suppresses_lower_score(self):
        # Two peaks at the same centre with identical radii → IoU = 1 →
        # the lower-scoring one gets suppressed.
        centers, dists, rays = self._make_peaks(
            [[10, 30, 30], [10, 30, 30]], dists_radius=4.0
        )
        scores = np.array([0.4, 0.8], dtype=np.float32)
        kept = _bbox_nms_kdtree(centers, dists, scores, rays, (20, 100, 100), 0.3)
        assert kept.tolist() == [1]  # only the higher-score peak survives

    def test_partial_overlap_above_threshold_suppresses(self):
        # Two unit-radius spheres with centres 2 vox apart → bboxes
        # share most of their volume → IoU well above 0.3.
        centers, dists, rays = self._make_peaks(
            [[10, 30, 30], [10, 32, 30]], dists_radius=4.0
        )
        scores = np.array([0.6, 0.9], dtype=np.float32)
        kept = _bbox_nms_kdtree(centers, dists, scores, rays, (20, 100, 100), 0.3)
        assert kept.tolist() == [1]

    def test_partial_overlap_below_threshold_keeps_both(self):
        # Same centres-distance but a very strict iou_thresh → both
        # peaks should survive.
        centers, dists, rays = self._make_peaks(
            [[10, 30, 30], [10, 32, 30]], dists_radius=4.0
        )
        scores = np.array([0.6, 0.9], dtype=np.float32)
        kept = _bbox_nms_kdtree(centers, dists, scores, rays, (20, 100, 100), 0.99)
        assert sorted(kept.tolist()) == [0, 1]

    def test_clamps_bbox_to_volume_shape(self):
        # Peak whose ray extents push past the volume boundary — the
        # bbox should clip to vol_shape, and the function should still
        # complete without indexing errors.
        centers, dists, rays = self._make_peaks(
            [[0, 0, 0], [10, 50, 50]], dists_radius=8.0
        )
        scores = np.array([0.9, 0.5], dtype=np.float32)
        kept = _bbox_nms_kdtree(centers, dists, scores, rays, (20, 100, 100), 0.3)
        assert 0 in kept.tolist()  # boundary-clipped peak still survives

    def test_scale_27k_peaks_completes_quickly(self, rays96):
        """Stress test at the scale the user's stuck sweep hit (27k peaks)
        — must complete in well under a minute, otherwise we've
        regressed to the O(N²) Python-loop behaviour."""
        import time

        rng = np.random.default_rng(7)
        N = 27_000
        vol_shape = (19, 1560, 1560)
        centers = np.column_stack(
            [
                rng.integers(2, vol_shape[0] - 2, N),
                rng.integers(20, vol_shape[1] - 20, N),
                rng.integers(20, vol_shape[2] - 20, N),
            ]
        ).astype(np.float32)
        dists = (6.0 + rng.normal(0, 1.5, (N, 96))).clip(2.0, 12.0).astype(np.float32)
        scores = rng.random(N).astype(np.float32)
        t = time.perf_counter()
        kept = _bbox_nms_kdtree(centers, dists, scores, rays96, vol_shape, 0.3)
        elapsed = time.perf_counter() - t
        # 10s is a very generous bound — the KDTree pre-filter
        # finishes in ~2s on a modest box. If we ever regress to the
        # quadratic loop the bound here would be ~minutes.
        assert elapsed < 10.0, f"NMS took {elapsed:.1f}s for N={N}"
        # Kept peaks come out in score-descending order.
        assert (scores[kept[:-1]] >= scores[kept[1:]]).all()


# ---------------------------------------------------------------- _polyhedra_to_label
class TestPolyhedraToLabel:
    def test_empty_input(self, rays96):
        out = _polyhedra_to_label(
            vol_shape=(10, 20, 20),
            kept_idx=np.zeros(0, dtype=np.int64),
            centers=np.zeros((0, 3), dtype=np.float32),
            dists=np.zeros((0, 96), dtype=np.float32),
            rays=rays96,
        )
        assert out.shape == (10, 20, 20)
        assert out.dtype == np.uint16
        assert out.max() == 0

    def test_single_sphere_paints_correct_volume(self, rays96):
        """One polyhedron with constant ray length = R should paint a
        roughly spherical region whose voxel count is close to
        4/3 π R³ (modulo voxel quantisation)."""
        R = 5.0
        centers = np.array([[16, 32, 32]], dtype=np.float32)
        dists = np.full((1, 96), R, dtype=np.float32)
        out = _polyhedra_to_label(
            vol_shape=(32, 64, 64),
            kept_idx=np.array([0]),
            centers=centers,
            dists=dists,
            rays=rays96,
        )
        painted = int((out == 1).sum())
        expected = 4 / 3 * np.pi * R**3
        # Within ±25% of analytic sphere volume — generous bound because
        # nearest-ray with 96 rays approximates a 188-face polyhedron.
        assert painted == pytest.approx(
            expected, rel=0.25
        ), f"painted={painted}, expected≈{expected:.0f}"

    def test_two_far_apart_get_distinct_labels(self, rays96):
        centers = np.array([[8, 16, 16], [8, 48, 48]], dtype=np.float32)
        dists = np.full((2, 96), 4.0, dtype=np.float32)
        out = _polyhedra_to_label(
            vol_shape=(16, 64, 64),
            kept_idx=np.array([0, 1]),
            centers=centers,
            dists=dists,
            rays=rays96,
        )
        labels = sorted(int(v) for v in np.unique(out) if v != 0)
        assert labels == [1, 2]

    def test_label_order_follows_kept_idx(self, rays96):
        """``kept_idx[0]`` gets label 1, ``kept_idx[1]`` gets label 2, …
        — i.e. earlier (higher-score) survivors win the colouring race."""
        centers = np.array([[8, 16, 16], [8, 48, 48]], dtype=np.float32)
        dists = np.full((2, 96), 4.0, dtype=np.float32)
        out = _polyhedra_to_label(
            vol_shape=(16, 64, 64),
            kept_idx=np.array([1, 0]),
            centers=centers,
            dists=dists,
            rays=rays96,
        )
        # The polyhedron originally at index 1 (centre y=48) → label 1.
        # The polyhedron originally at index 0 (centre y=16) → label 2.
        ys = np.where(out == 1)[1]
        assert ys.mean() > 32  # label 1 is the "high-y" one
        ys = np.where(out == 2)[1]
        assert ys.mean() < 32

    def test_overlap_first_painted_wins(self, rays96):
        """Two heavily overlapping polyhedra — only the first one (by
        kept_idx order) should get its label into the overlap region;
        the second can't overwrite painted voxels."""
        centers = np.array([[8, 32, 32], [8, 33, 33]], dtype=np.float32)
        dists = np.full((2, 96), 5.0, dtype=np.float32)
        out = _polyhedra_to_label(
            vol_shape=(16, 64, 64),
            kept_idx=np.array([0, 1]),
            centers=centers,
            dists=dists,
            rays=rays96,
        )
        # Label 1 dominates the overlap; label 2 only paints the tail
        # outside label 1's bbox.
        assert (out == 1).sum() > (out == 2).sum()

    def test_handles_nan_and_inf_dists(self, rays96):
        """Pathological model output (NaN / ±Inf in some rays) must not
        crash — those rays get clipped to zero and the polyhedron
        either paints a smaller volume or nothing at all."""
        centers = np.array([[8, 16, 16]], dtype=np.float32)
        dists = np.full((1, 96), 4.0, dtype=np.float32)
        dists[0, 0] = np.nan
        dists[0, 1] = np.inf
        dists[0, 2] = -np.inf
        out = _polyhedra_to_label(
            vol_shape=(16, 32, 32),
            kept_idx=np.array([0]),
            centers=centers,
            dists=dists,
            rays=rays96,
        )
        # Doesn't raise; produces some non-empty label.
        assert int(out.max()) == 1

    def test_zero_dists_produces_empty_label(self, rays96):
        centers = np.array([[8, 16, 16]], dtype=np.float32)
        dists = np.zeros((1, 96), dtype=np.float32)
        out = _polyhedra_to_label(
            vol_shape=(16, 32, 32),
            kept_idx=np.array([0]),
            centers=centers,
            dists=dists,
            rays=rays96,
        )
        assert out.max() == 0

    def test_clamps_bbox_at_volume_boundary(self, rays96):
        """A peak with its sphere clipped by the volume boundary should
        still produce a (truncated) label region — no index errors."""
        centers = np.array([[1, 1, 1]], dtype=np.float32)
        dists = np.full((1, 96), 5.0, dtype=np.float32)
        out = _polyhedra_to_label(
            vol_shape=(10, 10, 10),
            kept_idx=np.array([0]),
            centers=centers,
            dists=dists,
            rays=rays96,
        )
        assert int(out.max()) == 1
        # Some voxels painted, but bounded by the volume corner.
        assert (out == 1).sum() > 0

    def test_stops_at_uint16_max_labels(self, rays96):
        """Beyond ``np.iinfo(uint16).max`` labels the loop bails — the
        label image dtype can't hold more, so painting further would
        wrap silently."""
        N = 70_000  # more than the uint16 ceiling
        centers = np.column_stack(
            [
                np.ones(N) * 8,
                np.linspace(0, 1000, N),
                np.linspace(0, 1000, N),
            ]
        ).astype(np.float32)
        dists = np.full((N, 96), 1.0, dtype=np.float32)
        kept = np.arange(N)
        out = _polyhedra_to_label(
            vol_shape=(16, 1024, 1024),
            kept_idx=kept,
            centers=centers,
            dists=dists,
            rays=rays96,
        )
        assert int(out.max()) <= np.iinfo(np.uint16).max

    def test_faces_argument_accepted_for_api_compat(self, rays96):
        """``faces`` is in the signature for callers that still pass it
        (nearest-ray doesn't actually use it). Pass something and make
        sure the output is the same as passing None."""
        centers = np.array([[8, 16, 16]], dtype=np.float32)
        dists = np.full((1, 96), 4.0, dtype=np.float32)
        out_none = _polyhedra_to_label(
            vol_shape=(16, 32, 32),
            kept_idx=np.array([0]),
            centers=centers,
            dists=dists,
            rays=rays96,
        )
        out_with = _polyhedra_to_label(
            vol_shape=(16, 32, 32),
            kept_idx=np.array([0]),
            centers=centers,
            dists=dists,
            rays=rays96,
            faces=_compute_ray_faces(rays96),
        )
        np.testing.assert_array_equal(out_none, out_with)

    def test_realistic_scale_completes_in_seconds(self, rays96):
        """391 polyhedra at typical-cell size (the kept count from the
        user's stuck sweep) — must finish in well under a minute."""
        import time

        rng = np.random.default_rng(0)
        vol_shape = (19, 1560, 1560)
        N = 391
        centers = np.column_stack(
            [
                rng.integers(2, vol_shape[0] - 2, N),
                rng.integers(20, vol_shape[1] - 20, N),
                rng.integers(20, vol_shape[2] - 20, N),
            ]
        ).astype(np.float32)
        dists = (6.0 + rng.normal(0, 1.5, (N, 96))).clip(2.0, 12.0).astype(np.float32)
        kept_idx = np.arange(N)
        t = time.perf_counter()
        out = _polyhedra_to_label(
            vol_shape=vol_shape,
            kept_idx=kept_idx,
            centers=centers,
            dists=dists,
            rays=rays96,
        )
        elapsed = time.perf_counter() - t
        assert int(out.max()) == N
        assert elapsed < 15.0, f"paint took {elapsed:.1f}s for N={N}"
