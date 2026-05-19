"""Tests for the triangulated star-convex polyhedron rasteriser used by
the StarDist inference path.

The legacy code rasterised each peak as a union of cones (one per ray);
the new code uses the actual ConvexHull triangulation as a union of
tetrahedra ``(center, scaled_v_a, scaled_v_b, scaled_v_c)``, matching
upstream ``stardist.polyhedron_to_label``. These tests pin down the
geometric behaviour rather than the implementation detail.
"""

from __future__ import annotations

import numpy as np
import pytest

from kapoorlabs_vollseg.stardist import compute_faces, rays_3d_golden_spiral
from kapoorlabs_vollseg.stardist.inference import (
    _inside_polyhedron,
    _rasterize_to_bbox,
)


@pytest.fixture(scope="module")
def rays96():
    return rays_3d_golden_spiral(96)


@pytest.fixture(scope="module")
def faces96(rays96):
    return compute_faces(rays96)


class TestRasterizeDispatch:
    def test_polyhedron_path_with_faces(self, rays96, faces96):
        center = np.array([20, 20, 20], dtype=np.float64)
        dists = np.full(96, 6.0, dtype=np.float64)
        bbox, mask = _rasterize_to_bbox(center, rays96, dists, (40, 40, 40), faces96)
        # mask is bool, bbox slices fit the full shape, center is inside.
        assert mask.dtype == bool
        assert mask.any()
        # Center voxel is always inside the polyhedron.
        z, y, x = (s.start for s in bbox)
        assert mask[20 - z, 20 - y, 20 - x]

    def test_cone_fallback_without_faces(self, rays96):
        center = np.array([20, 20, 20], dtype=np.float64)
        dists = np.full(96, 6.0, dtype=np.float64)
        bbox, mask = _rasterize_to_bbox(center, rays96, dists, (40, 40, 40), faces=None)
        assert mask.any()

    def test_polyhedron_underestimates_sphere(self, rays96, faces96):
        # A polyhedron with 96 rays at uniform distance r is INSCRIBED in a
        # sphere of radius r — the rasterised mask volume must be less than
        # the true sphere volume (4/3·πr³). The cone fallback isn't bound
        # by this since cones can over-paint at gaps between rays.
        center = np.array([24, 24, 24], dtype=np.float64)
        r = 8.0
        dists = np.full(96, r, dtype=np.float64)
        _, mask = _rasterize_to_bbox(center, rays96, dists, (48, 48, 48), faces96)
        true_sphere_vol = 4.0 / 3.0 * np.pi * r**3
        # Within ~20% of sphere volume but strictly less.
        assert mask.sum() < true_sphere_vol
        assert mask.sum() > 0.7 * true_sphere_vol


class TestConcavity:
    def test_short_rays_carve_volume(self, rays96, faces96):
        # Uniform polyhedron has some volume V; shrinking every 4th ray
        # to 1/4 of the others MUST reduce the rasterised volume —
        # this distinguishes a true star-polyhedron from a max-radius
        # cone-or-ball approximation.
        center = np.array([20, 20, 20], dtype=np.float64)
        uniform = np.full(96, 8.0)
        carved = uniform.copy()
        carved[::4] = 2.0

        _, mask_uni = _rasterize_to_bbox(center, rays96, uniform, (40, 40, 40), faces96)
        _, mask_carve = _rasterize_to_bbox(
            center, rays96, carved, (40, 40, 40), faces96
        )
        assert mask_carve.sum() < mask_uni.sum()
        # The carved polyhedron is still strictly contained in the uniform
        # one (every point inside carved is inside uniform since dists are
        # only reduced).
        # Same bbox? Both polyhedra share the center; uniform bbox is
        # bigger. Skip the strict-subset check across different bboxes.


class TestInsidePolyhedronCore:
    def test_origin_always_inside(self, rays96, faces96):
        # Center voxel (origin of coords) is by construction inside.
        dists = np.full(96, 5.0, dtype=np.float64)
        coords = np.array([[0.0, 0.0, 0.0]])
        inside = _inside_polyhedron(coords, rays96, dists, faces96)
        assert inside.tolist() == [True]

    def test_far_outside_excluded(self, rays96, faces96):
        # Voxel at 100 units from origin must be rejected for any
        # plausible polyhedron with dists ≤ 5.
        dists = np.full(96, 5.0, dtype=np.float64)
        coords = np.array([[50.0, 50.0, 50.0]])
        inside = _inside_polyhedron(coords, rays96, dists, faces96)
        assert inside.tolist() == [False]

    def test_along_ray_at_dist(self, rays96, faces96):
        # A voxel placed exactly at d·ray[i] for the largest ray should
        # be on the boundary (inside up to numerical tolerance).
        dists = np.full(96, 7.0, dtype=np.float64)
        coords = (dists[:, None] * rays96).astype(np.float64) * 0.99
        inside = _inside_polyhedron(coords, rays96, dists, faces96)
        # Slightly-shrunk points along every ray should all be inside.
        assert inside.all()

    def test_empty_inputs(self, rays96, faces96):
        coords = np.zeros((0, 3))
        out = _inside_polyhedron(coords, rays96, np.zeros(96), faces96)
        assert out.shape == (0,)


class TestNanDistsAreSanitised:
    def test_nan_dists_do_not_crash(self, rays96, faces96):
        center = np.array([20, 20, 20], dtype=np.float64)
        dists = np.full(96, 5.0)
        dists[3] = np.nan
        dists[7] = np.inf
        # Should not raise; the rasteriser zero-fills the bogus rays.
        _, mask = _rasterize_to_bbox(center, rays96, dists, (40, 40, 40), faces96)
        assert mask.any()
