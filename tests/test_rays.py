"""Tests for kapoorlabs_vollseg.stardist.rays — geometry of 2D angles,
3D golden-spiral, and the ConvexHull triangulation used to define the
polyhedron-rasterisation faces.
"""

from __future__ import annotations

import numpy as np
import pytest

from kapoorlabs_vollseg.stardist import (
    compute_faces,
    rays_2d,
    rays_3d_golden_spiral,
)


class TestRays2D:
    def test_shape(self):
        r = rays_2d(8)
        assert r.shape == (8, 2)

    def test_unit_length(self):
        r = rays_2d(96)
        norms = np.linalg.norm(r, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_uniform_spacing(self):
        # Adjacent rays separated by 2π/N — same dot product across the ring.
        r = rays_2d(32)
        dots = (r[:-1] * r[1:]).sum(axis=1)
        np.testing.assert_allclose(dots, dots[0], atol=1e-6)

    def test_first_ray_is_dy0_dx1(self):
        # rays_2d(N) starts at angle 0 → (sin 0, cos 0) = (0, 1)
        r = rays_2d(8)
        np.testing.assert_allclose(r[0], [0.0, 1.0], atol=1e-12)

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            rays_2d(2)


class TestRays3DGoldenSpiral:
    def test_shape(self):
        r = rays_3d_golden_spiral(64)
        assert r.shape == (64, 3)

    def test_unit_length_isotropic(self):
        r = rays_3d_golden_spiral(96)
        norms = np.linalg.norm(r, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_unit_length_anisotropic(self):
        # When anisotropy is applied, rays are renormalized → still unit length.
        r = rays_3d_golden_spiral(96, anisotropy=(2.0, 1.0, 1.0))
        norms = np.linalg.norm(r, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_anisotropy_division_convention(self):
        # We match upstream stardist: anisotropy DIVIDES the unit-sphere
        # vertices before re-normalisation. With aniso=(3,1,1), the z
        # component is divided by 3 first, so after re-normalisation the
        # mean |z| is *smaller* than isotropic — rays tilt away from z.
        iso = rays_3d_golden_spiral(96)
        ani = rays_3d_golden_spiral(96, anisotropy=(3.0, 1.0, 1.0))
        assert np.mean(np.abs(ani[:, 0])) < np.mean(np.abs(iso[:, 0]))

    def test_first_and_last_match_upstream(self):
        # Our golden-spiral uses z = linspace(-1, 1, n) — the i=0 ray
        # should be (-1, 0, 0) and the i=n-1 ray should be (+1, 0, ~0).
        r = rays_3d_golden_spiral(96)
        np.testing.assert_allclose(r[0], [-1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(r[-1, 0], 1.0, atol=1e-6)
        # the (y, x) at the +z pole rotate with phi*(n-1) — just verify
        # the radial component vanishes.
        assert abs(np.hypot(r[-1, 1], r[-1, 2])) < 1e-6

    def test_invalid_anisotropy(self):
        with pytest.raises(ValueError):
            rays_3d_golden_spiral(96, anisotropy=(0.0, 1.0, 1.0))
        with pytest.raises(ValueError):
            rays_3d_golden_spiral(96, anisotropy=(1.0, 1.0))

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            rays_3d_golden_spiral(3)


class TestUpstreamStardistParity:
    """Verify ray geometry matches upstream stardist.Rays_GoldenSpiral exactly.

    The model checkpoint can only be loaded against the rays it was
    trained on; matching upstream means PyTorch-trained weights can in
    principle be cross-validated against keras-trained `.h5` files.
    """

    def test_isotropic_matches_upstream(self):
        stardist = pytest.importorskip("stardist")
        ours = rays_3d_golden_spiral(96).astype(np.float64)
        theirs = stardist.rays3d.Rays_GoldenSpiral(n=96).vertices.astype(np.float64)
        np.testing.assert_allclose(ours, theirs, atol=1e-6)

    def test_anisotropic_matches_upstream(self):
        stardist = pytest.importorskip("stardist")
        aniso = (2.0, 1.0, 1.0)
        ours = rays_3d_golden_spiral(96, anisotropy=aniso).astype(np.float64)
        theirs = stardist.rays3d.Rays_GoldenSpiral(
            n=96, anisotropy=aniso
        ).vertices.astype(np.float64)
        np.testing.assert_allclose(ours, theirs, atol=1e-6)


class TestComputeFaces:
    def test_3d_face_count_topology(self):
        # ConvexHull of n unit-sphere points produces 2n - 4 triangles
        # (Euler characteristic for a sphere).
        for n in (24, 64, 96):
            rays = rays_3d_golden_spiral(n)
            faces = compute_faces(rays)
            assert faces.shape == (2 * n - 4, 3)

    def test_face_indices_in_range(self):
        rays = rays_3d_golden_spiral(64)
        faces = compute_faces(rays)
        assert faces.min() >= 0
        assert faces.max() < rays.shape[0]

    def test_face_vertices_unique(self):
        # No triangle should have a repeated vertex.
        faces = compute_faces(rays_3d_golden_spiral(64))
        for tri in faces:
            assert len(set(tri.tolist())) == 3

    def test_2d_rays_no_faces(self):
        # ConvexHull-based faces are 3D-only; 2D rays get an empty array.
        faces = compute_faces(rays_2d(32))
        assert faces.shape == (0, 3)

    def test_faces_cover_each_ray(self):
        # For a closed triangulation of the sphere, every vertex must be
        # touched by at least one face.
        n = 48
        rays = rays_3d_golden_spiral(n)
        faces = compute_faces(rays)
        assert set(faces.ravel().tolist()) == set(range(n))
