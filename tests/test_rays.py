"""Tests for kapoorlabs_vollseg.stardist.rays — geometry of 2D angles and 3D golden-spiral."""

from __future__ import annotations

import numpy as np
import pytest

from kapoorlabs_vollseg.stardist import rays_2d, rays_3d_golden_spiral


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

    def test_anisotropy_skews_z(self):
        # Anisotropy stretches the ray vectors before re-normalization, so
        # the new rays should have *more* of their length on the stretched
        # axis on average than the isotropic version.
        iso = rays_3d_golden_spiral(96)
        ani = rays_3d_golden_spiral(96, anisotropy=(3.0, 1.0, 1.0))
        # Mean absolute z-component is larger for the anisotropic set.
        assert np.mean(np.abs(ani[:, 0])) > np.mean(np.abs(iso[:, 0]))

    def test_invalid_anisotropy(self):
        with pytest.raises(ValueError):
            rays_3d_golden_spiral(96, anisotropy=(0.0, 1.0, 1.0))
        with pytest.raises(ValueError):
            rays_3d_golden_spiral(96, anisotropy=(1.0, 1.0))

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            rays_3d_golden_spiral(3)
