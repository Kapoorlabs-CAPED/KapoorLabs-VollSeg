"""Ray geometry for StarDist.

Rays are the unit-vector directions along which per-pixel distance to the
object boundary is predicted. 2D uses uniformly-spaced angles; 3D uses
the golden-spiral construction (same algorithm as upstream stardist's
:func:`Rays_GoldenSpiral`), generalized for axis anisotropy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def rays_2d(n_rays: int) -> np.ndarray:
    """Return a ``(n_rays, 2)`` array of unit-length 2D ray directions.

    Directions are equally spaced over ``[0, 2π)`` starting from angle 0.
    Returned axis order is ``(y, x)`` — matches numpy image indexing.
    """
    if n_rays < 3:
        raise ValueError(f"n_rays must be >= 3, got {n_rays}")
    theta = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    return np.stack([np.sin(theta), np.cos(theta)], axis=1)  # (N, 2): (dy, dx)


def rays_3d_golden_spiral(
    n_rays: int,
    anisotropy: Optional[tuple[float, float, float]] = None,
) -> np.ndarray:
    """Return a ``(n_rays, 3)`` array of unit-length 3D ray directions.

    Uses the Fibonacci / golden-spiral parameterization so the points are
    nearly-uniformly distributed on the unit sphere.

    Parameters
    ----------
    n_rays
        Number of rays to generate.
    anisotropy
        Optional ``(z, y, x)`` voxel spacing scale. When set, the rays are
        scaled by this factor and re-normalized — preferred for training
        on non-isotropic 3D microscopy data so distances are meaningful in
        the network's native voxel space.

    Returned axis order: ``(z, y, x)``.
    """
    if n_rays < 4:
        raise ValueError(f"n_rays must be >= 4, got {n_rays}")

    indices = np.arange(n_rays, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    z = 1.0 - 2.0 * indices / (n_rays - 1)  # z ∈ [-1, 1]
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = phi * indices
    y = np.sin(theta) * radius
    x = np.cos(theta) * radius

    rays = np.stack([z, y, x], axis=1)  # (N, 3): (dz, dy, dx)

    if anisotropy is not None:
        anisotropy = np.asarray(anisotropy, dtype=np.float64)
        if anisotropy.shape != (3,):
            raise ValueError(
                f"anisotropy must be a 3-tuple, got {tuple(anisotropy.shape)}"
            )
        if np.any(anisotropy <= 0):
            raise ValueError(f"anisotropy entries must be > 0, got {anisotropy}")
        rays = rays * anisotropy
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)

    return rays
