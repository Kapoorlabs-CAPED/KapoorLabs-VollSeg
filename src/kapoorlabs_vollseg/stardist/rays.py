"""Ray geometry for StarDist.

Rays are the unit-vector directions along which per-pixel distance to the
object boundary is predicted. 2D uses uniformly-spaced angles; 3D uses
the golden-spiral construction (same parameterization as upstream
``stardist.Rays_GoldenSpiral``), generalized for axis anisotropy.

The 3D variant also returns triangulated **faces** (via
:class:`scipy.spatial.ConvexHull` of the unit-sphere points) — those
faces define the actual star-convex polyhedron the inference code
rasterises, instead of the looser "nearest-ray cone" approximation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull


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

    Matches upstream :class:`stardist.Rays_GoldenSpiral` exactly so weights
    trained against the keras pipeline are directly transferable: the
    ``i``-th ray here is the ``i``-th ray there, and ``anisotropy`` is
    handled with the same sign convention.

    Parameters
    ----------
    n_rays
        Number of rays to generate (must be ≥ 4).
    anisotropy
        Optional ``(z, y, x)`` voxel spacing. Upstream stardist treats
        this as physical spacing and **divides** the unit-sphere
        vertices by it before re-normalising — that biases ray density
        toward the axes with finer spacing (typically y/x for confocal),
        which is the correct convention for non-isotropic 3D microscopy.

    Returned axis order: ``(z, y, x)``.
    """
    if n_rays < 4:
        raise ValueError(f"n_rays must be >= 4, got {n_rays}")

    g = (3.0 - np.sqrt(5.0)) * np.pi  # golden angle, same as upstream
    phi = g * np.arange(n_rays)
    z = np.linspace(-1.0, 1.0, n_rays)
    rho = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    rays = np.stack([z, rho * np.sin(phi), rho * np.cos(phi)], axis=1)

    if anisotropy is not None:
        anisotropy = np.asarray(anisotropy, dtype=np.float64)
        if anisotropy.shape != (3,):
            raise ValueError(
                f"anisotropy must be a 3-tuple, got {tuple(anisotropy.shape)}"
            )
        if np.any(anisotropy <= 0):
            raise ValueError(f"anisotropy entries must be > 0, got {anisotropy}")
        rays = rays / anisotropy  # upstream convention — divide, don't multiply

    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    return rays


def compute_faces(rays: np.ndarray) -> np.ndarray:
    """Return a ``(n_faces, 3)`` triangle index array for the polyhedron
    spanned by ``rays``.

    Faces come from :class:`scipy.spatial.ConvexHull` on the unit-sphere
    rays (so the triangulation is identical to upstream stardist's:
    :class:`stardist.Rays_GoldenSpiral` uses the same ConvexHull call).
    Each face is three indices into ``rays``; the three corresponding
    rays form a spherical triangle on the unit sphere, and after
    per-peak distance scaling they form one triangular face of the
    star-convex polyhedron the inference code rasterises.

    Only meaningful for 3D rays. For 2D ray arrays this returns an
    empty ``(0, 3)`` array (caller stays on the polygon fast-path).
    """
    if rays.ndim != 2 or rays.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.int64)
    hull = ConvexHull(np.asarray(rays, dtype=np.float64))
    return np.asarray(hull.simplices, dtype=np.int64)
