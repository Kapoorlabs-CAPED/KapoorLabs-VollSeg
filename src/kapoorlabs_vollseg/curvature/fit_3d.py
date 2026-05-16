"""Algebraic sphere fit to 3D points (Coope linear least-squares).

Expands ``(x - a)² + (y - b)² + (z - c)² = r²`` to the linear form
``x² + y² + z² + Dx + Ey + Fz + G = 0`` and solves for ``[D, E, F, G]``
via NumPy lstsq. O(N) per fit; correct for any non-coplanar 4+ points.
"""

from __future__ import annotations

import numpy as np


def fit_sphere_3d(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a sphere to ``(N, 3)`` points; return ``(centre[3], radius)``.

    Returns radius 0 when the points are coplanar (the linear system is
    rank-deficient and ``a² + b² + c² - G`` goes negative) — the caller
    should treat those windows as degenerate and drop them.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got shape {points.shape}")
    if points.shape[0] < 4:
        raise ValueError(f"Need ≥ 4 points for a sphere fit, got {points.shape[0]}")

    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    z = points[:, 2].astype(np.float64)
    a_mat = np.column_stack([x, y, z, np.ones_like(x)])
    rhs = -(x**2 + y**2 + z**2)
    coef, *_ = np.linalg.lstsq(a_mat, rhs, rcond=None)
    d, e, f, g = coef
    cx, cy, cz = -d / 2.0, -e / 2.0, -f / 2.0
    r2 = cx * cx + cy * cy + cz * cz - g
    radius = float(np.sqrt(r2)) if r2 > 0 else 0.0
    return np.array([cx, cy, cz], dtype=np.float64), radius
