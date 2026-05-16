"""Algebraic circle fit to 2D points (Kasa method).

Solves ``(x - a)² + (y - b)² = r²`` by expanding to the linear form
``x² + y² + Dx + Ey + F = 0`` and solving via least-squares for
``[D, E, F]``. Fast (one SVD) and accurate enough for sliding-window
curvature with ``n_window ~ 20`` boundary points.

Replace with Pratt or Taubin if you find Kasa biases bother you on
very noisy / sparse data; the rest of the package only needs the
``(centre, radius)`` contract.
"""

from __future__ import annotations

import numpy as np


def fit_circle_2d(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a circle to ``(N, 2)`` points; return ``(centre[2], radius)``.

    For collinear (or near-collinear) inputs the system is rank-deficient
    and the fitted radius collapses to 0 — the caller should check and
    discard those windows.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be (N, 2), got shape {points.shape}")
    if points.shape[0] < 3:
        raise ValueError(f"Need ≥ 3 points for a circle fit, got {points.shape[0]}")

    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    a_mat = np.column_stack([x, y, np.ones_like(x)])
    rhs = -(x**2 + y**2)
    coef, *_ = np.linalg.lstsq(a_mat, rhs, rcond=None)
    d, e, f = coef
    cx, cy = -d / 2.0, -e / 2.0
    r2 = cx * cx + cy * cy - f
    radius = float(np.sqrt(r2)) if r2 > 0 else 0.0
    return np.array([cx, cy], dtype=np.float64), radius
