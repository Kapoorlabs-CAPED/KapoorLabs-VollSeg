"""Turn an instance label image into StarDist training targets.

Two outputs per label image:

- :func:`foreground_probability_map` — distance to nearest background,
  normalized per object (the standard StarDist object-probability target).
- :func:`compute_distance_map` — per-pixel distance to the object
  boundary along each of ``n_rays`` ray directions, ``shape = (n_rays, *spatial)``.

Background = 0 in the input label image. The ray-march kernels are
numba-jitted *if* numba is installed (≈50× speedup); otherwise pure
NumPy is used as a transparent fallback.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops


# ---------------------------------------------------------------- prob target

def foreground_probability_map(labels: np.ndarray) -> np.ndarray:
    """EDT-normalized object-probability target — peak 1 at object centers, 0 at background."""
    if labels.dtype not in (np.int32, np.uint16, np.uint32):
        labels = labels.astype(np.int32)

    out = np.zeros(labels.shape, dtype=np.float32)
    if labels.max() == 0:
        return out

    for prop in regionprops(labels):
        mask = labels == prop.label
        edt = distance_transform_edt(mask)
        peak = edt.max()
        if peak > 0:
            out[mask] = (edt[mask] / peak).astype(np.float32)
    return out


# ---------------------------------------------------------------- dist target

def compute_distance_map(
    labels: np.ndarray,
    rays: np.ndarray,
    *,
    max_dist: Optional[int] = None,
) -> np.ndarray:
    """Per-pixel distance to the object boundary along each ray direction."""
    if labels.ndim not in (2, 3):
        raise ValueError(f"labels must be 2D or 3D, got ndim={labels.ndim}")
    if rays.ndim != 2 or rays.shape[1] != labels.ndim:
        raise ValueError(f"rays must have shape (N, {labels.ndim}), got {rays.shape}")
    if max_dist is None:
        max_dist = int(np.ceil(np.linalg.norm(labels.shape)))

    # Numba kernels expect contiguous int32 / float32 / int32 inputs.
    labels32 = np.ascontiguousarray(labels, dtype=np.int32)
    rays32 = np.ascontiguousarray(rays, dtype=np.float32)

    n_rays = rays32.shape[0]
    out = np.zeros((n_rays,) + labels.shape, dtype=np.float32)
    fg_mask = labels32 > 0
    if not fg_mask.any():
        return out

    fg_coords = np.argwhere(fg_mask).astype(np.int32)
    fg_label_ids = labels32[fg_mask].astype(np.int32)

    if labels.ndim == 2:
        _march_2d_kernel(out, fg_coords, fg_label_ids, labels32, rays32, np.int32(max_dist))
    else:
        _march_3d_kernel(out, fg_coords, fg_label_ids, labels32, rays32, np.int32(max_dist))
    return out


# ---------------------------------------------------------------- ray-march kernels

def _march_2d_kernel(out, fg_coords, fg_label_ids, labels, rays, max_dist):
    n_rays = rays.shape[0]
    n_fg = fg_coords.shape[0]
    H = labels.shape[0]
    W = labels.shape[1]
    for r_idx in range(n_rays):
        dy = rays[r_idx, 0]
        dx = rays[r_idx, 1]
        for i in range(n_fg):
            y0 = fg_coords[i, 0]
            x0 = fg_coords[i, 1]
            label = fg_label_ids[i]
            d = 0
            y_f = float(y0)
            x_f = float(x0)
            while d < max_dist:
                d += 1
                y_f = y_f + dy
                x_f = x_f + dx
                yi = int(round(y_f))
                xi = int(round(x_f))
                if yi < 0 or yi >= H or xi < 0 or xi >= W:
                    break
                if labels[yi, xi] != label:
                    break
            out[r_idx, y0, x0] = float(d)


def _march_3d_kernel(out, fg_coords, fg_label_ids, labels, rays, max_dist):
    n_rays = rays.shape[0]
    n_fg = fg_coords.shape[0]
    Z = labels.shape[0]
    Y = labels.shape[1]
    X = labels.shape[2]
    for r_idx in range(n_rays):
        dz = rays[r_idx, 0]
        dy = rays[r_idx, 1]
        dx = rays[r_idx, 2]
        for i in range(n_fg):
            z0 = fg_coords[i, 0]
            y0 = fg_coords[i, 1]
            x0 = fg_coords[i, 2]
            label = fg_label_ids[i]
            d = 0
            z_f = float(z0)
            y_f = float(y0)
            x_f = float(x0)
            while d < max_dist:
                d += 1
                z_f = z_f + dz
                y_f = y_f + dy
                x_f = x_f + dx
                zi = int(round(z_f))
                yi = int(round(y_f))
                xi = int(round(x_f))
                if zi < 0 or zi >= Z or yi < 0 or yi >= Y or xi < 0 or xi >= X:
                    break
                if labels[zi, yi, xi] != label:
                    break
            out[r_idx, z0, y0, x0] = float(d)


# Numba JIT if available — kernel signatures match int32/float32 inputs.
try:
    from numba import njit
    _march_2d_kernel = njit(cache=True, fastmath=True)(_march_2d_kernel)
    _march_3d_kernel = njit(cache=True, fastmath=True)(_march_3d_kernel)
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
