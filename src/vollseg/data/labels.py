"""Label-image morphology: scale, erode, fill, binary↔instance conversion.

These operate on integer label images where ``0`` is background and each
positive value is a unique instance ID. Care is taken everywhere to avoid
accidentally blending labels (e.g. by interpolation > 0).
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes, find_objects, zoom
from skimage.measure import label as cc_label
from skimage.measure import regionprops


# ------------------------------------------------------------ binary <-> labels

def binary_to_labels(binary: np.ndarray) -> np.ndarray:
    """Connected-components label a binary mask → uint16."""
    return cc_label(binary.astype(bool)).astype(np.uint16)


def labels_to_binary(labels: np.ndarray) -> np.ndarray:
    """Collapse instance labels → boolean foreground mask."""
    return labels.astype(bool)


# ----------------------------------------------------------------- erosion

def erode_labels(labels: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Erode each instance independently so they don't get fused.

    Pixels lost to erosion are set to background (0). Each instance keeps
    its original ID.
    """
    if iterations <= 0:
        return labels
    out = np.zeros_like(labels)
    for prop in regionprops(labels):
        lid = prop.label
        mask = labels == lid
        eroded = binary_erosion(mask, iterations=iterations)
        out[eroded] = lid
    return out


# ------------------------------------------------------------ hole filling

def fill_label_holes(labels: np.ndarray, **kwargs) -> np.ndarray:
    """Fill internal holes within each instance, per-instance.

    Borrowed from the StarDist tooling: walks each ``find_objects`` slice
    so that holes spanning the image edge are still correctly handled.
    """
    out = np.zeros_like(labels)

    def _grow(sl, interior):
        return tuple(
            slice(s.start - int(w[0]), s.stop + int(w[1]))
            for s, w in zip(sl, interior)
        )

    def _shrink(interior):
        return tuple(slice(int(w[0]), -1 if w[1] else None) for w in interior)

    objects = find_objects(labels)
    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, labels.shape)]
        grown = labels[_grow(sl, interior)] == i
        filled = binary_fill_holes(grown, **kwargs)[_shrink(interior)]
        out[sl][filled] = i
    return out


# ----------------------------------------------------------------- scaling

def scale_labels(
    labels: np.ndarray,
    scale_factors: Union[Tuple[float, ...], float],
) -> np.ndarray:
    """Resize an integer label image with nearest-neighbor (no blending).

    Accepts a scalar (applied to all spatial axes) or a per-axis tuple.
    For 4D ``(Z, C, Y, X)`` arrays the channel axis is held fixed.
    """
    if labels.size == 0:
        raise ValueError("Input label array is empty")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"Labels must be integer type, got {labels.dtype}")

    if np.isscalar(scale_factors):
        if labels.ndim == 3:
            zf = (scale_factors,) * 3
        elif labels.ndim == 4:
            zf = (scale_factors, 1.0, scale_factors, scale_factors)
        else:
            raise ValueError(f"Unsupported ndim={labels.ndim}")
    else:
        scale_factors = tuple(scale_factors)
        if len(scale_factors) != 3:
            raise ValueError("scale_factors must have 3 elements (Z, Y, X)")
        if labels.ndim == 3:
            zf = scale_factors
        elif labels.ndim == 4:
            zf = (scale_factors[0], 1.0, scale_factors[1], scale_factors[2])
        else:
            raise ValueError(f"Unsupported ndim={labels.ndim}")

    if any(f <= 0 for f in zf):
        raise ValueError("All scale factors must be > 0")
    if all(abs(f - 1.0) < 1e-6 for f in zf):
        return labels

    return zoom(labels, zf, order=0).astype(labels.dtype)


def upscale_labels(labels: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Resize labels to exactly ``target_shape`` (nearest-neighbor + crop/pad)."""
    if len(target_shape) != labels.ndim:
        raise ValueError(
            f"target_shape ndim ({len(target_shape)}) != labels ndim ({labels.ndim})"
        )
    if labels.ndim == 4 and target_shape[1] != labels.shape[1]:
        raise ValueError("4D channel dim must be preserved")

    factors = tuple(t / s for t, s in zip(target_shape, labels.shape))
    out = zoom(labels, factors, order=0)

    if out.shape != tuple(target_shape):
        slices = tuple(slice(0, min(o, t)) for o, t in zip(out.shape, target_shape))
        out = out[slices]
        if out.shape != tuple(target_shape):
            pad = [(0, t - o) for o, t in zip(out.shape, target_shape)]
            out = np.pad(out, pad, mode="constant", constant_values=0)
    return out.astype(labels.dtype)
