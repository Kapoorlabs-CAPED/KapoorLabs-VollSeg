"""Watershed-based fusion of StarDist instances + U-Net semantic mask.

This is the seedpool VollSeg algorithm in one place, dispatched on ``ndim``:

1. Take every StarDist instance whose centroid sits *outside* every U-Net
   seed bounding box, and burn it into the mask (these are confident
   detections the U-Net missed).
2. If ``seedpool=True``, also take every U-Net seed whose centroid sits
   *outside* every StarDist instance bounding box, and add it as a marker
   (these are detections StarDist missed).
3. Run a marker-controlled watershed on ``-image`` constrained by the mask.

Ported from ``utils.py:WatershedwithMask3D`` / ``SuperWatershedwithMask``
in the original VollSeg.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import measure, morphology
from skimage.filters import threshold_otsu
from skimage.morphology import label
from skimage.segmentation import find_boundaries, relabel_sequential, watershed

from .seedpool import SeedPool, UnetStarMask


def watershed_fuse(
    image: np.ndarray,
    stardist_labels: np.ndarray,
    unet_mask: np.ndarray,
    *,
    seedpool: bool = True,
) -> np.ndarray:
    """Fuse StarDist instances with a U-Net semantic mask via watershed.

    Parameters
    ----------
    image
        Original (or denoised) intensity image — drives the watershed gradient.
    stardist_labels
        Instance label image from StarDist (same shape as ``image``).
    unet_mask
        Boolean / 0-1 semantic mask from the U-Net (same shape as ``image``).
    seedpool
        If True, add U-Net-only seeds as additional watershed markers.
        If False, the watershed is seeded by StarDist centroids only.

    Returns
    -------
    np.ndarray
        Fused instance label image, same shape as ``image``.
    """
    if image.ndim not in (2, 3):
        raise ValueError(
            f"watershed_fuse expects 2D or 3D image, got ndim={image.ndim}"
        )

    mask = unet_mask.astype(bool).copy()
    star_props = measure.regionprops(stardist_labels)
    star_centroids = [p.centroid for p in star_props]
    star_bboxes = [p.bbox for p in star_props]
    star_label_ids = [p.label for p in star_props]

    binary_props = measure.regionprops(label(mask))
    unet_centroids = [p.centroid for p in binary_props]

    # Step 1: burn confident StarDist instances into the mask.
    for box, lbl in zip(star_bboxes, star_label_ids):
        outside_all_unet = all(UnetStarMask(box, c).masking() for c in unet_centroids)
        if outside_all_unet:
            mask[stardist_labels == lbl] = True

    # Recompute U-Net seeds after the mask edit.
    binary_props = measure.regionprops(label(mask))
    unet_centroids = [p.centroid for p in binary_props]
    unet_bboxes = [p.bbox for p in binary_props]

    # Sort StarDist centroids deterministically before extending.
    centroids = sorted(star_centroids)

    # Step 2: optionally pool in U-Net-only seeds.
    if seedpool:
        for box, c in zip(unet_bboxes, unet_centroids):
            outside_all_star = all(SeedPool(box, s).pooling() for s in centroids)
            if outside_all_star:
                centroids.append(c)

    # Background sentinel marker so watershed has a "0" region.
    centroids.append((0,) * image.ndim)
    coords = np.round(np.asarray(centroids)).astype(int)

    markers_raw = np.zeros_like(image, dtype=np.int32)
    markers_raw[tuple(coords.T)] = 1 + np.arange(len(centroids))

    selem = morphology.ball(2) if image.ndim == 3 else morphology.disk(2)
    markers = morphology.dilation(markers_raw.astype(np.uint16), selem)

    return watershed(-image, markers, mask=mask)


def cellpose_watershed_fuse(
    membrane_image: np.ndarray,
    nuclei_labels: np.ndarray,
    cellpose_mask: np.ndarray,
) -> np.ndarray:
    """Watershed membrane image, seeded by nuclei centroids, gated by CellPose mask.

    This is the membrane-side counterpart to :func:`watershed_fuse`. It
    takes already-segmented nuclei (from any nuclei pipeline), uses their
    centroids as watershed markers, and grows them across the membrane
    image's boundary signal — staying inside the (lightly cleaned-up)
    CellPose mask. Labels that touch the eroded interior of the mask are
    suppressed as spurious membrane-thickness blobs.

    Ported from ``utils.py:CellPoseWater`` in the original VollSeg.
    """
    mask = cellpose_mask.astype(bool)
    if mask.ndim == 2 and membrane_image.ndim == 3:
        mask = np.repeat(mask[np.newaxis, :, :], membrane_image.shape[0], axis=0)

    # Light cleanup of the mask boundary.
    mask = binary_erosion(mask, iterations=1)
    mask = binary_dilation(mask, iterations=1)

    # Build markers from nuclei centroids.
    centroids = [p.centroid for p in measure.regionprops(nuclei_labels)]
    centroids.append((0,) * membrane_image.ndim)  # background sentinel
    coords = np.round(np.asarray(centroids)).astype(int)
    markers_raw = np.zeros_like(nuclei_labels)
    markers_raw[tuple(coords.T)] = 1 + np.arange(len(centroids))

    selem = morphology.ball(2) if membrane_image.ndim == 3 else morphology.disk(2)
    markers = morphology.dilation(markers_raw.astype(np.uint16), selem)

    # Watershed against the membrane-boundary signal, constrained by the mask.
    thresh = threshold_otsu(membrane_image)
    binary = membrane_image > thresh
    boundary = find_boundaries(binary, mode="outer") * 255
    inner = binary_erosion(binary, iterations=2)

    result = watershed(boundary, markers, mask=mask)
    result, _, _ = relabel_sequential(result.astype(np.uint16))

    # Suppress labels that lie inside the eroded interior — these are
    # bulk objects rather than membrane-thickness compartments.
    bad = np.unique(result[inner > 0])
    for lid in bad:
        if lid != 0:
            result[result == lid] = 0
    return result
