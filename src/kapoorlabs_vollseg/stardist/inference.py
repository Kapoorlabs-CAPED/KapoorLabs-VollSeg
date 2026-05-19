"""End-to-end StarDist inference: predict → stitch → peaks → NMS → label image.

Algorithm (works for both 2D and 3D, dispatched on ``rays.shape[1]``):

1. **Tile** the input volume via :class:`kapoorlabs_vollseg._lightning.CarePredictionDataset`.
2. **Predict** ``prob`` (sigmoid) and ``dist`` (raw distances) per tile,
   then **stitch** via linear-blend overlap (same kernel CARE uses).
3. **Detect peaks** as local maxima of the prob map above ``prob_thresh``.
4. **Rasterize** each peak's predicted star-polyhedron into a small
   bounding-box mask.
5. **NMS** greedily keeps high-prob peaks whose rasterized mask doesn't
   overlap any already-kept peak above ``nms_thresh``.
6. **Paint** the survivors into a uint16 label image.

The rasterizer uses the "nearest-ray" approximation: for each candidate
voxel, find the ray whose direction is closest (largest dot product with
the unit vector from the center to the voxel) and accept the voxel iff
its distance from the center is at most that ray's predicted length.
For ~64+ rays this is a good approximation of the true star-polyhedron;
for very few rays a multi-ray interpolation would be more accurate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from skimage.feature import peak_local_max
from torch.utils.data import DataLoader

from .._lightning.dataset import CarePredictionDataset, compute_tile_shape
from .._lightning.transforms import PercentileNormalize
from .lightning_module import StarDistModule


# ============================================================== top-level API


@dataclass
class StarDistResult:
    labels: np.ndarray  # (*spatial) uint16 instance labels
    prob_map: np.ndarray  # (*spatial) float32 stitched probability
    n_objects: int


def predict_volume(
    model: StarDistModule,
    image: np.ndarray,
    rays: np.ndarray,
    *,
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.4,
    min_distance: int = 2,
    n_tiles: Optional[tuple[int, ...]] = None,
    tile_overlap: float = 0.125,
    batch_size: int = 4,
    num_workers: int = 0,
    pmin: Optional[float] = 0.1,
    pmax: Optional[float] = 99.9,
    device: Optional[str] = None,
) -> StarDistResult:
    """Run StarDist inference end-to-end on a single 3D (or 2D) volume.

    Parameters
    ----------
    model
        A trained :class:`StarDistModule`.
    image
        Input volume; ``ndim`` must equal ``rays.shape[1]``.
    rays
        ``(n_rays, ndim)`` ray geometry — must match what the model was
        trained on (same ``n_rays``, same anisotropy if 3D).
    prob_thresh
        Minimum object probability for a peak to be considered.
    nms_thresh
        IoU threshold (on rasterized polyhedra) above which two peaks
        are considered duplicates and the lower-prob one is dropped.
    min_distance
        Minimum spacing (in voxels) between peaks at the detection step
        (forwarded to :func:`skimage.feature.peak_local_max`).
    n_tiles
        Per-axis tile count for inference; defaults match
        :class:`kapoorlabs_vollseg.CAREDenoiser` (``[1, 4, 4]`` for 3D).
    tile_overlap, batch_size, num_workers, pmin, pmax, device
        As in :class:`kapoorlabs_vollseg.CAREDenoiser`.
    """
    if image.ndim != rays.shape[1]:
        raise ValueError(
            f"image.ndim={image.ndim} must equal rays.shape[1]={rays.shape[1]}"
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 1+2. Tile + predict + stitch.
    prob_map, dist_map = _predict_and_stitch(
        model,
        image,
        n_tiles=n_tiles,
        tile_overlap=tile_overlap,
        batch_size=batch_size,
        num_workers=num_workers,
        pmin=pmin,
        pmax=pmax,
        device=device,
    )

    labels = nms_to_labels(
        prob_map,
        dist_map,
        rays,
        image.shape,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        min_distance=min_distance,
    )
    return StarDistResult(
        labels=labels,
        prob_map=prob_map,
        n_objects=int(labels.max()),
    )


def nms_to_labels(
    prob_map: np.ndarray,
    dist_map: np.ndarray,
    rays: np.ndarray,
    vol_shape: tuple,
    *,
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.4,
    min_distance: int = 2,
) -> np.ndarray:
    """Peak-detect → rasterise → NMS → label image.

    Pulled out of :func:`predict_volume` so the threshold optimiser can
    cache the (slow) ``_predict_and_stitch`` network forward and only
    rerun this (fast) step per ``(prob_thresh, nms_thresh)`` candidate.

    Same contract as the corresponding block inside ``predict_volume``;
    empty inputs return a zero label image.
    """
    centers = peak_local_max(
        prob_map,
        threshold_abs=prob_thresh,
        min_distance=min_distance,
        exclude_border=False,
    )
    if centers.size == 0:
        return np.zeros(vol_shape, dtype=np.uint16)

    scores = prob_map[tuple(centers.T)]
    dists = np.stack(
        [dist_map[(slice(None),) + tuple(c)] for c in centers], axis=0
    )  # (M, n_rays)

    _, kept_bbox, kept_masks = _nms_polyhedra(
        centers=centers,
        dists=dists,
        scores=scores,
        rays=rays,
        vol_shape=vol_shape,
        iou_thresh=nms_thresh,
    )
    return _paint_labels(vol_shape, kept_bbox, kept_masks)


# ============================================================= internals


def _predict_and_stitch(
    model,
    image,
    *,
    n_tiles,
    tile_overlap,
    batch_size,
    num_workers,
    pmin,
    pmax,
    device,
) -> tuple[np.ndarray, np.ndarray]:
    n = tuple(n_tiles) if n_tiles is not None else tuple(model.n_tiles)
    tile_shape = compute_tile_shape(image.shape, n)
    normalizer = (
        PercentileNormalize(pmin=pmin, pmax=pmax)
        if pmin is not None and pmax is not None
        else None
    )

    dataset = CarePredictionDataset(
        volume=image.astype(np.float32),
        tile_shape=tile_shape,
        overlap=tile_overlap,
        normalizer=normalizer,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    prob_acc = np.zeros(image.shape, dtype=np.float32)
    dist_acc = np.zeros((model.n_rays, *image.shape), dtype=np.float32)
    weight = np.zeros(image.shape, dtype=np.float32)

    with torch.no_grad():
        for tiles, coords in loader:
            tiles = tiles.to(device)
            prob, dists, coords_out = model.predict_step((tiles, coords), batch_idx=0)
            prob = prob.numpy()  # (B, 1, *spatial)
            dists = dists.numpy()  # (B, N, *spatial)
            for i in range(prob.shape[0]):
                zs, ys, xs, tz, ty, tx = (int(v) for v in coords_out[i].tolist())
                w = _make_blend_weight(image.ndim, (tz, ty, tx), tile_overlap)
                if image.ndim == 2:
                    prob_acc[ys : ys + ty, xs : xs + tx] += prob[i, 0] * w
                    dist_acc[:, ys : ys + ty, xs : xs + tx] += dists[i] * w[None]
                    weight[ys : ys + ty, xs : xs + tx] += w
                else:
                    prob_acc[zs : zs + tz, ys : ys + ty, xs : xs + tx] += prob[i, 0] * w
                    dist_acc[:, zs : zs + tz, ys : ys + ty, xs : xs + tx] += (
                        dists[i] * w[None]
                    )
                    weight[zs : zs + tz, ys : ys + ty, xs : xs + tx] += w

    mask = weight > 0
    prob_acc[mask] /= weight[mask]
    dist_acc[:, mask] /= weight[mask]  # broadcast over channel axis
    return prob_acc, dist_acc


def _make_blend_weight(ndim, tile_shape, overlap_fraction):
    weight = np.ones(tile_shape, dtype=np.float32)
    for axis in range(ndim):
        size = tile_shape[axis]
        overlap_px = max(1, int(size * overlap_fraction))
        ramp = np.linspace(0, 1, overlap_px, dtype=np.float32)
        w1d = np.ones(size, dtype=np.float32)
        w1d[:overlap_px] = ramp
        w1d[-overlap_px:] = ramp[::-1]
        shape = [1] * ndim
        shape[axis] = size
        weight *= w1d.reshape(shape)
    return weight


# --------------------------------------------------------- rasterization


def _rasterize_to_bbox(center, rays, dists, vol_shape):
    """Return ``(bbox_slices, mask)`` for the star polyhedron at ``center``.

    ``mask`` is a boolean array sized to fit the polyhedron's bounding
    box (clipped to the volume); ``bbox_slices`` is the per-axis slice
    that places ``mask`` back into the full volume.
    """
    ndim = rays.shape[1]
    # Sanitise the distance vector — degenerate model outputs sometimes
    # have NaN / ±inf in some rays and that propagates through the
    # bbox math and emits "invalid value encountered in cast" warnings.
    dists = np.nan_to_num(dists, nan=0.0, posinf=0.0, neginf=0.0)
    # Bounding box: the polyhedron lies within max_dist voxels of center
    # on each axis. Use the per-axis ray×dist extent for a tight box.
    ray_extents = rays * dists[:, None]  # (n_rays, ndim)
    pos = np.maximum(0, ray_extents.max(axis=0))  # outward extent per axis
    neg = np.maximum(0, -ray_extents.min(axis=0))
    lo = np.floor(center - neg).astype(np.int64)
    hi = np.ceil(center + pos).astype(np.int64) + 1
    lo = np.clip(lo, 0, np.array(vol_shape))
    hi = np.clip(hi, 0, np.array(vol_shape))
    if np.any(hi <= lo):
        return tuple(slice(0, 0) for _ in range(ndim)), np.zeros(
            (0,) * ndim, dtype=bool
        )

    # Voxel coord grid inside the bbox, relative to center.
    axes = [np.arange(lo[d], hi[d]) - center[d] for d in range(ndim)]
    grids = np.meshgrid(*axes, indexing="ij")  # ndim arrays of bbox shape
    coords = np.stack([g.ravel() for g in grids], axis=1).astype(
        np.float32
    )  # (M, ndim)

    norm = np.linalg.norm(coords, axis=1)  # (M,)
    safe = norm > 1e-6
    # Unit vectors from center to each voxel.
    unit = np.zeros_like(coords)
    unit[safe] = coords[safe] / norm[safe, None]

    # Ray directions, normalized.
    ray_norm = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    # Pick the nearest ray (largest dot product) for each voxel.
    dots = unit @ ray_norm.T  # (M, n_rays)
    nearest = np.argmax(dots, axis=1)  # (M,)
    inside = norm <= dists[nearest]
    inside |= ~safe  # the center itself is in
    mask = inside.reshape(grids[0].shape)
    bbox_slices = tuple(slice(int(lo[d]), int(hi[d])) for d in range(ndim))
    return bbox_slices, mask


def _bbox_iou(bbox_a, mask_a, bbox_b, mask_b) -> float:
    """IoU between two rasterized polyhedra given their bounding-box slices."""
    overlap_slices_a = []
    overlap_slices_b = []
    for sa, sb in zip(bbox_a, bbox_b):
        lo = max(sa.start, sb.start)
        hi = min(sa.stop, sb.stop)
        if hi <= lo:
            return 0.0
        overlap_slices_a.append(slice(lo - sa.start, hi - sa.start))
        overlap_slices_b.append(slice(lo - sb.start, hi - sb.start))
    inter = np.logical_and(
        mask_a[tuple(overlap_slices_a)], mask_b[tuple(overlap_slices_b)]
    ).sum()
    if inter == 0:
        return 0.0
    union = mask_a.sum() + mask_b.sum() - inter
    return float(inter) / float(union)


def _nms_polyhedra(*, centers, dists, scores, rays, vol_shape, iou_thresh):
    """Greedy NMS — sort by descending score, drop those overlapping kept ones."""
    order = np.argsort(-scores)
    kept_idx, kept_bbox, kept_masks = [], [], []
    for i in order:
        bbox_i, mask_i = _rasterize_to_bbox(centers[i], rays, dists[i], vol_shape)
        if not mask_i.any():
            continue
        suppress = False
        for bbox_j, mask_j in zip(kept_bbox, kept_masks):
            if _bbox_iou(bbox_i, mask_i, bbox_j, mask_j) >= iou_thresh:
                suppress = True
                break
        if not suppress:
            kept_idx.append(int(i))
            kept_bbox.append(bbox_i)
            kept_masks.append(mask_i)
    return kept_idx, kept_bbox, kept_masks


def _paint_labels(vol_shape, bboxes, masks) -> np.ndarray:
    out = np.zeros(vol_shape, dtype=np.uint16)
    for k, (bbox, mask) in enumerate(zip(bboxes, masks), start=1):
        # Painting in order keeps later (lower-score) survivors from
        # overwriting earlier ones — but since NMS already enforces no
        # significant overlap, "where empty" is enough.
        region = out[bbox]
        region[(region == 0) & mask] = k
        out[bbox] = region
    return out
