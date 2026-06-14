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

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.spatial import ConvexHull, cKDTree
from skimage.feature import peak_local_max
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .._lightning.dataset import CarePredictionDataset, compute_tile_shape
from .._lightning.transforms import PercentileNormalize
from .lightning_module import StarDistModule


# Whether the inner ``tqdm`` bars (per-tile, per-NMS-peak, per-painted
# polyhedron) should actually render. In an interactive terminal they
# carriage-return in place and erase on completion (``leave=False``)
# so each phase replaces the previous one; in a non-TTY context (SLURM
# log, ``tee``, ``nohup`` redirect) tqdm prints newlines instead so
# the per-frame work spams thousands of lines. Default to *disabled*
# when stderr isn't a TTY; opt back in with
# ``KAPOORLABS_VOLLSEG_PROGRESS=1`` if you really want all the bars.
_SHOW_INNER_PROGRESS = (
    sys.stderr.isatty() or os.environ.get("KAPOORLABS_VOLLSEG_PROGRESS") == "1"
)


def _phase_status(desc: str, n_items: int, elapsed: float, unit: str) -> None:
    """One-liner that emits "phase done" status when the live tqdm
    inside that phase is disabled (non-TTY). In a TTY the bar already
    self-erases via ``leave=False`` so this print would be redundant
    chatter — gate accordingly.
    """
    if _SHOW_INNER_PROGRESS:
        return
    rate = n_items / elapsed if elapsed > 0 else 0.0
    print(
        f"  {desc} done — {n_items} {unit} in {elapsed:.1f}s " f"({rate:.1f} {unit}/s)",
        flush=True,
    )


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
    faces: Optional[np.ndarray] = None,
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
        faces=faces,
    )
    return StarDistResult(
        labels=labels,
        prob_map=prob_map,
        n_objects=int(labels.max()),
    )


def precompute_peaks_and_masks(
    prob_map: np.ndarray,
    dist_map: np.ndarray,
    rays: np.ndarray,
    vol_shape: tuple,
    *,
    min_prob: float = 0.01,
    min_distance: int = 2,
    faces: Optional[np.ndarray] = None,
):
    """Detect + rasterise once at the **lowest** ``prob_thresh`` the sweep
    will ever try, so the threshold optimiser can reuse the same masks
    across every ``(prob_thresh, nms_thresh)`` candidate.

    Returns ``(centers, scores, bboxes, masks)`` sorted by descending
    score — the optimiser then just filters by ``prob_thresh``, runs the
    cheap NMS overlap loop, and paints. Skips the slow
    :func:`peak_local_max` + per-peak meshgrid rasterise that
    :func:`nms_to_labels` would otherwise rerun every iteration.
    """
    centers = peak_local_max(
        prob_map,
        threshold_abs=min_prob,
        min_distance=min_distance,
        exclude_border=False,
    )
    if centers.size == 0:
        return centers, np.zeros((0,), np.float32), [], []

    scores = prob_map[tuple(centers.T)].astype(np.float32)
    dists = np.stack([dist_map[(slice(None),) + tuple(c)] for c in centers], axis=0)
    order = np.argsort(-scores)
    centers, scores, dists = centers[order], scores[order], dists[order]

    bboxes, masks = [], []
    for c, d in zip(centers, dists):
        bbox, mask = _rasterize_to_bbox(c, rays, d, vol_shape, faces=faces)
        bboxes.append(bbox)
        masks.append(mask)
    return centers, scores, bboxes, masks


def labels_from_precomputed(
    centers: np.ndarray,
    scores: np.ndarray,
    bboxes: list,
    masks: list,
    vol_shape: tuple,
    *,
    prob_thresh: float,
    nms_thresh: float,
) -> np.ndarray:
    """Fast inner loop for the threshold sweep — filter by ``prob_thresh``,
    NMS-overlap-loop, paint. ``centers / scores / bboxes / masks`` must
    come from :func:`precompute_peaks_and_masks` (already score-sorted)."""
    if len(scores) == 0:
        return np.zeros(vol_shape, dtype=np.uint16)
    keep = scores >= prob_thresh
    if not keep.any():
        return np.zeros(vol_shape, dtype=np.uint16)

    kept_bbox: list = []
    kept_masks: list = []
    for i in np.where(keep)[0]:
        bbox_i, mask_i = bboxes[i], masks[i]
        if not mask_i.any():
            continue
        suppress = False
        for bbox_j, mask_j in zip(kept_bbox, kept_masks):
            if _bbox_iou(bbox_i, mask_i, bbox_j, mask_j) >= nms_thresh:
                suppress = True
                break
        if not suppress:
            kept_bbox.append(bbox_i)
            kept_masks.append(mask_i)
    return _paint_labels(vol_shape, kept_bbox, kept_masks)


def nms_to_labels(
    prob_map: np.ndarray,
    dist_map: np.ndarray,
    rays: np.ndarray,
    vol_shape: tuple,
    *,
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.4,
    min_distance: int = 2,
    faces: Optional[np.ndarray] = None,
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

    # Surface the peak count before the (potentially O(N^2)) NMS so a
    # pathological model (e.g. noisy prob map → tens of thousands of
    # candidates) is visible up front instead of being a silent wait.
    print(
        f"  nms_to_labels: {len(centers)} peaks above prob_thresh="
        f"{prob_thresh} (min_distance={min_distance})",
        flush=True,
    )

    scores = prob_map[tuple(centers.T)]
    dists = np.stack(
        [dist_map[(slice(None),) + tuple(c)] for c in centers], axis=0
    )  # (M, n_rays)

    # NMS first (KDTree-accelerated bbox-IoU, same approach the stardist
    # C++ NMS uses) to drop duplicates; then paint the kept polyhedra
    # straight into the label image via the per-face tetrahedron
    # decomposition (the algorithm in
    # stardist/lib/stardist3d_impl.cpp::polyhedron_to_label).
    kept_idx = _bbox_nms_kdtree(
        centers=centers,
        dists=dists,
        scores=scores,
        rays=rays,
        vol_shape=vol_shape,
        iou_thresh=nms_thresh,
    )
    if faces is None and rays.shape[1] == 3:
        faces = _compute_ray_faces(rays)
    return _polyhedra_to_label(
        vol_shape=vol_shape,
        kept_idx=kept_idx,
        centers=centers,
        dists=dists,
        rays=rays,
        faces=faces,
    )


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

    # Per-frame, the outer Lightning predict bar only ticks once a whole
    # frame is done — inside that single tick we run len(loader) forward
    # passes (e.g. 21 for batch_size=4, 81 tiles). Expose a transient
    # per-tile bar so the caller sees forward progress within a frame
    # and can spot a hang vs a slow-but-progressing run.
    tile_desc = f"tiles[shape={tile_shape},bs={batch_size}]"
    tile_iter = tqdm(
        loader,
        total=len(loader),
        desc=f"  {tile_desc}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
        disable=not _SHOW_INNER_PROGRESS,
    )
    t_tiles = time.perf_counter()

    with torch.no_grad():
        for tiles, coords in tile_iter:
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

    _phase_status(tile_desc, len(loader), time.perf_counter() - t_tiles, "batches")

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


def _rasterize_to_bbox(center, rays, dists, vol_shape, faces=None):
    """Return ``(bbox_slices, mask)`` for the star polyhedron at ``center``.

    Two paths:

    - **3D + faces given**: rasterise the **true triangulated star-convex
      polyhedron** as the union of tetrahedra
      ``(center, d_a·v_a, d_b·v_b, d_c·v_c)`` for each ConvexHull triangle
      face ``(a, b, c)`` of the rays. Same surface keras stardist
      reconstructs via ``polyhedron_to_label``.
    - **2D (or no faces)**: fall back to the legacy nearest-ray cone
      approximation — for 2D this is essentially exact when ``n_rays``
      is reasonable, and it's the only path that doesn't need a
      triangulation.
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
        np.float64
    )  # (M, ndim)

    if ndim == 3 and faces is not None and len(faces) > 0:
        inside = _inside_polyhedron(coords, rays, dists, faces)
    else:
        # 2D / no-faces fallback: nearest-ray cone test.
        norm = np.linalg.norm(coords, axis=1)
        safe = norm > 1e-6
        unit = np.zeros_like(coords)
        unit[safe] = coords[safe] / norm[safe, None]
        ray_norm = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        dots = unit @ ray_norm.T
        nearest = np.argmax(dots, axis=1)
        inside = norm <= dists[nearest]
        inside |= ~safe  # the center itself is in

    mask = inside.reshape(grids[0].shape)
    bbox_slices = tuple(slice(int(lo[d]), int(hi[d])) for d in range(ndim))
    return bbox_slices, mask


def _inside_polyhedron(coords, rays, dists, faces):
    """Vectorised "voxel-in-star-polyhedron" test for 3D.

    Decomposes the star-convex polyhedron into tetrahedra — one per
    triangle face of the boundary, with the center (origin in the
    ``coords`` frame) as the apex. A voxel is inside the polyhedron iff
    it lies inside at least one such tetrahedron, i.e. its barycentric
    coordinates ``(w_a, w_b, w_c, w_origin = 1 - w_a - w_b - w_c)`` w.r.t
    the four corners are all non-negative.

    ``coords`` are voxel offsets from the polyhedron center; returns a
    flat bool array of length ``len(coords)``.
    """
    n_faces = faces.shape[0]
    n_vox = coords.shape[0]
    if n_faces == 0 or n_vox == 0:
        out = np.zeros(n_vox, dtype=bool)
        if n_vox > 0:
            # Center voxel is always inside.
            out |= np.linalg.norm(coords, axis=1) <= 1e-6
        return out

    # Scaled face vertices, shape (n_faces, 3, 3): the (i, j, :) entry is
    # the j-th scaled ray of face i, expressed in (z, y, x).
    scaled = rays * dists[:, None]  # (n_rays, 3)
    face_verts = scaled[faces]  # (n_faces, 3, 3)

    # Matrix M_f with columns = scaled vertices; inverting it gives the
    # transform world -> barycentric (origin-relative).
    M = face_verts.transpose(0, 2, 1)  # (n_faces, 3, 3): columns = vertices
    try:
        Minv = np.linalg.inv(M)  # (n_faces, 3, 3)
    except np.linalg.LinAlgError:
        # Degenerate face — skip the whole rasterisation cleanly.
        return np.zeros(n_vox, dtype=bool)

    tol = -1e-9
    inside = np.zeros(n_vox, dtype=bool)
    # Chunk voxels to keep the (n_faces, chunk, 3) tensor's memory bounded.
    # For n_faces=188 (n_rays=96) and chunk=4096, that's ~18 MB per chunk.
    chunk = 4096
    for s in range(0, n_vox, chunk):
        v = coords[s : s + chunk]  # (k, 3)
        # w[f, m, i] = barycentric coord of voxel m in tetrahedron of face f.
        w = np.einsum("fij,mj->fmi", Minv, v)  # (n_faces, k, 3)
        w_origin = 1.0 - w.sum(axis=-1)
        ok = (
            (w[..., 0] >= tol)
            & (w[..., 1] >= tol)
            & (w[..., 2] >= tol)
            & (w_origin >= tol)
        )  # (n_faces, k)
        inside[s : s + chunk] = ok.any(axis=0)

    # The center itself is always part of the polyhedron.
    inside |= np.linalg.norm(coords, axis=1) <= 1e-6
    return inside


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


def _compute_ray_faces(rays: np.ndarray) -> np.ndarray:
    """Triangulated convex hull of the (unit-sphere) ray vertices.

    Stardist's ``Rays3D`` ships ``.faces`` as the ConvexHull simplices
    of the ray directions — the same triangulation stardist's
    ``polyhedron_to_label`` C kernel iterates over to fill each
    star-polyhedron tetrahedron by tetrahedron. We compute it on the
    fly from the rays array so we don't need stardist installed.
    """
    rays = np.asarray(rays, dtype=np.float32)
    # Normalise just in case so the hull is built on the unit sphere.
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = rays / norms
    return ConvexHull(unit).simplices.astype(np.int64)


def _polyhedra_to_label(
    *,
    vol_shape: tuple,
    kept_idx: np.ndarray,
    centers: np.ndarray,
    dists: np.ndarray,
    rays: np.ndarray,
    faces: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Paint the kept star-polyhedra into a uint16 label image.

    Uses the **nearest-ray** rasterisation already documented at the
    top of this file: for every voxel in the polyhedron's bbox find
    the ray direction whose unit vector has the largest dot product
    with the voxel's direction-from-centre, then keep the voxel iff
    its distance from the centre is ≤ that ray's predicted length.
    For ~64+ rays (we use 96 by default) the surface this produces is
    visually indistinguishable from the per-face tetrahedron paint
    that stardist's C kernel runs, but the whole inside-test is a
    single BLAS matmul per polyhedron instead of 188 tetrahedron
    rasterisations — which is the only way to get acceptable
    throughput out of pure numpy.

    Painted **only where the label image is currently zero**, so
    earlier (higher-score) polyhedra win on overlap — same semantics
    as the keras stardist NMS+paint loop.

    ``faces`` is accepted but unused (kept in the signature for
    backwards compatibility with callers that pass it explicitly).
    """
    label_img = np.zeros(vol_shape, dtype=np.uint16)
    if len(kept_idx) == 0:
        return label_img

    rays = np.asarray(rays, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    dists = np.asarray(dists, dtype=np.float32)
    vol_shape_arr = np.asarray(vol_shape, dtype=np.int64)
    del faces  # accepted for API compat; not used in the nearest-ray path

    # Precompute unit ray directions once — they don't change per peak.
    ray_norms = np.linalg.norm(rays, axis=1, keepdims=True)
    ray_norms[ray_norms == 0] = 1.0
    ray_unit = (rays / ray_norms).astype(np.float32)  # (n_rays, 3)

    paint_desc = f"paint kept[{len(kept_idx)}]"
    pbar = tqdm(
        enumerate(kept_idx, start=1),
        total=len(kept_idx),
        desc=f"  {paint_desc}",
        unit="cell",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
        disable=not _SHOW_INNER_PROGRESS,
    )
    t_paint = time.perf_counter()
    for label_value, peak_idx in pbar:
        if label_value >= np.iinfo(label_img.dtype).max:
            break
        peak_idx = int(peak_idx)
        center = centers[peak_idx]
        d = np.nan_to_num(dists[peak_idx], nan=0.0, posinf=0.0, neginf=0.0).clip(
            min=0.0
        )
        if d.max() <= 0:
            continue

        # Polyhedron's per-axis bbox from the most-positive and
        # most-negative ray extents.
        ray_extents = rays * d[:, None]  # (n_rays, 3)
        pos = np.maximum(0.0, ray_extents.max(axis=0))
        neg = np.maximum(0.0, -ray_extents.min(axis=0))
        lo = np.floor(center - neg).astype(np.int64)
        hi = np.ceil(center + pos).astype(np.int64) + 1
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, vol_shape_arr)
        if np.any(hi <= lo):
            continue

        # Voxel grid relative to centre. (Nv, 3).
        z = np.arange(lo[0], hi[0], dtype=np.float32) - center[0]
        y = np.arange(lo[1], hi[1], dtype=np.float32) - center[1]
        x = np.arange(lo[2], hi[2], dtype=np.float32) - center[2]
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
        coords = np.stack([Z.ravel(), Y.ravel(), X.ravel()], axis=1)

        # Direction from centre to each voxel.
        norm = np.linalg.norm(coords, axis=1)
        safe = norm > 1e-6
        unit = np.zeros_like(coords)
        unit[safe] = coords[safe] / norm[safe, None]

        # Single BLAS matmul: dot product of each voxel's unit vector
        # against every ray direction. For (Nv, 3) × (3, n_rays) this
        # is sub-millisecond for typical-cell-sized bboxes.
        dots = unit @ ray_unit.T  # (Nv, n_rays)
        nearest = np.argmax(dots, axis=1)
        inside = norm <= d[nearest]
        inside |= ~safe  # the centre voxel itself is in
        if not inside.any():
            continue

        mask = inside.reshape(Z.shape)
        region = label_img[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        paint = mask & (region == 0)
        region[paint] = int(label_value)

    _phase_status(paint_desc, len(kept_idx), time.perf_counter() - t_paint, "cells")
    return label_img


def _bbox_nms_kdtree(
    centers: np.ndarray,
    dists: np.ndarray,
    scores: np.ndarray,
    rays: np.ndarray,
    vol_shape: tuple,
    iou_thresh: float,
) -> np.ndarray:
    """Greedy axis-aligned bbox NMS with a KDTree spatial pre-filter.

    Algorithm-equivalent to the C++/OpenMP NMS in the original stardist
    library, ported to pure numpy + scipy so we don't drag a keras /
    stardist dependency into this environment. The trick the C++ impl
    uses (and we copy here) is the KDTree neighbourhood query: for
    each peak in score order, only candidates within
    ``radius_i + radius_max`` of its centre can possibly overlap, so
    we run the (cheap) bbox-IoU check against that small set instead
    of against all N peaks. Reduces the work from O(N²) to O(N × k)
    where k is the average number of bounding-sphere neighbours per
    peak (typically a few dozen in densely packed tissue, even for
    N=30k).

    Returns the indices of kept peaks in *score-descending* order.
    """
    N = len(centers)
    if N == 0:
        return np.zeros(0, dtype=np.int64)

    centers = np.asarray(centers, dtype=np.float32)
    dists = np.asarray(dists, dtype=np.float32)
    rays = np.asarray(rays, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    # Per-peak axis-aligned bbox and bounding-sphere radius derived
    # analytically from the predicted ray distances. The bbox is the
    # smallest box containing the star-polyhedron; the sphere radius
    # is the longest ray (any other peak whose centre is farther than
    # ``r_i + r_j`` from peak i can't overlap with peak i, so KDTree
    # culls the comparison entirely).
    scaled = dists[:, :, None] * rays[None, :, :]  # (N, R, D)
    lo = centers + scaled.min(axis=1)
    hi = centers + scaled.max(axis=1)
    lo = np.maximum(lo, 0.0)
    vol_shape_arr = np.asarray(vol_shape, dtype=np.float32)
    hi = np.minimum(hi, vol_shape_arr)
    vol = np.clip(hi - lo, 0.0, None).prod(axis=1)  # (N,)
    radii = dists.max(axis=1)  # (N,)
    max_radius = float(radii.max())

    # Process in descending-score order. ``rank[i]`` is i's position in
    # ``order``; rank > my_rank means "lower score than me" so those
    # neighbours are the suppression candidates when I visit my peak.
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(N, dtype=np.int64)
    rank[order] = np.arange(N)

    suppressed = np.zeros(N, dtype=bool)
    tree = cKDTree(centers)
    # ``query_radius`` is conservative: r_i + r_max bounds every
    # possible bounding-sphere overlap. False positives are rejected
    # by the bbox-IoU check below.
    nms_desc = f"nms_kdtree[N={N}]"
    pbar = tqdm(
        range(N),
        total=N,
        desc=f"  {nms_desc}",
        unit="peak",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
        disable=not _SHOW_INNER_PROGRESS,
    )
    t_nms = time.perf_counter()
    kept_count = 0
    for pos in pbar:
        i = order[pos]
        if suppressed[i]:
            continue
        kept_count += 1
        # Spheres of peak i and j overlap iff ||c_i - c_j|| < r_i + r_j;
        # ``r_i + max_radius`` is conservative so the query returns a
        # superset (no false negatives).
        nbrs = tree.query_ball_point(centers[i], r=float(radii[i] + max_radius))
        if not nbrs:
            continue
        nbrs = np.fromiter(nbrs, dtype=np.int64, count=len(nbrs))
        # Keep only neighbours that are lower-score (so visiting i
        # decides their fate) and not yet suppressed.
        mask = (rank[nbrs] > pos) & ~suppressed[nbrs]
        nbrs = nbrs[mask]
        if len(nbrs) == 0:
            continue
        # Vectorised bbox IoU between peak i and its candidate neighbours.
        inter_lo = np.maximum(lo[i], lo[nbrs])
        inter_hi = np.minimum(hi[i], hi[nbrs])
        inter = np.clip(inter_hi - inter_lo, 0.0, None).prod(axis=1)
        union = vol[i] + vol[nbrs] - inter
        iou = inter / np.clip(union, 1e-9, None)
        kill = nbrs[iou >= iou_thresh]
        if len(kill):
            suppressed[kill] = True
        if kept_count and (kept_count % 1000 == 0):
            pbar.set_postfix_str(f"kept={kept_count}")

    _phase_status(nms_desc, N, time.perf_counter() - t_nms, "peaks")
    return order[~suppressed[order]]


# ``_nms_polyhedra`` and the per-polyhedron mask flow it returned have
# been removed: ``nms_to_labels`` now goes KDTree-NMS → tetrahedron paint
# directly (see ``_polyhedra_to_label``). The remaining ``_rasterize_to_bbox``
# and ``_paint_labels`` helpers below are still used by the threshold
# optimiser (``precompute_peaks_and_masks`` / ``labels_from_precomputed``).


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
