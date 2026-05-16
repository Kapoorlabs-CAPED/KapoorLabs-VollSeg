"""Nearest-neighbour cell tracking across timelapse label volumes.

Given a 2D+T ``(T, Y, X)`` or 3D+T ``(T, Z, Y, X)`` label volume, produce
a *relabeled* volume of the same shape in which each cell carries a
**track ID** stable across frames.

The linker (Hungarian by default, greedy as a faster fallback) operates
on a cost matrix you can shape with a user-chosen set of morphodynamic
features. The cost between previous track *i* and current label *j* is

    cost[i, j] = Σ_f  α_f · d_f(i, j)

where ``f`` ranges over the features in ``features`` and ``α_f`` is the
weight in ``weights[f]`` (or the registry default). Per-feature
distances ``d_f`` are documented next to each entry of
:data:`FEATURE_DEFAULT_WEIGHTS`. ``max_link_distance`` is a hard gate
applied to the **centroid** distance regardless of which features are
in the mix, so the user keeps a familiar "no jumps farther than X μm"
safety net.

Births get fresh track IDs; deaths drop out of the candidate pool
(no gap-closing — use Trackastra / btrack if you need that).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops_table


_BIG = 1e18


# ============================================================ feature registry

# Default per-feature weights (``α_f`` in the cost equation). The scale
# of each penalty is documented in the inline comment so the defaults
# make a roughly comparable contribution to the centroid term, given
# typical animal-cell microscopy. Override any subset via the
# ``weights`` kwarg of :func:`link_labels_timelapse`.
FEATURE_DEFAULT_WEIGHTS: dict[str, float] = {
    # Centroid Euclidean distance, in physical units (μm if you fed
    # spacing in μm). Pure-position cost.
    "centroid": 1.0,
    # |R_curr − R_prev| where R is the equivalent radius of the
    # region (sqrt(area/π) in 2D, ((3V)/(4π))^(1/3) in 3D). Length
    # units → naturally combinable with centroid.
    "radius": 1.0,
    # |log(V_curr) − log(V_prev)| — scale-invariant: a 2× volume jump
    # contributes log 2 ≈ 0.69. Multiply by ~5 to make a doubling
    # ≈ 3.5 μm of centroid drift.
    "volume": 5.0,
    # |ecc_curr − ecc_prev| (2D only; range 0..1). Scale 10 so Δecc
    # of 0.1 ≈ 1 μm of centroid drift.
    "eccentricity": 10.0,
    # |sol_curr − sol_prev| where sol = V / V_convex (range 0..1).
    "solidity": 10.0,
    # |ext_curr − ext_prev| where ext = V / V_bbox (range 0..1).
    "extent": 10.0,
    # min(|Δθ|, π − |Δθ|) radians, 2D only.
    "orientation": 5.0,
    # |μ_curr − μ_prev| / max(μ_prev, ε) — relative intensity change.
    # Requires ``intensity_image``. Scale 5 so a 20 % change ≈ 1 unit.
    "mean_intensity": 5.0,
    "max_intensity": 5.0,
    "min_intensity": 5.0,
}

# Which regionprops_table property each feature pulls.
_FEATURE_TO_PROP: dict[str, str] = {
    "radius": "area",
    "volume": "area",
    "eccentricity": "eccentricity",
    "solidity": "solidity",
    "extent": "extent",
    "orientation": "orientation",
    "mean_intensity": "mean_intensity",
    "max_intensity": "max_intensity",
    "min_intensity": "min_intensity",
}

# Features that only make sense in 2D — eccentricity / orientation rely
# on the 2D inertia ellipse; skimage refuses them in 3D.
_FEATURES_2D_ONLY: set[str] = {"eccentricity", "orientation"}

# Features that need ``intensity_image`` to be supplied.
_FEATURES_NEED_INTENSITY: set[str] = {
    "mean_intensity",
    "max_intensity",
    "min_intensity",
}


def available_features(spatial_ndim: int) -> list[str]:
    """Return the names of features supported for ``spatial_ndim`` (2 or 3)."""
    if spatial_ndim == 2:
        return list(FEATURE_DEFAULT_WEIGHTS.keys())
    return [f for f in FEATURE_DEFAULT_WEIGHTS if f not in _FEATURES_2D_ONLY]


# ============================================================ per-frame extract


def _frame_centroids(
    frame: np.ndarray,
    spacing_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(label_ids, centroids)`` for a single frame (physical units)."""
    ids = np.unique(frame)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return ids, np.zeros((0, frame.ndim), dtype=np.float64)
    coms = ndimage.center_of_mass(frame > 0, frame, ids.tolist())
    centroids = np.asarray(coms, dtype=np.float64) * spacing_arr
    return ids, centroids


def _compute_frame_features(
    frame: np.ndarray,
    ids: np.ndarray,
    *,
    spacing_arr: np.ndarray,
    features: tuple[str, ...],
    intensity_frame: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Compute the per-label feature arrays for one frame.

    Returns ``{feature_name: array_indexed_by_ids_position}``. The
    ``centroid`` feature is *not* included here — it's already returned
    by :func:`_frame_centroids`.
    """
    extra = [f for f in features if f != "centroid"]
    if not extra or len(ids) == 0:
        return {}

    needs_intensity = any(f in _FEATURES_NEED_INTENSITY for f in extra)
    if needs_intensity and intensity_frame is None:
        raise ValueError(
            "intensity_image is required for features: "
            f"{sorted(f for f in extra if f in _FEATURES_NEED_INTENSITY)}"
        )

    skimage_props = sorted({_FEATURE_TO_PROP[f] for f in extra})
    props = regionprops_table(
        frame,
        intensity_image=intensity_frame,
        properties=("label",) + tuple(skimage_props),
    )
    label_pos = {int(lbl): i for i, lbl in enumerate(props["label"])}

    voxel_vol = float(np.prod(spacing_arr))
    out: dict[str, np.ndarray] = {}

    for f in extra:
        key = _FEATURE_TO_PROP[f]
        raw = np.asarray(
            [props[key][label_pos[int(lid)]] for lid in ids],
            dtype=np.float64,
        )
        if f == "volume":
            out[f] = raw * voxel_vol
        elif f == "radius":
            phys_vol = raw * voxel_vol
            if frame.ndim == 2:
                out[f] = np.sqrt(np.maximum(phys_vol, 0.0) / np.pi)
            else:
                out[f] = np.cbrt(np.maximum(phys_vol, 0.0) * 3.0 / (4.0 * np.pi))
        else:
            out[f] = raw

    return out


# ====================================================== per-feature distances


def _feature_distance_matrix(
    name: str,
    prev_vals: np.ndarray,
    curr_vals: np.ndarray,
) -> np.ndarray:
    """Return the ``(n_prev, n_curr)`` pairwise distance for one feature."""
    if name == "volume":
        # Log-distance so a 2× change costs ln 2 regardless of cell size.
        p = np.log(np.maximum(prev_vals, 1e-12))
        c = np.log(np.maximum(curr_vals, 1e-12))
        return np.abs(p[:, None] - c[None, :])
    if name == "orientation":
        d = np.abs(prev_vals[:, None] - curr_vals[None, :])
        return np.minimum(d, np.pi - d)
    if name in _FEATURES_NEED_INTENSITY:
        # Relative change vs. previous, so brightness-invariant cells
        # don't dominate dim cells.
        rel = np.abs(prev_vals[:, None] - curr_vals[None, :]) / np.maximum(
            np.abs(prev_vals[:, None]),
            1e-12,
        )
        return rel
    # Default: absolute difference.
    return np.abs(prev_vals[:, None] - curr_vals[None, :])


def _greedy_match(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively pick the smallest cost-matrix entry, mask its row/col."""
    cost = cost.copy()
    n_rows, n_cols = cost.shape
    rows, cols = [], []
    for _ in range(min(n_rows, n_cols)):
        flat_idx = int(np.argmin(cost))
        r, c = divmod(flat_idx, n_cols)
        if cost[r, c] >= _BIG:
            break
        rows.append(r)
        cols.append(c)
        cost[r, :] = _BIG
        cost[:, c] = _BIG
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


# =============================================================== main entry


def link_labels_timelapse(
    labels: np.ndarray,
    *,
    spatial_ndim: int,
    spacing: Optional[tuple[float, ...]] = None,
    max_link_distance: Optional[float] = None,
    method: str = "hungarian",
    features: tuple[str, ...] = ("centroid",),
    weights: Optional[dict[str, float]] = None,
    intensity_image: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    """Link per-frame labels with a user-shapeable cost matrix.

    Parameters
    ----------
    labels
        ``(T, Y, X)`` for 2D+T or ``(T, Z, Y, X)`` for 3D+T. Background
        must be 0.
    spatial_ndim
        2 or 3.
    spacing
        Spatial voxel size; ``None`` → unit spacing.
    max_link_distance
        Hard gate on *centroid distance* (physical units). Pairs
        farther apart cannot link regardless of how well other
        features agree.
    method
        ``"hungarian"`` (default) or ``"greedy"``.
    features
        Tuple of feature names to include in the cost matrix. Any of:

        - ``"centroid"`` — Euclidean centroid distance (default).
        - ``"radius"`` — equivalent-radius change.
        - ``"volume"`` — log-volume change (scale invariant).
        - ``"eccentricity"`` — 2D only.
        - ``"solidity"`` — V / V_convex change.
        - ``"extent"`` — V / V_bbox change.
        - ``"orientation"`` — 2D only (radians, wraparound aware).
        - ``"mean_intensity"`` / ``"max_intensity"`` /
          ``"min_intensity"`` — relative change; needs
          ``intensity_image``.

        Use :func:`available_features` to enumerate valid names per
        ``spatial_ndim``.
    weights
        ``{feature_name: α}`` — per-feature scale factor. Missing
        entries fall back to :data:`FEATURE_DEFAULT_WEIGHTS`. Setting
        a weight to ``0.0`` effectively disables that feature.
    intensity_image
        Same shape as ``labels`` (``(T, …)``); required only when
        any intensity feature is in ``features``.

    Returns
    -------
    relabeled, tracks
        See module docstring.
    """
    if labels.ndim != spatial_ndim + 1:
        raise ValueError(
            f"labels.ndim ({labels.ndim}) must equal spatial_ndim + 1 "
            f"(={spatial_ndim + 1})."
        )
    if method not in {"hungarian", "greedy"}:
        raise ValueError(f"method must be 'hungarian' or 'greedy', got {method!r}")
    if intensity_image is not None and intensity_image.shape != labels.shape:
        raise ValueError(
            f"intensity_image shape {intensity_image.shape} must match "
            f"labels shape {labels.shape}"
        )

    # Validate feature names against the registry and dim restrictions.
    features = tuple(features)
    unknown = [f for f in features if f not in FEATURE_DEFAULT_WEIGHTS]
    if unknown:
        raise ValueError(
            f"Unknown features: {unknown}. "
            f"Available: {available_features(spatial_ndim)}"
        )
    bad_dim = [f for f in features if spatial_ndim == 3 and f in _FEATURES_2D_ONLY]
    if bad_dim:
        raise ValueError(f"Features {bad_dim} are 2D-only; got spatial_ndim=3")

    eff_weights: dict[str, float] = {
        f: float((weights or {}).get(f, FEATURE_DEFAULT_WEIGHTS[f])) for f in features
    }

    if spacing is None:
        spacing = (1.0,) * spatial_ndim
    spacing_arr = np.asarray(spacing[:spatial_ndim], dtype=np.float64)

    n_frames = labels.shape[0]
    relabeled = np.zeros_like(labels, dtype=np.int32)
    tracks: dict[int, list[tuple[int, int]]] = {}
    next_track_id = 1

    # prev_active[tid] = {"centroid": (ndim,), "<feature>": float, ...}
    prev_active: dict[int, dict[str, np.ndarray]] = {}

    for t in range(n_frames):
        frame = labels[t]
        ids, centroids = _frame_centroids(frame, spacing_arr)
        if len(ids) == 0:
            prev_active = {}
            continue

        intensity_frame = intensity_image[t] if intensity_image is not None else None
        extra_feats = _compute_frame_features(
            frame,
            ids,
            spacing_arr=spacing_arr,
            features=features,
            intensity_frame=intensity_frame,
        )

        # Bundle this frame's features into a per-label dict so we can
        # promote matched ones into ``new_active`` cleanly.
        per_label: list[dict[str, np.ndarray]] = []
        for j in range(len(ids)):
            bundle = {"centroid": centroids[j]}
            for f, arr in extra_feats.items():
                bundle[f] = arr[j]
            per_label.append(bundle)

        label_to_track: dict[int, int] = {}
        new_active: dict[int, dict[str, np.ndarray]] = {}

        if not prev_active:
            for lid, bundle in zip(ids, per_label):
                tid = next_track_id
                next_track_id += 1
                label_to_track[int(lid)] = tid
                new_active[tid] = bundle
                tracks[tid] = [(t, int(lid))]
        else:
            prev_tids = list(prev_active.keys())
            n_prev, n_curr = len(prev_tids), len(ids)

            # Centroid distance always computed (needed for gating).
            prev_centroids = np.stack([prev_active[k]["centroid"] for k in prev_tids])
            centroid_dist = np.linalg.norm(
                prev_centroids[:, None, :] - centroids[None, :, :],
                axis=-1,
            )

            cost = np.zeros((n_prev, n_curr), dtype=np.float64)
            for f, alpha in eff_weights.items():
                if alpha == 0.0:
                    continue
                if f == "centroid":
                    cost += alpha * centroid_dist
                    continue
                prev_vals = np.asarray(
                    [prev_active[k].get(f, np.nan) for k in prev_tids],
                    dtype=np.float64,
                )
                curr_vals = extra_feats[f]
                # A previous track may lack this feature only if it
                # was born in a frame before the feature was requested
                # — not possible here since `features` is global — so
                # NaN should not arise. Belt-and-braces NaN→0 cost.
                d = _feature_distance_matrix(f, prev_vals, curr_vals)
                d = np.where(np.isnan(d), 0.0, d)
                cost += alpha * d

            # Hard gate on physical centroid distance.
            if max_link_distance is not None:
                cost = np.where(centroid_dist > max_link_distance, _BIG, cost)

            if method == "hungarian":
                row_ind, col_ind = linear_sum_assignment(cost)
            else:
                row_ind, col_ind = _greedy_match(cost)

            matched_curr: set[int] = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= _BIG:
                    continue
                tid = prev_tids[int(r)]
                lid = int(ids[int(c)])
                label_to_track[lid] = tid
                tracks[tid].append((t, lid))
                new_active[tid] = per_label[int(c)]
                matched_curr.add(int(c))

            for j, lid in enumerate(ids):
                if j in matched_curr:
                    continue
                tid = next_track_id
                next_track_id += 1
                label_to_track[int(lid)] = tid
                new_active[tid] = per_label[j]
                tracks[tid] = [(t, int(lid))]

        max_id = int(frame.max())
        lut = np.zeros(max_id + 1, dtype=np.int32)
        for lid, tid in label_to_track.items():
            lut[lid] = tid
        relabeled[t] = lut[frame]

        prev_active = new_active

    return relabeled, tracks
