"""Timelapse curvature: per-frame profiles + cross-frame tracking.

This module composes :func:`link_labels_timelapse` with
:func:`compute_curvature` so a 2D+T or 3D+T label volume becomes:

1. A **relabeled timelapse** (``(T, Y, X)`` / ``(T, Z, Y, X)``) where
   each cell keeps the same integer ID across all frames — easy to
   load in napari as a tracking layer.
2. **Per-frame curvature profiles**, keyed by *track ID* (not the
   per-frame label), so cell ``track_id=5`` has a well-defined
   curvature trajectory through time.
3. A **rendered curvature TIFF stack** with the same ``(T, …)`` shape
   as the input labels — open it as an extra channel beside the raw
   data and you can scrub through time to watch κ evolve.
4. An optional **per-track CSV** summarising
   median / IQR curvature (and pressure / bending density when those
   physics knobs were supplied) for each frame of each track —
   exactly the "timelapse of curvature profile for all the integer
   labelled cells" picture.

The rendering / file-naming follows the same convention as the static
helpers in ``render.py``: a sibling ``curvature/`` folder next to the
segmentation folder, one ``*_kappa.tif`` etc. per input.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tifffile

from .api import compute_curvature
from .profile import CurvatureProfile
from .render import render_curvature_volume
from .tracking import link_labels_timelapse


# ============================================================== container


@dataclass
class CurvatureTimelapse:
    """All artefacts produced for one timelapse label volume."""

    spatial_ndim: int  # 2 or 3
    n_frames: int
    relabeled: np.ndarray  # (T, …) track IDs
    per_frame: list[dict[int, CurvatureProfile]]  # idx=t, key=track_id
    tracks: dict[int, list[tuple[int, int]]]  # tid -> [(t, orig_label)]
    spacing: tuple[float, ...] = field(default_factory=tuple)

    # ---------------------------------------------------------- queries

    @property
    def track_ids(self) -> list[int]:
        return sorted(self.tracks.keys())

    def track_series(
        self,
        track_id: int,
        *,
        statistic: str = "median",
        field: str = "kappa",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(frames, values)`` for one track across time.

        ``statistic`` is one of ``"median"``, ``"mean"``, ``"max"``,
        ``"min"`` — applied per frame to ``profile.<field>``.
        ``field`` is any per-window attribute: ``"kappa"``,
        ``"pressure"``, ``"bending_density"``, ``"radii"``.

        Frames where the track is missing (e.g. it died and was
        re-born — though gap-closing is off by default so this is
        rare) are simply absent from the returned arrays.
        """
        op = {
            "median": np.median,
            "mean": np.mean,
            "max": np.max,
            "min": np.min,
        }.get(statistic)
        if op is None:
            raise ValueError(f"Unknown statistic: {statistic!r}")

        frames, values = [], []
        for t, frame_profiles in enumerate(self.per_frame):
            profile = frame_profiles.get(track_id)
            if profile is None:
                continue
            vals = getattr(profile, field, None)
            if vals is None or len(vals) == 0:
                continue
            frames.append(t)
            values.append(float(op(vals)))
        return np.asarray(frames, dtype=np.int64), np.asarray(values, dtype=np.float64)

    def summary_table(self) -> list[dict]:
        """Flat list of dicts: one row per (frame, track) for CSV / pandas."""
        rows = []
        for t, frame_profiles in enumerate(self.per_frame):
            for tid, profile in frame_profiles.items():
                row = {"frame": int(t), "track_id": int(tid), **profile.summary()}
                rows.append(row)
        return rows


# ============================================================ orchestrator


def compute_curvature_timelapse(
    labels: np.ndarray,
    *,
    spatial_ndim: int,
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
    n_window: int = 21,
    stride: int = 5,
    geodesic: bool = True,
    geodesic_method: str = "bfs",
    surface_tension: Optional[float] = None,
    bending_modulus: Optional[float] = None,
    spontaneous_curvature: float = 0.0,
    saddle_splay_modulus: Optional[float] = None,
    max_link_distance: Optional[float] = None,
    link_method: str = "hungarian",
    link_features: tuple[str, ...] = ("centroid",),
    link_weights: Optional[dict[str, float]] = None,
    intensity_image: Optional[np.ndarray] = None,
) -> CurvatureTimelapse:
    """Track cells across frames, then compute per-frame curvature.

    Parameters
    ----------
    labels
        Per-frame integer label volume. ``(T, Y, X)`` for 2D+T,
        ``(T, Z, Y, X)`` for 3D+T.
    spatial_ndim
        2 or 3 — the *spatial* dimensionality of each frame.
    spacing
        Spatial voxel size — ``(dy, dx)`` for 2D+T or
        ``(dz, dy, dx)`` for 3D+T. **No time spacing.** If you pass
        a longer tuple it is truncated to ``spatial_ndim``.
    n_window, stride, geodesic, geodesic_method
        Forwarded to :func:`compute_curvature` per frame.
    surface_tension, bending_modulus, spontaneous_curvature, saddle_splay_modulus
        Forwarded to :func:`compute_curvature` per frame.
    max_link_distance, link_method
        Forwarded to :func:`link_labels_timelapse` for the tracker.
    link_features
        Features that shape the linker's cost matrix. See
        :func:`link_labels_timelapse` and
        :func:`kapoorlabs_vollseg.curvature.tracking.available_features`.
    link_weights
        ``{feature: α}`` overrides for the per-feature weights.
    intensity_image
        Same shape as ``labels``; required only when any intensity
        feature is in ``link_features``.
    """
    if spatial_ndim not in (2, 3):
        raise ValueError(f"spatial_ndim must be 2 or 3, got {spatial_ndim}")
    spatial_spacing = tuple(spacing[:spatial_ndim])

    relabeled, tracks = link_labels_timelapse(
        labels,
        spatial_ndim=spatial_ndim,
        spacing=spatial_spacing,
        max_link_distance=max_link_distance,
        method=link_method,
        features=link_features,
        weights=link_weights,
        intensity_image=intensity_image,
    )

    per_frame: list[dict[int, CurvatureProfile]] = []
    for t in range(relabeled.shape[0]):
        profiles = compute_curvature(
            relabeled[t],
            spacing=spatial_spacing,
            n_window=n_window,
            stride=stride,
            geodesic=geodesic,
            geodesic_method=geodesic_method,
            surface_tension=surface_tension,
            bending_modulus=bending_modulus,
            spontaneous_curvature=spontaneous_curvature,
            saddle_splay_modulus=saddle_splay_modulus,
        )
        per_frame.append(profiles)

    return CurvatureTimelapse(
        spatial_ndim=spatial_ndim,
        n_frames=int(relabeled.shape[0]),
        relabeled=relabeled,
        per_frame=per_frame,
        tracks=tracks,
        spacing=spatial_spacing,
    )


# ================================================================ output


def save_curvature_timelapse_tiffs(
    timelapse: CurvatureTimelapse,
    *,
    out_dir: Union[str, Path],
    stem: str,
    fields: tuple[str, ...] = ("kappa", "pressure", "bending_density"),
    splat_radius: int = 0,
    reduce: str = "mean",
    write_tracks: bool = True,
    write_csv: bool = True,
) -> dict[str, Path]:
    """Write the timelapse curvature volume(s) and the relabeled tracks.

    Outputs in ``out_dir``:

    - ``<stem>_<field>.tif`` — float32 stack, shape ``(T, …)``, one per
      field (skipped if not populated, e.g. pressure without γ).
    - ``<stem>_tracks.tif`` — the relabeled label timelapse (uint32),
      so napari can show stable cell IDs alongside the curvature map.
    - ``<stem>_tracks.csv`` — one row per (frame, track) with
      summary statistics for plotting in pandas / Excel.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    frame_shape = timelapse.relabeled.shape[1:]
    for fname in fields:
        # Skip fields no profile populated.
        has_any = any(
            getattr(p, fname, None) is not None
            for frame_profiles in timelapse.per_frame
            for p in frame_profiles.values()
        )
        if not has_any:
            continue
        stack = np.zeros((timelapse.n_frames,) + frame_shape, dtype=np.float32)
        for t, profiles in enumerate(timelapse.per_frame):
            if not profiles:
                continue
            stack[t] = render_curvature_volume(
                profiles,
                shape=frame_shape,
                spacing=timelapse.spacing,
                field=fname,
                splat_radius=splat_radius,
                reduce=reduce,
            )
        path = out_dir / f"{stem}_{fname}.tif"
        tifffile.imwrite(path, stack)
        written[fname] = path

    if write_tracks:
        track_path = out_dir / f"{stem}_tracks.tif"
        tifffile.imwrite(track_path, timelapse.relabeled.astype(np.uint32))
        written["tracks"] = track_path

    if write_csv:
        csv_path = out_dir / f"{stem}_tracks.csv"
        rows = timelapse.summary_table()
        if rows:
            keys = sorted({k for r in rows for k in r.keys()})
            # Keep frame, track_id, label_id first for readability.
            head = [k for k in ("frame", "track_id", "label_id") if k in keys]
            tail = [k for k in keys if k not in head]
            fieldnames = head + tail
            with csv_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
        else:
            csv_path.write_text("frame,track_id\n")
        written["csv"] = csv_path

    return written


# =========================================================== folder helper


def process_timelapse_folder(
    label_dir: Union[str, Path],
    *,
    spatial_ndim: int,
    out_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*.tif",
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
    n_window: int = 21,
    stride: int = 5,
    geodesic: bool = True,
    surface_tension: Optional[float] = None,
    bending_modulus: Optional[float] = None,
    spontaneous_curvature: float = 0.0,
    saddle_splay_modulus: Optional[float] = None,
    max_link_distance: Optional[float] = None,
    link_method: str = "hungarian",
    link_features: tuple[str, ...] = ("centroid",),
    link_weights: Optional[dict[str, float]] = None,
    intensity_dir: Optional[Union[str, Path]] = None,
    fields: tuple[str, ...] = ("kappa", "pressure", "bending_density"),
    splat_radius: int = 0,
    reduce: str = "mean",
    verbose: bool = True,
) -> dict[str, dict[str, Path]]:
    """Walk a folder of timelapse label TIFFs, render curvature maps + tracks.

    Mirrors :func:`process_label_folder` but for timelapses. Each file
    must be ``(T, Y, X)`` (``spatial_ndim=2``) or ``(T, Z, Y, X)``
    (``spatial_ndim=3``). Output folder defaults to
    ``<label_dir>.parent / "curvature"``.
    """
    label_dir = Path(label_dir)
    if not label_dir.is_dir():
        raise NotADirectoryError(f"{label_dir} is not a directory")
    if out_dir is None:
        out_dir = label_dir.parent / "curvature"
    out_dir = Path(out_dir)

    results: dict[str, dict[str, Path]] = {}
    for label_path in sorted(label_dir.glob(pattern)):
        labels = tifffile.imread(label_path)
        if labels.ndim != spatial_ndim + 1:
            if verbose:
                print(
                    f"  ! skipping {label_path.name}: ndim={labels.ndim}, "
                    f"expected {spatial_ndim + 1} for spatial_ndim={spatial_ndim}"
                )
            continue
        intensity = None
        if intensity_dir is not None:
            int_path = Path(intensity_dir) / label_path.name
            if int_path.exists():
                intensity = tifffile.imread(int_path)
                if intensity.shape != labels.shape:
                    raise ValueError(
                        f"intensity {intensity.shape} does not match "
                        f"labels {labels.shape} for {label_path.name}"
                    )
        tl = compute_curvature_timelapse(
            labels,
            spatial_ndim=spatial_ndim,
            spacing=spacing,
            n_window=n_window,
            stride=stride,
            geodesic=geodesic,
            surface_tension=surface_tension,
            bending_modulus=bending_modulus,
            spontaneous_curvature=spontaneous_curvature,
            saddle_splay_modulus=saddle_splay_modulus,
            max_link_distance=max_link_distance,
            link_method=link_method,
            link_features=link_features,
            link_weights=link_weights,
            intensity_image=intensity,
        )
        written = save_curvature_timelapse_tiffs(
            tl,
            out_dir=out_dir,
            stem=label_path.stem,
            fields=fields,
            splat_radius=splat_radius,
            reduce=reduce,
        )
        results[label_path.name] = written
        if verbose:
            wrote = ", ".join(p.name for p in written.values()) or "(no outputs)"
            print(
                f"  → {label_path.name}: T={tl.n_frames}, "
                f"{len(tl.tracks)} tracks  →  {wrote}"
            )
    return results
