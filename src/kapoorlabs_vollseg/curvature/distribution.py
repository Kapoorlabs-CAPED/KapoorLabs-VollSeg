"""Aggregate per-window curvature samples into a histogram + summary stats.

Two input kinds, one output type:

- ``dict[label_id, CurvatureProfile]`` — output of
  :func:`compute_curvature` on a static volume. All labels' κ values
  are stacked into a single 1D histogram.
- :class:`CurvatureTimelapse` — output of
  :func:`compute_curvature_timelapse`. One histogram per frame is
  computed against a *shared* bin grid so the resulting ``(T, n_bins)``
  matrix can be plotted as a kymograph-style heatmap.

The summary stats (median, p25, p75, mean, std) accompany the
histogram and are scalar / list-per-frame respectively.

The same routine works on any per-window scalar field —
``field="kappa"`` is the default, but ``"pressure"`` /
``"bending_density"`` / ``"radii"`` all work when those columns were
populated upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .profile import CurvatureProfile
from .timelapse import CurvatureTimelapse


@dataclass
class CurvatureDistribution:
    """Histogram + descriptive stats for a curvature run.

    Attributes
    ----------
    bin_edges
        ``(n_bins + 1,)`` shared by every histogram in this object.
    counts
        ``(n_bins,)`` for a static run, ``(T, n_bins)`` for a timelapse.
    n_samples
        Total number of contributing per-window values; scalar for a
        static run, ``(T,)`` array for a timelapse.
    summary
        Single dict ``{"median", "p25", "p75", "mean", "std"}`` for a
        static run, ``list[dict]`` (one per frame) for a timelapse.
    field
        Which :class:`CurvatureProfile` attribute was histogrammed.
    is_timelapse
        ``True`` iff ``counts.ndim == 2``.
    """

    bin_edges: np.ndarray
    counts: np.ndarray
    n_samples: Union[int, np.ndarray]
    summary: Union[dict, list]
    field: str = "kappa"

    @property
    def is_timelapse(self) -> bool:
        return self.counts.ndim == 2

    @property
    def n_frames(self) -> int:
        return int(self.counts.shape[0]) if self.is_timelapse else 1

    @property
    def n_bins(self) -> int:
        return int(self.counts.shape[-1])

    @property
    def bin_centers(self) -> np.ndarray:
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    def to_dict(self) -> dict:
        out = {
            "bin_edges": self.bin_edges,
            "bin_centers": self.bin_centers,
            "counts": self.counts,
            "n_samples": self.n_samples,
            "field": self.field,
            "is_timelapse": self.is_timelapse,
        }
        return out


# ============================================================ aggregation


def _collect_field(
    profiles: dict[int, CurvatureProfile],
    field: str,
) -> np.ndarray:
    """Concatenate ``profile.<field>`` across every profile in a frame."""
    chunks = []
    for prof in profiles.values():
        vals = getattr(prof, field, None)
        if vals is None:
            continue
        arr = np.asarray(vals, dtype=np.float64).ravel()
        if arr.size:
            chunks.append(arr)
    if not chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chunks)


def _summary(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            k: float("nan")
            for k in ("median", "p25", "p75", "mean", "std", "min", "max")
        }
    return {
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _resolve_bins_and_range(
    samples_iter: list[np.ndarray],
    bins: Union[int, np.ndarray],
    value_range: Optional[tuple[float, float]],
) -> tuple[int, tuple[float, float]]:
    """Compute final ``(n_bins, (lo, hi))`` shared across all histograms."""
    if isinstance(bins, np.ndarray) and bins.ndim == 1 and bins.size > 1:
        # User supplied explicit edges — honour them, ignore value_range.
        return int(bins.size - 1), (float(bins[0]), float(bins[-1]))

    n_bins = int(bins)
    if value_range is not None:
        return n_bins, (float(value_range[0]), float(value_range[1]))

    pooled = (
        np.concatenate([s for s in samples_iter if s.size])
        if any(s.size for s in samples_iter)
        else np.array([0.0, 1.0])
    )
    if pooled.size == 0:
        return n_bins, (0.0, 1.0)
    lo, hi = float(pooled.min()), float(pooled.max())
    if hi == lo:  # degenerate (constant κ)
        hi = lo + 1e-6
    return n_bins, (lo, hi)


# ============================================================ public API


def compute_curvature_distribution(
    source: Union[dict[int, CurvatureProfile], CurvatureTimelapse],
    *,
    field: str = "kappa",
    bins: Union[int, np.ndarray] = 50,
    value_range: Optional[tuple[float, float]] = None,
) -> CurvatureDistribution:
    """Build a curvature histogram from one static or one timelapse run.

    For a **static** ``dict[label_id, CurvatureProfile]`` (the output of
    :func:`compute_curvature`), every label's per-window values are
    pooled and a single 1D histogram is returned.

    For a :class:`CurvatureTimelapse`, the per-frame label dicts are
    each pooled and one histogram per frame is computed against a
    shared bin grid — so ``counts`` is ``(T, n_bins)`` and the matrix
    can be plotted as a kymograph (κ along x, time along y).

    Parameters
    ----------
    source
        Output of :func:`compute_curvature` or
        :func:`compute_curvature_timelapse`.
    field
        Per-window attribute to histogram (``"kappa"``, ``"pressure"``,
        ``"bending_density"``, ``"radii"``, …). Profiles missing this
        field are silently skipped.
    bins
        Either an int (number of bins) or a 1D array of explicit edges.
        Defaults to 50.
    value_range
        Optional explicit ``(lo, hi)`` for the bin edges; otherwise the
        pooled min/max across the whole input is used so every frame
        in a timelapse shares the same axis.

    Returns
    -------
    :class:`CurvatureDistribution`
    """
    # --- static dict[label_id, CurvatureProfile] -----------------------
    if isinstance(source, dict):
        values = _collect_field(source, field)
        n_bins, val_range = _resolve_bins_and_range([values], bins, value_range)
        counts, edges = np.histogram(values, bins=n_bins, range=val_range)
        return CurvatureDistribution(
            bin_edges=edges.astype(np.float64),
            counts=counts.astype(np.int64),
            n_samples=int(values.size),
            summary=_summary(values),
            field=field,
        )

    # --- timelapse -----------------------------------------------------
    if isinstance(source, CurvatureTimelapse):
        per_frame_values = [
            _collect_field(frame_profiles, field) for frame_profiles in source.per_frame
        ]
        n_bins, val_range = _resolve_bins_and_range(
            per_frame_values,
            bins,
            value_range,
        )
        # Histogram each frame against the shared edges.
        first_counts, edges = np.histogram(
            per_frame_values[0] if per_frame_values else np.empty(0),
            bins=n_bins,
            range=val_range,
        )
        counts = np.zeros((len(per_frame_values), n_bins), dtype=np.int64)
        counts[0] = first_counts
        for t in range(1, len(per_frame_values)):
            counts[t], _ = np.histogram(
                per_frame_values[t],
                bins=edges,
            )
        n_samples = np.asarray([v.size for v in per_frame_values], dtype=np.int64)
        summary = [_summary(v) for v in per_frame_values]
        return CurvatureDistribution(
            bin_edges=edges.astype(np.float64),
            counts=counts,
            n_samples=n_samples,
            summary=summary,
            field=field,
        )

    raise TypeError(
        "compute_curvature_distribution expects a "
        "dict[label_id, CurvatureProfile] or a CurvatureTimelapse, "
        f"got {type(source).__name__}"
    )
