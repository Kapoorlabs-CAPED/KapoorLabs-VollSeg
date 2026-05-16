"""Line-profile sampling + timelapse recording.

A *line profile* is the polyline the user draws in napari's Shapes
layer. For each timepoint we:

1. Slice the displayed plane out of the current 2D / 3D / 3D+T volume
   (the napari viewer's current Z if there is one).
2. For each line, sample the underlying image along that polyline via
   :func:`skimage.measure.profile_line`.
3. Stack the per-timepoint samples into a ``(T, L)`` kymograph per
   line and per channel (curvature, intensity).

Everything in this module is pure-numpy — no napari / Qt — so it's
easy to unit-test offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from skimage.measure import profile_line


@dataclass
class LineProfile:
    """One polyline drawn by the user.

    ``points`` are in ``(y, x)`` order (napari convention for Shapes
    layers in 2D). For polylines with > 2 vertices we sample each
    segment in order and concatenate, so the kymograph reads
    left-to-right along the drawn path.
    """

    line_id: int
    points: np.ndarray  # (N, 2), (y, x)
    linewidth: int = 1  # profile_line averaging window

    def length(self) -> float:
        diffs = np.diff(self.points, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))


def sample_polyline(
    image_2d: np.ndarray,
    points: np.ndarray,
    *,
    linewidth: int = 1,
    num_samples: Optional[int] = None,
) -> np.ndarray:
    """Sample ``image_2d`` along a multi-segment polyline.

    ``points`` is ``(N, 2)`` in ``(y, x)``. Segments are sampled with
    :func:`skimage.measure.profile_line` (linear interpolation,
    ``linewidth``-pixel averaging). If ``num_samples`` is given the
    final concatenated profile is resampled to that length so all
    lines in a kymograph share a length axis.
    """
    if image_2d.ndim != 2:
        raise ValueError(
            f"sample_polyline expects a 2D image, got ndim={image_2d.ndim}"
        )
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError(f"points must be (N, 2) with N >= 2, got shape {points.shape}")

    segments: list[np.ndarray] = []
    for src, dst in zip(points[:-1], points[1:]):
        seg = profile_line(
            image_2d,
            tuple(src),
            tuple(dst),
            linewidth=int(linewidth),
            order=1,
            mode="constant",
            cval=0.0,
        )
        if segments:
            seg = seg[1:]  # avoid duplicating shared vertex
        segments.append(seg)
    profile = np.concatenate(segments).astype(np.float32)

    if num_samples is not None and num_samples > 1:
        # Linear resample so every line has the same `L` dimension.
        x_old = np.linspace(0.0, 1.0, profile.size)
        x_new = np.linspace(0.0, 1.0, int(num_samples))
        profile = np.interp(x_new, x_old, profile).astype(np.float32)
    return profile


@dataclass
class TimelapseRecording:
    """Output of :func:`record_timelapse_kymograph`."""

    # ``kymographs[channel][line_id]`` = (T, L) float32 array.
    kymographs: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    line_lengths: dict[int, float] = field(default_factory=dict)
    n_frames: int = 0
    samples_per_line: int = 0


def _frame_at(
    volume: np.ndarray,
    t: int,
    z: Optional[int],
) -> np.ndarray:
    """Slice out the 2D YX plane displayed at (t, z) for any ndim ≤ 4."""
    if volume.ndim == 2:
        return volume
    if volume.ndim == 3:
        # Ambiguous: (T, Y, X) or (Z, Y, X). We treat axis 0 as the
        # "time-or-z" axis indexed by ``t`` when ``z is None``, else
        # by ``z`` — the caller decides based on what's wired up.
        return volume[t if z is None else z]
    if volume.ndim == 4:
        return volume[t, z if z is not None else 0]
    raise ValueError(f"Unsupported volume ndim={volume.ndim}")


def record_timelapse_kymograph(
    volumes: dict[str, np.ndarray],
    lines: list[LineProfile],
    *,
    n_frames: int,
    z: Optional[int] = None,
    samples_per_line: int = 256,
) -> TimelapseRecording:
    """Build per-line ``(T, L)`` kymographs for every channel in ``volumes``.

    Parameters
    ----------
    volumes
        ``{"curvature": ..., "intensity": ...}`` (any string keys).
        Each value is a 2D, 3D, or 4D numpy array. A 2D volume
        produces ``T=1`` kymographs (it's a static frame).
    lines
        Polylines to sample.
    n_frames
        How many time points to walk through. Use ``1`` for a single
        static frame.
    z
        Z-slice to take from a 4D ``(T, Z, Y, X)`` volume. ``None``
        for 3D ``(T, Y, X)`` or 2D ``(Y, X)``.
    samples_per_line
        Length axis of each kymograph; every line gets resampled to
        this length so they can be stacked / compared.
    """
    out: dict[str, dict[int, np.ndarray]] = {key: {} for key in volumes}
    for key in volumes:
        for line in lines:
            out[key][line.line_id] = np.zeros(
                (n_frames, samples_per_line),
                dtype=np.float32,
            )

    for t in range(n_frames):
        for key, vol in volumes.items():
            frame = _frame_at(vol, t, z)
            for line in lines:
                out[key][line.line_id][t] = sample_polyline(
                    frame,
                    line.points,
                    linewidth=line.linewidth,
                    num_samples=samples_per_line,
                )

    return TimelapseRecording(
        kymographs=out,
        line_lengths={ln.line_id: ln.length() for ln in lines},
        n_frames=int(n_frames),
        samples_per_line=int(samples_per_line),
    )
