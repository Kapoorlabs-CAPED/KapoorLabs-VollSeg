"""Project per-label curvature profiles back into the input image coordinate
system and write them as TIFFs.

The boundary/surface points in :class:`CurvatureProfile` are stored in
*physical* units (multiplied by ``spacing`` at extraction time). To
visualise the curvature map on top of the original image we divide
back by ``spacing`` to recover voxel indices, then splat the κ /
pressure / bending-density values at those indices. The resulting
volume has exactly the same shape as the label image and can be loaded
as a second channel in napari, Fiji, Imaris, etc.

Two entry points:

- :func:`render_curvature_volume` — in-memory ``np.ndarray``.
- :func:`save_curvature_tiffs` — writes one TIFF per requested field.
- :func:`process_label_folder` — batch convenience: walk a folder of
  label images and emit a sibling ``curvature/`` folder of TIFFs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
from collections.abc import Iterable

import numpy as np
import tifffile

from .api import compute_curvature
from .profile import CurvatureProfile


# =================================================================== utils


def _ball_offsets(radius: int, ndim: int) -> np.ndarray:
    """Integer offsets to every voxel inside a (open) ball of given radius."""
    r = int(radius)
    grids = np.meshgrid(*([np.arange(-r, r + 1)] * ndim), indexing="ij")
    coords = np.stack([g.ravel() for g in grids], axis=1)
    mask = np.sum(coords**2, axis=1) <= r * r
    return coords[mask].astype(np.int64)


def _splat_max_abs(acc: np.ndarray, coords: np.ndarray, values: np.ndarray) -> None:
    """Update ``acc`` in place with values, keeping the largest |x| per voxel.

    ``np.maximum.at`` doesn't have an "absolute" variant, so this loops in
    Python — only used when ``reduce='max_abs'`` is explicitly requested.
    """
    for vc, v in zip(coords, values):
        idx = tuple(int(c) for c in vc)
        if abs(v) > abs(acc[idx]):
            acc[idx] = v


# ============================================================== rendering


def render_curvature_volume(
    profiles: dict[int, CurvatureProfile],
    *,
    shape: tuple[int, ...],
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
    field: str = "kappa",
    splat_radius: int = 0,
    reduce: str = "mean",
) -> np.ndarray:
    """Project ``profile.<field>`` back into a volume of ``shape``.

    Parameters
    ----------
    profiles
        Output of :func:`compute_curvature`.
    shape
        Shape of the original label image. The returned array has this
        shape and ``dtype=float32``.
    spacing
        Per-axis voxel size that was passed to :func:`compute_curvature`.
        Used to convert the stored physical-unit centres back to voxel
        coordinates.
    field
        Attribute of :class:`CurvatureProfile` to splat. Common values:
        ``"kappa"``, ``"pressure"``, ``"bending_density"``. Profiles
        whose ``field`` is ``None`` are skipped.
    splat_radius
        If > 0, splat each value over a ball of that voxel radius (makes
        the boundary visible without zooming in). 0 → single voxel.
    reduce
        How to combine values from overlapping windows at the same voxel:

        - ``"mean"`` (default) — sum / count.
        - ``"max_abs"`` — keep the largest-magnitude value (preserves
          sign). Slower; uses a Python loop.
    """
    ndim = len(shape)
    spacing_arr = np.asarray(spacing[:ndim], dtype=np.float64)
    if reduce not in {"mean", "max_abs"}:
        raise ValueError(f"reduce must be 'mean' or 'max_abs', got {reduce!r}")

    acc = np.zeros(shape, dtype=np.float32)
    counts = np.zeros(shape, dtype=np.int32) if reduce == "mean" else None
    offsets = _ball_offsets(splat_radius, ndim) if splat_radius > 0 else None
    shape_arr = np.asarray(shape, dtype=np.int64)

    for profile in profiles.values():
        values = getattr(profile, field, None)
        if values is None or len(values) == 0:
            continue
        voxel_coords = np.round(profile.centers / spacing_arr).astype(np.int64)
        values = np.asarray(values, dtype=np.float32)

        if offsets is None:
            blocks: Iterable[tuple[np.ndarray, np.ndarray]] = [(voxel_coords, values)]
        else:
            blocks = ((voxel_coords + off, values) for off in offsets)

        for coords, vals in blocks:
            ok = np.all((coords >= 0) & (coords < shape_arr), axis=1)
            coords = coords[ok]
            vals = vals[ok]
            if coords.size == 0:
                continue
            idx = tuple(coords.T)
            if reduce == "mean":
                np.add.at(acc, idx, vals)
                np.add.at(counts, idx, 1)
            else:
                _splat_max_abs(acc, coords, vals)

    if reduce == "mean":
        out = np.divide(
            acc,
            counts,
            out=np.zeros_like(acc),
            where=counts > 0,
        )
        return out
    return acc


# ============================================================== TIFF output


def save_curvature_tiffs(
    profiles: dict[int, CurvatureProfile],
    *,
    shape: tuple[int, ...],
    out_dir: Union[str, Path],
    stem: str,
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
    fields: tuple[str, ...] = ("kappa", "pressure", "bending_density"),
    splat_radius: int = 0,
    reduce: str = "mean",
) -> dict[str, Path]:
    """Render and write one TIFF per requested field.

    Output filenames are ``<out_dir>/<stem>_<field>.tif`` and the volume
    shape matches the input label image so napari/Fiji can stack the
    layer over the original raw / labels.

    Fields not populated in any profile (e.g. ``pressure`` when
    ``surface_tension`` was not supplied) are silently skipped — no
    empty file is written.

    Returns the dict of written paths keyed by field name.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for field in fields:
        if not any(getattr(p, field, None) is not None for p in profiles.values()):
            continue
        volume = render_curvature_volume(
            profiles,
            shape=shape,
            spacing=spacing,
            field=field,
            splat_radius=splat_radius,
            reduce=reduce,
        )
        path = out_dir / f"{stem}_{field}.tif"
        tifffile.imwrite(path, volume.astype(np.float32))
        written[field] = path
    return written


# ====================================================== folder convenience


def process_label_folder(
    label_dir: Union[str, Path],
    *,
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
    fields: tuple[str, ...] = ("kappa", "pressure", "bending_density"),
    splat_radius: int = 0,
    reduce: str = "mean",
    verbose: bool = True,
) -> dict[str, dict[str, Path]]:
    """Compute curvature for every label TIFF in ``label_dir`` and write the
    rendered maps to a sibling folder.

    By default the output folder is ``<label_dir>.parent / "curvature"``
    so a layout like::

        experiment/
          segmentation/   <- input label TIFFs
          raw/            <- original images

    grows a third sibling::

        experiment/
          segmentation/
          raw/
          curvature/      <- one *_kappa.tif (and optionally
                             *_pressure.tif, *_bending_density.tif)
                             per input label TIFF, same shape & axis order

    Pass ``out_dir`` to override the destination.

    Returns
    -------
    dict
        ``{label_filename: {field: written_path}}``
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
        if labels.ndim not in (2, 3):
            if verbose:
                print(f"  ! skipping {label_path.name}: ndim={labels.ndim}")
            continue
        profiles = compute_curvature(
            labels,
            spacing=spacing,
            n_window=n_window,
            stride=stride,
            geodesic=geodesic,
            surface_tension=surface_tension,
            bending_modulus=bending_modulus,
            spontaneous_curvature=spontaneous_curvature,
            saddle_splay_modulus=saddle_splay_modulus,
        )
        written = save_curvature_tiffs(
            profiles,
            shape=labels.shape,
            out_dir=out_dir,
            stem=label_path.stem,
            spacing=spacing,
            fields=fields,
            splat_radius=splat_radius,
            reduce=reduce,
        )
        results[label_path.name] = written
        if verbose:
            wrote = ", ".join(p.name for p in written.values()) or "(no fields)"
            print(f"  → {label_path.name}: {len(profiles)} labels  →  {wrote}")
    return results
