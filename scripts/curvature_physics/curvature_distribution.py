"""
Per-volume curvature distribution.

For each label TIFF in ``label_dir``:

- compute κ (and optional pressure / bending density) via
  :func:`kapoorlabs_vollseg.curvature.compute_curvature` for static
  inputs, or :func:`compute_curvature_timelapse` for ``timelapse=True``.
- aggregate the per-window values across **every** label in that
  volume / frame into a histogram via
  :func:`compute_curvature_distribution`.
- write the histogram + summary stats to ``output_dir``:
    - ``<stem>_<field>_hist.csv``     bin_center, count (+ frame for timelapse)
    - ``<stem>_<field>_hist.tif``     (T, n_bins) heatmap for timelapse only
    - ``<stem>_<field>_summary.csv``  median / p25 / p75 / mean / std (per frame)

Run::

    python curvature_distribution.py \\
        experiment_data_paths.base_data_dir=/path/to/exp \\
        experiment_data_paths.label_dir=segmentation \\
        experiment_data_paths.output_dir=curvature_dist \\
        parameters.timelapse=true \\
        parameters.field=kappa
"""

from __future__ import annotations

import csv
import os
from glob import glob
from pathlib import Path

import hydra
import numpy as np
import tifffile
from hydra.core.config_store import ConfigStore

from kapoorlabs_vollseg.curvature import (
    compute_curvature,
    compute_curvature_distribution,
    compute_curvature_timelapse,
)

from scenarios import CurvatureScenario


configstore = ConfigStore.instance()
configstore.store(name="CurvatureScenario", node=CurvatureScenario)


def _write_histogram_csv(path: Path, dist) -> None:
    """Two-column CSV for static, three-column for timelapse."""
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        if dist.is_timelapse:
            w.writerow(["frame", "bin_center", "count"])
            for t in range(dist.n_frames):
                for c, n in zip(dist.bin_centers, dist.counts[t]):
                    w.writerow([t, float(c), int(n)])
        else:
            w.writerow(["bin_center", "count"])
            for c, n in zip(dist.bin_centers, dist.counts):
                w.writerow([float(c), int(n)])


def _write_summary_csv(path: Path, dist) -> None:
    fieldnames = [
        "frame",
        "n_samples",
        "median",
        "p25",
        "p75",
        "mean",
        "std",
        "min",
        "max",
    ]
    rows = []
    if dist.is_timelapse:
        for t, s in enumerate(dist.summary):
            rows.append({"frame": t, "n_samples": int(dist.n_samples[t]), **s})
    else:
        rows.append({"frame": 0, "n_samples": int(dist.n_samples), **dist.summary})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


@hydra.main(
    config_path="../conf",
    config_name="scenario_curvature_distribution",
    version_base="1.3",
)
def main(config: CurvatureScenario):
    p = config.parameters
    paths = config.experiment_data_paths

    label_dir = os.path.join(paths.base_data_dir, paths.label_dir)
    output_dir = Path(paths.base_data_dir) / paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    spacing = tuple(float(v) for v in p.spacing[: p.spatial_ndim])
    value_range = tuple(p.value_range) if p.value_range is not None else None

    files = sorted(glob(os.path.join(label_dir, p.file_type)))
    print(f"Found {len(files)} label TIFFs in {label_dir}")
    print(f"  field={p.field}, n_bins={p.n_bins}, timelapse={p.timelapse}")

    for label_path in files:
        stem = Path(label_path).stem
        print(f"\nProcessing: {stem}")
        labels = tifffile.imread(label_path)
        expected_ndim = p.spatial_ndim + (1 if p.timelapse else 0)
        if labels.ndim != expected_ndim:
            print(f"  ! skipping (ndim={labels.ndim}, expected {expected_ndim})")
            continue

        if p.timelapse:
            tl = compute_curvature_timelapse(
                labels,
                spatial_ndim=p.spatial_ndim,
                spacing=spacing,
                n_window=p.n_window,
                stride=p.stride,
                geodesic=p.geodesic,
                surface_tension=p.surface_tension,
                bending_modulus=p.bending_modulus,
                spontaneous_curvature=p.spontaneous_curvature,
                saddle_splay_modulus=p.saddle_splay_modulus,
                max_link_distance=p.max_link_distance,
            )
            dist = compute_curvature_distribution(
                tl,
                field=p.field,
                bins=p.n_bins,
                value_range=value_range,
            )
            print(
                f"  timelapse: T={dist.n_frames}, bins={dist.n_bins}, "
                f"total samples={int(np.sum(dist.n_samples))}"
            )
        else:
            profiles = compute_curvature(
                labels,
                spacing=spacing,
                n_window=p.n_window,
                stride=p.stride,
                geodesic=p.geodesic,
                surface_tension=p.surface_tension,
                bending_modulus=p.bending_modulus,
                spontaneous_curvature=p.spontaneous_curvature,
                saddle_splay_modulus=p.saddle_splay_modulus,
            )
            dist = compute_curvature_distribution(
                profiles,
                field=p.field,
                bins=p.n_bins,
                value_range=value_range,
            )
            print(f"  static: {len(profiles)} labels, samples={int(dist.n_samples)}")

        hist_csv = output_dir / f"{stem}_{p.field}_hist.csv"
        summary_csv = output_dir / f"{stem}_{p.field}_summary.csv"
        _write_histogram_csv(hist_csv, dist)
        _write_summary_csv(summary_csv, dist)
        print(f"  → {hist_csv.name}, {summary_csv.name}")

        if dist.is_timelapse:
            # (T, n_bins) heatmap — open as a napari Image to see how the
            # distribution evolves in time.
            tif_path = output_dir / f"{stem}_{p.field}_hist.tif"
            tifffile.imwrite(tif_path, dist.counts.astype(np.float32))
            print(f"  → {tif_path.name} (heatmap, shape={dist.counts.shape})")

    print("\nDone.")


if __name__ == "__main__":
    main()
