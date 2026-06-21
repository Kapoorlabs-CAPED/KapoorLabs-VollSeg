"""Per-timepoint nuclei statistics: ROI Mask-UNet → StarDist vs Keras.

Mirrors :mod:`compare-stardist-vs-keras` but wraps the StarDist
checkpoint in an :class:`ROIPipeline` whose ROI is produced by a
single, fixed Mask-UNet (path from ``train_data_paths.mask_unet_log_path``).
Matches the keras ``VollSeg.utils.VollSeg2D`` flow exactly — Mask-UNet
ROI first, then StarDist inside the ROI bounding box.

Useful for **early embryo timepoints** where most of the volume is
empty space: cropping to the ROI before percentile-normalisation
fixes the saturation issue that otherwise pushes the StarDist
distance head to predict miscalibrated polyhedra on near-empty
frames.

Output layout (same as the non-ROI version, distinct filenames so
both sets coexist in the same out_dir)::

    <compare_data_paths.out_dir>/
    ├── <input_stem>.compare_roi.csv      ← per-frame stats CSV
    └── <input_stem>.roi_stardist.tif     ← prediction TIFF (uint16)

Run::

    python compare-roi-stardist-vs-keras.py

Override the model folders or input data the usual Hydra way::

    python compare-roi-stardist-vs-keras.py \\
        train_data_paths.log_path=/lustre/.../models_stardist_pytorch \\
        train_data_paths.mask_unet_log_path=/lustre/.../models_maskunet_pytorch \\
        compare_data_paths.input_dir=/lustre/.../demo_data
"""

from __future__ import annotations

import csv
from pathlib import Path

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from skimage.measure import marching_cubes, mesh_surface_area, regionprops
from tifffile import imread, imwrite
from tqdm import tqdm

from kapoorlabs_vollseg import (
    MaskUNetSegmenter,
    ROIPipeline,
    StarDistSegmenter,
    predict_timelapse,
)
from kapoorlabs_vollseg._backbones._config import read_thresholds

from scenario_compare_roi_stardist_vs_keras import (
    CompareRoiStarDistVsKerasScenario,
)


ConfigStore.instance().store(
    name="CompareRoiStarDistVsKerasScenario",
    node=CompareRoiStarDistVsKerasScenario,
)


def _subset_timepoints(T: int, n_each: int) -> list[int]:
    """Same first/mid/last subset the sweep uses."""
    if T <= 0:
        return []
    if T <= 3 * n_each:
        return list(range(T))
    first = list(range(0, n_each))
    mid_start = max(n_each, (T - n_each) // 2)
    mid_start = min(mid_start, T - 2 * n_each)
    mid = list(range(mid_start, mid_start + n_each))
    last = list(range(T - n_each, T))
    return sorted(set(first + mid + last))


def _surface_area_marching_cubes(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5)
    except (RuntimeError, ValueError):
        boundary = 0
        for axis in range(mask.ndim):
            for delta in (-1, 1):
                shifted = np.roll(mask, delta, axis=axis)
                boundary += int((mask & ~shifted).sum())
        return float(boundary)
    return float(mesh_surface_area(verts, faces))


def _frame_stats(label_image: np.ndarray) -> dict:
    props = regionprops(label_image)
    if not props:
        return {
            "n_nuclei": 0,
            "mean_volume_vox": 0.0,
            "total_volume_vox": 0.0,
            "mean_radius_vox": 0.0,
            "mean_surface_area": 0.0,
            "total_surface_area": 0.0,
        }
    volumes, radii, areas = [], [], []
    for p in props:
        vol = int(p.area)
        radius = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        zlo, ylo, xlo, zhi, yhi, xhi = p.bbox
        cell_mask = label_image[zlo:zhi, ylo:yhi, xlo:xhi] == p.label
        area = _surface_area_marching_cubes(cell_mask)
        volumes.append(vol)
        radii.append(radius)
        areas.append(area)
    return {
        "n_nuclei": len(props),
        "mean_volume_vox": float(np.mean(volumes)),
        "total_volume_vox": float(np.sum(volumes)),
        "mean_radius_vox": float(np.mean(radii)),
        "mean_surface_area": float(np.mean(areas)),
        "total_surface_area": float(np.sum(areas)),
    }


@hydra.main(
    config_path="conf",
    config_name="scenario_compare_roi_stardist_vs_keras",
    version_base="1.3",
)
def main(config: CompareRoiStarDistVsKerasScenario):
    p = config.parameters
    paths_train = config.train_data_paths
    paths_compare = config.compare_data_paths

    input_dir = Path(paths_compare.input_dir)
    keras_dir = Path(paths_compare.keras_dir)
    out_dir = Path(paths_compare.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = paths_train.log_path
    mask_unet_log_path = paths_train.mask_unet_log_path
    print(f"StarDist model:  {log_path}")
    print(f"Mask-UNet model: {mask_unet_log_path}")
    print(f"Input dir:       {input_dir}")
    print(f"Keras dir:       {keras_dir}")
    print(f"Output dir:      {out_dir}")

    star = StarDistSegmenter.from_folder(log_path, batch_size=p.batch_size)
    mask_unet = MaskUNetSegmenter.from_folder(
        mask_unet_log_path, batch_size=p.batch_size
    )
    # ``from_folder`` only reads arch knobs — read tuned thresholds
    # separately and pass them through every predict call.
    overrides = read_thresholds(log_path)
    prob_thresh = overrides.get("prob_thresh", star.prob_thresh)
    nms_thresh = overrides.get("nms_thresh", star.nms_thresh)
    print(
        f"Thresholds:      prob_thresh={prob_thresh}  nms_thresh={nms_thresh}  "
        f"({'tuned' if overrides else 'default — no training_config.json'})"
    )

    # Build the ROI pipeline once and reuse across every input file.
    pipeline = ROIPipeline(roi_unet=mask_unet, downstream=star)
    n_tiles = tuple(p.n_tiles)

    input_files = sorted(input_dir.glob(paths_compare.input_pattern))
    if not input_files:
        raise FileNotFoundError(
            f"No files matching {paths_compare.input_pattern!r} in {input_dir}"
        )
    print(f"Found {len(input_files)} input file(s)")

    for j, f in enumerate(input_files):
        stem = Path(f.name).stem
        csv_path = out_dir / f"{stem}.compare_roi.csv"
        if csv_path.is_file() and not p.force:
            print(
                f"[{j + 1}/{len(input_files)}] {csv_path.name} exists — "
                f"skipping (force=false)"
            )
            continue

        print(f"\n[{j + 1}/{len(input_files)}] reading {f.name}")
        vol = imread(f)
        if vol.ndim != 4:
            print(f"   {f.name} is {vol.ndim}D, skipping (need 4D timelapse)")
            continue
        T = vol.shape[0]
        keras_indices = _subset_timepoints(T, p.subset_n_each)
        print(
            f"   full shape={tuple(vol.shape)} dtype={vol.dtype} → "
            f"comparing T-indices {keras_indices}"
        )

        # ── 1. ROI → StarDist prediction on the subset
        vol_subset = vol[keras_indices]
        out = predict_timelapse(
            pipeline,
            vol_subset,
            devices=p.devices,
            accelerator=p.accelerator,
            strategy=p.strategy,
            n_tiles=n_tiles,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
        )
        if not out:
            print("   non-rank-0 worker — skipping")
            continue
        roi_stardist_labels = out["labels"]  # (T_subset, Z, Y, X)
        if roi_stardist_labels is None:
            print("   pipeline returned no labels — skipping")
            continue
        print(f"   ROI-StarDist labels stack: {roi_stardist_labels.shape}")

        # Save the prediction TIFF alongside the CSV.
        tif_path = out_dir / f"{stem}.roi_stardist.tif"
        imwrite(
            tif_path,
            np.ascontiguousarray(roi_stardist_labels, dtype=np.uint16),
            bigtiff=True,
        )
        print(f"   wrote {tif_path}  ({roi_stardist_labels.shape}, uint16)")

        # ── 2. load keras reference at the same indices
        keras_path = keras_dir / f.name
        if not keras_path.is_file():
            print(f"   keras ref missing: {keras_path} — skipping comparison")
            continue
        keras_full = imread(keras_path)
        if keras_full.ndim != 4:
            print(
                f"   keras ref is {keras_full.ndim}D — expected 4D timelapse "
                f"({keras_path}); skipping"
            )
            continue
        keras_labels = keras_full[keras_indices]
        print(f"   Keras labels stack:        {keras_labels.shape}")
        del keras_full

        # ── 3. per-frame stats for both sources
        rows: list[dict] = []
        for i, t in enumerate(tqdm(keras_indices, desc="stats", unit="frame")):
            sd_stats = _frame_stats(np.asarray(roi_stardist_labels[i]))
            kr_stats = _frame_stats(np.asarray(keras_labels[i]))
            # ``source`` value remains "stardist" so the existing
            # plotting notebook works without any column rename.
            rows.append({"t_index": int(t), "source": "stardist", **sd_stats})
            rows.append({"t_index": int(t), "source": "keras", **kr_stats})

        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"   wrote {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
