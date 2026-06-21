"""Per-timepoint nuclei statistics: StarDist (this checkpoint) vs Keras.

Predicts the winner StarDist model on the **same** first/mid/last
``subset_n_each`` timepoints the sweep already used, then for each
timepoint computes per-instance regionprops (volume in voxels,
equivalent-sphere radius, surface area) for both the StarDist
prediction and the keras reference. Aggregates to one row per
timepoint per source and writes a long-format CSV at::

    <compare_data_paths.out_dir>/<input_stem>.compare.csv

The companion notebook ``compare_stardist_vs_keras.ipynb`` loads the
CSV and plots bar charts so you can eyeball whether the winner over-
or under-segments relative to keras.

Run::

    python compare-stardist-vs-keras.py train_data_paths=xenopus_jeanzay

Override the model folder or input data on the CLI in the usual Hydra
way::

    python compare-stardist-vs-keras.py \\
        train_data_paths.log_path=/lustre/.../stardist_sweep_adam_lr1p0e-3_noscheduler \\
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

from kapoorlabs_vollseg import StarDistSegmenter, predict_timelapse
from kapoorlabs_vollseg._backbones._config import read_thresholds

from scenario_compare_stardist_vs_keras import CompareStarDistVsKerasScenario


ConfigStore.instance().store(
    name="CompareStarDistVsKerasScenario",
    node=CompareStarDistVsKerasScenario,
)


def _subset_timepoints(T: int, n_each: int) -> list[int]:
    """Same first/mid/last subset the sweep uses — keeps CSV indices
    consistent with ``sweep_predict_summary.csv`` rows."""
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
    """Surface area of a 3D binary mask via marching cubes.

    Pads the mask by 1 voxel so labels touching the bbox edge produce
    a closed surface. Returns 0 for masks with no foreground.
    """
    if not mask.any():
        return 0.0
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5)
    except (RuntimeError, ValueError):
        # Mask too small for marching cubes — fall back to voxel
        # boundary count (every voxel with at least one non-self
        # neighbour). Approximate but doesn't fail.
        boundary = 0
        for axis in range(mask.ndim):
            for delta in (-1, 1):
                shifted = np.roll(mask, delta, axis=axis)
                boundary += int((mask & ~shifted).sum())
        return float(boundary)
    return float(mesh_surface_area(verts, faces))


def _frame_stats(label_image: np.ndarray) -> dict:
    """Per-frame nuclei statistics from a 3D label image.

    Returns aggregates: ``n_nuclei``, ``mean_volume_vox``,
    ``total_volume_vox``, ``mean_radius_vox``, ``mean_surface_area``,
    ``total_surface_area``. Empty frames return zeros so the CSV
    schema stays flat.
    """
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
        # Equivalent-sphere radius from voxel volume: V = (4/3)π r³.
        radius = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        # Surface area via marching cubes on the per-cell binary mask
        # cropped to its bounding box.
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
    config_name="scenario_compare_stardist_vs_keras",
    version_base="1.3",
)
def main(config: CompareStarDistVsKerasScenario):
    p = config.parameters
    paths_train = config.train_data_paths
    paths_compare = config.compare_data_paths

    input_dir = Path(paths_compare.input_dir)
    keras_dir = Path(paths_compare.keras_dir)
    out_dir = Path(paths_compare.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = paths_train.log_path
    print(f"StarDist model: {log_path}")
    print(f"Input dir:      {input_dir}")
    print(f"Keras dir:      {keras_dir}")
    print(f"Output dir:     {out_dir}")

    star = StarDistSegmenter.from_folder(log_path, batch_size=p.batch_size)
    # ``from_folder`` only reads architecture knobs — NOT thresholds.
    # Read the tuned ``(prob_thresh, nms_thresh)`` separately via
    # ``read_thresholds`` and pass them explicitly to every predict
    # call below. Without this the comparison would run at the
    # ``StarDistSegmenter.__init__`` defaults (0.5 / 0.4) regardless
    # of what the optimiser wrote into ``training_config.json``.
    overrides = read_thresholds(log_path)
    prob_thresh = overrides.get("prob_thresh", star.prob_thresh)
    nms_thresh = overrides.get("nms_thresh", star.nms_thresh)
    print(
        f"Thresholds:     prob_thresh={prob_thresh}  nms_thresh={nms_thresh}  "
        f"({'tuned' if overrides else 'default — no training_config.json'})"
    )
    n_tiles = tuple(p.n_tiles)

    input_files = sorted(input_dir.glob(paths_compare.input_pattern))
    if not input_files:
        raise FileNotFoundError(
            f"No files matching {paths_compare.input_pattern!r} in {input_dir}"
        )
    print(f"Found {len(input_files)} input file(s)")

    for j, f in enumerate(input_files):
        stem = Path(f.name).stem
        csv_path = out_dir / f"{stem}.compare.csv"
        if csv_path.is_file() and not p.force:
            print(
                f"[{j + 1}/{len(input_files)}] {csv_path.name} exists — skipping (force=false)"
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

        # ── 1. predict on the subset
        vol_subset = vol[keras_indices]
        out = predict_timelapse(
            star,
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
        stardist_labels = out["labels"]  # (T_subset, Z, Y, X)
        if stardist_labels is None:
            print("   model returned no labels — skipping")
            continue
        print(f"   StarDist labels stack: {stardist_labels.shape}")

        # Save the StarDist prediction TIFF alongside the CSV — uint16
        # (matches the sweep + keras label TIFFs so you can drag-drop
        # all three into napari at the exact same T-indices).
        sd_tif_path = out_dir / f"{stem}.stardist.tif"
        imwrite(
            sd_tif_path,
            np.ascontiguousarray(stardist_labels, dtype=np.uint16),
            bigtiff=True,
        )
        print(f"   wrote {sd_tif_path}  ({stardist_labels.shape}, uint16)")

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
        print(f"   Keras labels stack:    {keras_labels.shape}")
        del keras_full

        # ── 3. per-frame stats for both sources
        rows: list[dict] = []
        for i, t in enumerate(tqdm(keras_indices, desc="stats", unit="frame")):
            sd_stats = _frame_stats(np.asarray(stardist_labels[i]))
            kr_stats = _frame_stats(np.asarray(keras_labels[i]))
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
