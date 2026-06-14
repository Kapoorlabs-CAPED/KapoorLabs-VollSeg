"""Sweep prediction + analysis for ROI Mask-UNet → StarDist.

Same flow as ``sweep_predict_and_analyze.py``, but wraps every StarDist
model in an :class:`ROIPipeline` whose ROI is produced once per frame by
a single, fixed Mask-UNet checkpoint. Matches the keras
``VollSeg.utils.VollSeg2D`` flow (ROI Mask-UNet → StarDist inside the
ROI bbox).

Motivation: early timepoints in our timelapses are mostly black voxels;
percentile-normalising the whole volume there computes
``(pmin, pmax)`` mostly from background → foreground signal saturates →
StarDist hallucinates "every pixel == 1". Cropping to the ROI bbox
first means the percentiles are computed inside the cropped patch
(mostly signal), so the foreground/background contrast is recovered and
the network sees inputs in the same distribution it saw during training.

For every trained model folder under ``sweep_root/`` that survives the
``skip_name_substrings`` filter:

1. Load the StarDist model via :meth:`StarDistSegmenter.from_folder`.
2. Wrap ``mask_unet`` + ``star`` in an :class:`ROIPipeline`.
3. Run tiled / DDP-sharded prediction on every TIFF in ``input_dir``,
   subsetting to the first/middle/last ``subset_n_each`` timepoints.
4. Compare to the matching keras reference TIFF at exactly those
   timepoints.
5. Read the final ``val_loss`` / ``train_loss`` off ``metrics.csv``.
6. Aggregate into ``stardist_roi_sweeps/sweep_predict_summary.csv`` and
   print the best models by accuracy + by training val_loss.

Edit the paths block at the top, then
``python sweep_predict_roi_stardist_and_analyze.py``.
"""

# %%
from __future__ import annotations

import csv
import functools
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tifffile import TiffFile, imread, imwrite

from kapoorlabs_vollseg import (
    MaskUNetSegmenter,
    ROIPipeline,
    StarDistSegmenter,
    predict_timelapse,
)
from kapoorlabs_vollseg._backbones._config import read_thresholds
from kapoorlabs_vollseg.eval import matching_dataset

# Same opt-in gate as kapoorlabs_vollseg.stardist.inference uses. See
# sweep_predict_and_analyze.py for the reasoning. Set
# ``KAPOORLABS_VOLLSEG_PROGRESS=1`` to opt the bars back in.
_INTERACTIVE = os.environ.get("KAPOORLABS_VOLLSEG_PROGRESS") == "1"


def _is_rank_zero() -> bool:
    """Whether we're the rank-0 worker (single-GPU default returns True).

    Under DDP every rank runs this script but only rank 0 has the
    gathered per-frame results from ``predict_timelapse``."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0


print = functools.partial(print, flush=True)


def _human(t: float) -> str:
    if t < 60:
        return f"{t:.1f}s"
    m, s = divmod(t, 60)
    return f"{int(m)}m{s:.0f}s"


# %% ─── paths (edit per cluster) ─────────────────────────────────────
sweep_root = Path(
    "/lustre/fsn1/projects/rech/jsy/uzj81mi/models_stardist_pytorch_sweep/"
)
# Single ROI Mask-UNet checkpoint used to gate every StarDist model.
mask_unet_dir = Path("/lustre/fsn1/projects/rech/jsy/uzj81mi/models_maskunet_pytorch/")
input_dir = Path("/lustre/fsn1/projects/rech/jsy/uzj81mi/demo_data/")
input_pattern = "*.tif"

keras_dir = Path("/lustre/fsn1/projects/rech/jsy/uzj81mi/demo_data/keras_prediction/")

skip_name_substrings = ("lr1p0ep1", "_sgd_")

subset_n_each = 5

# Predictions live next to the input data, under
# ``<input_dir>/predictions_stardist_roi/<model>/<input.tif>``.
predictions_root = input_dir / "predictions_stardist_roi"

force_repredict = False

devices = -1
accelerator = "auto"
strategy = "ddp"
n_tiles = (1, 8, 8)
predict_batch_size = 8

iou_threshs = (0.3, 0.5, 0.7)

primary_metric = "accuracy"
primary_iou = 0.5

script_dir = Path(__file__).resolve().parent
results_folder = script_dir / "stardist_roi_sweeps"
results_folder.mkdir(parents=True, exist_ok=True)
summary_csv = results_folder / "sweep_predict_summary.csv"


# %% ─── lazy frame loader (shared with compare_segmentations.py) ────
class _LazyFrames:
    """Sized indexable view over a TZYX TIFF — one frame per
    ``__getitem__`` via ``tifffile.series[0].asarray(key=t)`` so we
    don't hold the whole stack in memory."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._tf = TiffFile(self.path)
        series = self._tf.series[0]
        self.shape = tuple(series.shape)
        self.dtype = series.dtype
        self._series = series

    def __len__(self) -> int:
        return int(self.shape[0]) if len(self.shape) == 4 else 1

    def __getitem__(self, t: int) -> np.ndarray:
        if len(self.shape) == 4:
            return self._series.asarray(key=t).astype(np.int32, copy=False)
        if t != 0:
            raise IndexError(t)
        return self._series.asarray().astype(np.int32, copy=False)

    def close(self):
        self._tf.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# %% ─── helpers ─────────────────────────────────────────────────────
def _final_train_metrics(model_dir: Path) -> dict:
    csv_path = model_dir / "metrics.csv"
    if not csv_path.is_file():
        return {"val_loss_final": None, "train_loss_final": None, "epochs_done": None}

    df = pd.read_csv(csv_path)
    out = {"val_loss_final": None, "train_loss_final": None, "epochs_done": None}
    if "val_loss" in df.columns:
        last = df["val_loss"].dropna()
        out["val_loss_final"] = float(last.iloc[-1]) if len(last) else None
    if "train_loss" in df.columns:
        last = df["train_loss"].dropna()
        out["train_loss_final"] = float(last.iloc[-1]) if len(last) else None
    if "epoch" in df.columns:
        last = df["epoch"].dropna()
        out["epochs_done"] = int(last.iloc[-1]) if len(last) else None
    return out


def _parse_sweep_tags(model_dir: Path) -> dict:
    """Parse optimizer / lr / scheduler out of the StarDist sweep dir
    name (``stardist_sweep_<opt>_lr<tag>_<sched>``). Falls back to
    ``training_config.json`` when the dir name doesn't follow it."""
    name = model_dir.name
    parts = name.split("_")
    optimizer = lr = scheduler = None
    for i, p in enumerate(parts):
        if p in ("adam", "sgd", "lars", "adamw", "rmsprop", "adamw_clip"):
            optimizer = p
        elif p.startswith("lr"):
            lr = p[2:].replace("p", ".")
        elif p in ("cosine", "noscheduler", "none", "warm_cosine"):
            scheduler = p

    cfg_path = model_dir / "training_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text()).get("parameters", {})
        optimizer = optimizer or cfg.get("optimizer")
        lr = lr or cfg.get("learning_rate")
        scheduler = scheduler or cfg.get("scheduler") or "none"
    return {
        "experiment": name,
        "optimizer": optimizer,
        "learning_rate": lr,
        "scheduler": scheduler,
    }


def _subset_timepoints(T: int, n_each: int = 5) -> list[int]:
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


def _score_against_keras(
    pred_path: Path,
    keras_path: Path,
    keras_indices: list[int] | None = None,
) -> dict:
    with _LazyFrames(pred_path) as pred, _LazyFrames(keras_path) as ref:
        if keras_indices is not None:
            if len(pred) != len(keras_indices):
                return {
                    "matching_error": (
                        f"pred frames ({len(pred)}) != requested "
                        f"keras indices ({len(keras_indices)})"
                    )
                }
            if max(keras_indices) >= len(ref):
                return {
                    "matching_error": (
                        f"keras stack has {len(ref)} frames but indices "
                        f"go up to {max(keras_indices)}"
                    )
                }
            pred_frames = [pred[t] for t in range(len(pred))]
            ref_frames = [ref[t] for t in keras_indices]
            if pred_frames[0].shape != ref_frames[0].shape:
                return {
                    "matching_error": (
                        f"frame shape mismatch: pred={pred_frames[0].shape} "
                        f"ref={ref_frames[0].shape}"
                    )
                }
            stats = matching_dataset(
                ref_frames,
                pred_frames,
                thresh=iou_threshs,
                show_progress=_INTERACTIVE,
            )
        else:
            if len(pred) != len(ref) or pred.shape != ref.shape:
                return {
                    "matching_error": (
                        f"shape mismatch: pred={pred.shape} ref={ref.shape}"
                    )
                }
            stats = matching_dataset(
                ref,
                pred,
                thresh=iou_threshs,
                show_progress=_INTERACTIVE,
            )
    flat = {}
    for thr, m in zip(iou_threshs, stats):
        for k in (
            "precision",
            "recall",
            "f1",
            "accuracy",
            "panoptic_quality",
            "mean_matched_score",
            "mean_true_score",
        ):
            flat[f"{k}@iou{thr:.2f}"] = float(getattr(m, k))
        flat[f"tp@iou{thr:.2f}"] = int(m.tp)
        flat[f"fp@iou{thr:.2f}"] = int(m.fp)
        flat[f"fn@iou{thr:.2f}"] = int(m.fn)
    return flat


# %% ─── load the single Mask-UNet ROI model once ────────────────────
# It's reused across every StarDist sweep model — its ROI mask depends
# only on the input frame, not on which StarDist is downstream. Loading
# it once also keeps GPU memory pressure constant; only the StarDist
# half is reallocated each iteration.
print(f"Loading ROI Mask-UNet from {mask_unet_dir}")
mask_unet = MaskUNetSegmenter.from_folder(mask_unet_dir, batch_size=predict_batch_size)


# %% ─── walk the sweep ──────────────────────────────────────────────
all_model_dirs = sorted(p for p in sweep_root.iterdir() if p.is_dir())
model_dirs = [
    p for p in all_model_dirs if not any(tag in p.name for tag in skip_name_substrings)
]
skipped = [p.name for p in all_model_dirs if p not in model_dirs]
input_files = sorted(input_dir.glob(input_pattern))
print(f"Sweep root:    {sweep_root}")
print(
    f"Models found:  {len(all_model_dirs)} (kept {len(model_dirs)}, "
    f"skipped {len(skipped)} matching {skip_name_substrings})"
)
for name in skipped:
    print(f"   skip → {name}")
print(f"Inputs:        {len(input_files)} TIFFs from {input_dir}")
print(f"Keras refs:    {keras_dir}")
print(f"Timepoints:    first/mid/last {subset_n_each} per timelapse")
print(f"ROI gate:      {mask_unet_dir.name} (single Mask-UNet, reused)")
print(
    f"Ranking by:    {primary_metric}@iou={primary_iou} (prediction) + "
    f"val_loss_final (training)"
)
print()


# %% ─── per-model: predict + score ──────────────────────────────────
results = []
for i, model_dir in enumerate(model_dirs):
    tags = _parse_sweep_tags(model_dir)
    t_model = time.perf_counter()
    print(
        f"\n[{i + 1}/{len(model_dirs)}] {tags['experiment']}  "
        f"(opt={tags['optimizer']} lr={tags['learning_rate']} "
        f"sched={tags['scheduler']})"
    )
    pred_dir = predictions_root / model_dir.name
    pred_dir.mkdir(parents=True, exist_ok=True)
    if force_repredict:
        for stale in pred_dir.iterdir():
            stale.unlink()
        print(f"   force_repredict=True — cleared {pred_dir}")

    try:
        star = StarDistSegmenter.from_folder(model_dir, batch_size=predict_batch_size)
    except FileNotFoundError as e:
        print(f"   ✗ no checkpoint: {e}")
        continue

    overrides = read_thresholds(model_dir)
    prob_thresh = overrides.get("prob_thresh", 0.5)
    nms_thresh = overrides.get("nms_thresh", 0.3)
    print(
        f"   loaded  prob_thresh={prob_thresh}  nms_thresh={nms_thresh}  "
        f"n_tiles={n_tiles}  (ROI-gated by Mask-UNet)"
    )

    # The ROIPipeline owns the Mask-UNet (frozen across the sweep) and
    # the current StarDist. ``predict_timelapse`` calls
    # ``pipeline.predict(frame)`` per timepoint — ROIPipeline runs the
    # Mask-UNet for the bbox first, crops, dispatches to StarDist on the
    # crop, then pastes labels back into the full-shape result. This is
    # exactly the keras VollSeg2D flow.
    pipeline = ROIPipeline(roi_unet=mask_unet, downstream=star)

    per_file_scores = []
    for j, f in enumerate(input_files):
        out_path = pred_dir / f.name
        indices_path = pred_dir / f"{f.stem}.keras_indices.json"
        keras_indices: list[int] | None = None
        if not out_path.is_file():
            t_pred = time.perf_counter()
            print(
                f"   [{j + 1}/{len(input_files)}] reading {f.name}",
            )
            vol = imread(f)
            print(
                f"   [{j + 1}/{len(input_files)}] full shape={tuple(vol.shape)} "
                f"dtype={vol.dtype}",
            )
            if vol.ndim == 4:
                keras_indices = _subset_timepoints(vol.shape[0], subset_n_each)
                vol = vol[keras_indices]
                print(
                    f"   [{j + 1}/{len(input_files)}] subset T-indices "
                    f"{keras_indices} → predicting shape={tuple(vol.shape)}",
                )
                out = predict_timelapse(
                    pipeline,
                    vol,
                    devices=devices,
                    accelerator=accelerator,
                    strategy=strategy,
                    enable_progress_bar=_INTERACTIVE,
                    bar_desc=f"[{i + 1}/{len(model_dirs)}] {tags['experiment']}",
                    prob_thresh=prob_thresh,
                    nms_thresh=nms_thresh,
                    n_tiles=n_tiles,
                )
                if not out:  # non-zero DDP rank — skip the rank-0 work
                    continue
                labels_tzyx = np.stack(
                    [out["labels"][t] for t in range(out["labels"].shape[0])],
                    axis=0,
                )
                imwrite(out_path, np.ascontiguousarray(labels_tzyx, dtype=np.uint16))
                indices_path.write_text(json.dumps(keras_indices))
            else:
                result = pipeline.predict(
                    vol,
                    prob_thresh=prob_thresh,
                    nms_thresh=nms_thresh,
                    n_tiles=n_tiles,
                )
                imwrite(out_path, np.ascontiguousarray(result.labels, dtype=np.uint16))
            print(
                f"   [{j + 1}/{len(input_files)}] wrote {out_path.name} "
                f"in {_human(time.perf_counter() - t_pred)}"
            )
        else:
            if indices_path.is_file():
                keras_indices = json.loads(indices_path.read_text())
                source = "sidecar"
            else:
                with TiffFile(f) as tf:
                    shp = tuple(tf.series[0].shape)
                if len(shp) == 4:
                    keras_indices = _subset_timepoints(shp[0], subset_n_each)
                source = "recomputed (no sidecar)"
            with TiffFile(out_path) as tf:
                pred_T = tf.series[0].shape[0] if len(tf.series[0].shape) == 4 else 1
            if keras_indices is not None and pred_T != len(keras_indices):
                print(
                    f"   [{j + 1}/{len(input_files)}] {out_path.name} cached with "
                    f"{pred_T} frames but index list has {len(keras_indices)} entries "
                    f"({source}) — DELETING cached prediction and re-predicting"
                )
                out_path.unlink()
                indices_path.unlink(missing_ok=True)
                continue
            print(
                f"   [{j + 1}/{len(input_files)}] {out_path.name} already "
                f"exists — skipping predict (T-indices {keras_indices}, "
                f"from {source})"
            )

        keras_path = keras_dir / f.name
        if not keras_path.is_file():
            print(f"   keras ref missing for {f.name} — skipping score")
            continue
        t_score = time.perf_counter()
        print(f"   [{j + 1}/{len(input_files)}] scoring vs keras ref…")
        per_file_scores.append(
            _score_against_keras(out_path, keras_path, keras_indices=keras_indices)
        )
        print(
            f"   [{j + 1}/{len(input_files)}] scored in "
            f"{_human(time.perf_counter() - t_score)}"
        )

    # Free the StarDist half of the pipeline before loading the next
    # model. ``mask_unet`` is preserved across iterations — it's the
    # fixed ROI gate, not part of the sweep.
    del star, pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg = {}
    if per_file_scores:
        all_keys = set().union(*per_file_scores)
        for k in all_keys:
            vals = [
                s.get(k) for s in per_file_scores if isinstance(s.get(k), (int, float))
            ]
            if vals:
                avg[k] = float(np.mean(vals))

    row = {
        **tags,
        **_final_train_metrics(model_dir),
        **avg,
        "n_inputs_scored": len(per_file_scores),
    }
    results.append(row)

    headline = avg.get(f"{primary_metric}@iou{primary_iou:.2f}", float("nan"))
    print(
        f"   {primary_metric}@iou{primary_iou:.2f}: {headline:.4f}  "
        f"val_loss_final: {row.get('val_loss_final')!s:>8}  "
        f"model_total: {_human(time.perf_counter() - t_model)}"
    )


# %% ─── write summary + rank (rank 0 only) ─────────────────────────
if _is_rank_zero():
    if not results:
        print("No models scored. Check sweep_root / input_dir / keras_dir.")
    else:
        sorted_keys = list(results[0].keys())
        for r in results[1:]:
            for k in r:
                if k not in sorted_keys:
                    sorted_keys.append(k)

        with summary_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted_keys)
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"Summary CSV: {summary_csv}")
        print()

        key = f"{primary_metric}@iou{primary_iou:.2f}"
        by_pred = sorted(
            [r for r in results if r.get(key) is not None],
            key=lambda r: r[key],
            reverse=True,
        )
        by_train = sorted(
            [r for r in results if r.get("val_loss_final") is not None],
            key=lambda r: r["val_loss_final"],
        )

        print(f"── BEST BY PREDICTION QUALITY (top 5, {key}) ──")
        for r in by_pred[:5]:
            print(
                f"  {r[key]:.4f}  {r['experiment']:<60}  "
                f"opt={r['optimizer']!s:<6}  lr={r['learning_rate']!s:<10}  "
                f"sched={r['scheduler']!s:<12}  val_loss_final={r['val_loss_final']!s}"
            )

        print()
        print("── BEST BY TRAINING (top 5, val_loss_final ↓) ──")
        for r in by_train[:5]:
            print(
                f"  {r['val_loss_final']:.6f}  {r['experiment']:<60}  "
                f"opt={r['optimizer']!s:<6}  lr={r['learning_rate']!s:<10}  "
                f"sched={r['scheduler']!s:<12}  {key}={r.get(key)!s}"
            )


# %%
