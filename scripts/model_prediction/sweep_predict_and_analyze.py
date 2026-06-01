"""Sweep prediction + analysis for StarDist.

For every trained model folder under ``sweep_root/`` (one per
optimizer × LR × scheduler combo produced by
``slurm_sweep_stardist_jeanzay.sh``):

1. Load the model via :meth:`StarDistSegmenter.from_folder` — reads the
   ``training_config.json`` + ``rays.npy`` sidecars automatically.
2. Run tiled / DDP-sharded prediction on every TIFF in ``input_dir``
   (timelapse-aware: 4D inputs are sharded across ``devices`` GPUs).
3. Compare each prediction to the matching keras reference TIFF under
   ``keras_dir``. Streams one timepoint at a time so peak memory is
   ~2 frames, not the whole stack.
4. Read the final ``val_loss`` / ``train_loss`` off ``metrics.csv``.
5. Aggregate everything into ``sweep_root/sweep_summary.csv`` and
   print the **best by prediction accuracy** + **best by training
   val_loss**.

Edit the paths block at the top, then ``python sweep_predict_and_analyze.py``.
"""

# %%
from __future__ import annotations

import csv
import functools
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tifffile import TiffFile, imread, imwrite

from kapoorlabs_vollseg import StarDistSegmenter, predict_timelapse
from kapoorlabs_vollseg._backbones._config import read_thresholds
from kapoorlabs_vollseg.eval import matching_dataset

# Force every status line to flush immediately so SLURM logs show
# where the sweep is without waiting for the process to exit. Belt-
# and-braces alongside ``PYTHONUNBUFFERED=1`` / ``python -u``.
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
input_dir = Path("/lustre/fsn1/projects/rech/jsy/uzj81mi/demo_data/")
input_pattern = "*.tif"

# Keras reference TIFFs to score against — must have the same basenames
# as the inputs (i.e. ``timelapse_fifth_dataset.tif`` here, etc.).
keras_dir = Path("/lustre/fsn1/projects/rech/jsy/uzj81mi/demo_data/keras_prediction/")

# Multi-GPU sweep prediction knobs — usually fine to leave at single-GPU
# since the analysis loop is sequential model-by-model.
devices = 1
accelerator = "auto"
strategy = "auto"
n_tiles = (1, 8, 8)
# Per-tile batch size inside ``predict_volume``. Default of 4 underuses
# a V100; bump to 16 for ~3× wall-clock gain when VRAM allows. Drop back
# to 4 if you OOM mid-frame.
predict_batch_size = 16

# IoU thresholds at which prediction quality is scored.
iou_threshs = (0.3, 0.5, 0.7)

# Primary metric for ranking. Any field that `matching_dataset` returns:
# precision / recall / accuracy / f1 / panoptic_quality /
# mean_matched_score / mean_true_score
primary_metric = "accuracy"
primary_iou = 0.5

# Where to write the summary CSV.
summary_csv = sweep_root / "sweep_summary.csv"


# %% ─── lazy frame loader (shared with compare_segmentations.py) ────
class _LazyFrames:
    """Sized indexable view over a TZYX TIFF.

    ``__getitem__(t)`` pulls only the t-th sub-volume into memory via
    ``tifffile.series[0].asarray(key=t)``. Lets the matching loop
    stream one keras/pytorch pair at a time instead of holding the
    entire 70 GB stack in memory.
    """

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
    """Return ``{val_loss, train_loss, epochs_done}`` from the last
    non-empty row of ``metrics.csv``. Missing → values of ``None``."""
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
    """Parse optimizer / lr / scheduler out of the model dir name (the
    sweep script writes ``stardist_sweep_<opt>_lr<tag>_<sched>``).
    Falls back to reading ``training_config.json`` when the dir name
    doesn't follow the sweep convention."""
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


def _score_against_keras(pred_path: Path, keras_path: Path) -> dict:
    """Stream both stacks frame by frame through ``matching_dataset``.
    Returns a flat dict keyed ``<metric>@iou<thr>``."""
    with _LazyFrames(pred_path) as pred, _LazyFrames(keras_path) as ref:
        if len(pred) != len(ref) or pred.shape != ref.shape:
            return {
                "matching_error": (f"shape mismatch: pred={pred.shape} ref={ref.shape}")
            }
        stats = matching_dataset(ref, pred, thresh=iou_threshs, show_progress=True)
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


# %% ─── walk the sweep ──────────────────────────────────────────────
model_dirs = sorted(p for p in sweep_root.iterdir() if p.is_dir())
input_files = sorted(input_dir.glob(input_pattern))
print(f"Sweep root:    {sweep_root}")
print(f"Models found:  {len(model_dirs)}")
print(f"Inputs:        {len(input_files)} TIFFs from {input_dir}")
print(f"Keras refs:    {keras_dir}")
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
    pred_dir = model_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

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
        f"n_tiles={n_tiles}"
    )

    # ── predict each input ──
    per_file_scores = []
    for j, f in enumerate(input_files):
        out_path = pred_dir / f.name
        if not out_path.is_file():
            t_pred = time.perf_counter()
            print(
                f"   [{j + 1}/{len(input_files)}] reading {f.name}",
            )
            vol = imread(f)
            print(
                f"   [{j + 1}/{len(input_files)}] predicting "
                f"shape={tuple(vol.shape)} dtype={vol.dtype}",
            )
            if vol.ndim == 4:
                out = predict_timelapse(
                    star,
                    vol,
                    devices=devices,
                    accelerator=accelerator,
                    strategy=strategy,
                    enable_progress_bar=True,
                    prob_thresh=prob_thresh,
                    nms_thresh=nms_thresh,
                    n_tiles=n_tiles,
                )
                if not out:  # non-zero DDP rank
                    continue
                labels_tzyx = np.stack(
                    [out["labels"][t] for t in range(out["labels"].shape[0])],
                    axis=0,
                )
                imwrite(out_path, labels_tzyx.astype(np.uint32))
            else:
                result = star.predict(
                    vol,
                    prob_thresh=prob_thresh,
                    nms_thresh=nms_thresh,
                    n_tiles=n_tiles,
                )
                imwrite(out_path, result.labels.astype(np.uint32))
            print(
                f"   [{j + 1}/{len(input_files)}] wrote {out_path.name} "
                f"in {_human(time.perf_counter() - t_pred)}"
            )
        else:
            print(
                f"   [{j + 1}/{len(input_files)}] {out_path.name} already "
                f"exists — skipping predict"
            )

        # ── score against keras ref ──
        keras_path = keras_dir / f.name
        if not keras_path.is_file():
            print(f"   keras ref missing for {f.name} — skipping score")
            continue
        t_score = time.perf_counter()
        print(f"   [{j + 1}/{len(input_files)}] scoring vs keras ref…")
        per_file_scores.append(_score_against_keras(out_path, keras_path))
        print(
            f"   [{j + 1}/{len(input_files)}] scored in "
            f"{_human(time.perf_counter() - t_score)}"
        )

    # Free the StarDist backbone from GPU before loading the next model.
    # Otherwise 18 model loads in the same process pile up on the V100
    # and we OOM mid-sweep rather than on tile-1 of model-1.
    del star
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # average scores across input files
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


# %% ─── write summary + rank ────────────────────────────────────────
if not results:
    print("No models scored. Check sweep_root / input_dir / keras_dir.")
else:
    # Stable column order — tags + training metrics + per-IoU metrics.
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
