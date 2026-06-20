"""Sweep prediction + analysis for StarDist.

For every trained model folder under ``sweep_root/`` (one per
optimizer × LR × scheduler combo produced by
``slurm_sweep_stardist_jeanzay.sh``) that survives the
``skip_name_substrings`` filter:

1. Load the model via :meth:`StarDistSegmenter.from_folder` — reads the
   ``training_config.json`` + ``rays.npy`` sidecars automatically.
2. Run tiled / DDP-sharded prediction on every TIFF in ``input_dir``,
   but only on the **first ``subset_n_each``, middle ``subset_n_each``
   and last ``subset_n_each`` timepoints** of each 4D timelapse — i.e.
   15 frames total at the default. Cuts a full-sweep prediction from
   N×T frames down to N×15 so 14 sweep runs can be ranked by IoU in
   tractable time.
3. Compare each prediction to the matching keras reference TIFF under
   ``keras_dir`` at exactly those original timepoints (the keras stack
   is full-T, so we index into it).
4. Read the final ``val_loss`` / ``train_loss`` off ``metrics.csv``.
5. Aggregate everything into
   ``stardist_sweeps/sweep_predict_summary.csv`` next to the
   training-time ``sweep_summary.csv`` and print **best by prediction
   accuracy** + **best by training val_loss**.

Edit the paths block at the top, then ``python sweep_predict_and_analyze.py``.
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

from kapoorlabs_vollseg import StarDistSegmenter, predict_timelapse
from kapoorlabs_vollseg._backbones._config import read_thresholds
from kapoorlabs_vollseg.eval import matching_dataset


torch.set_float32_matmul_precision("high")
# Same opt-in gate as kapoorlabs_vollseg.stardist.inference uses. We
# tried letting ``sys.stderr.isatty()`` decide but it false-positives
# on SSH / ``screen`` / ``tee`` / SLURM-with-pty setups where ``\r``
# doesn't fully erase the prior line — tqdm then writes a fresh line
# per update and the nested per-tile / per-NMS-peak / per-cell bars
# from one frame spam hundreds of lines. The phase-done prints in
# inference.py give the same timing info without the spam. Set
# ``KAPOORLABS_VOLLSEG_PROGRESS=1`` to opt the bars back in.
_INTERACTIVE = os.environ.get("KAPOORLABS_VOLLSEG_PROGRESS") == "1"


def _is_rank_zero() -> bool:
    """Whether we're the rank-0 worker (single-GPU default returns True).

    Under DDP every rank runs this script but only rank 0 has the gathered
    per-frame results from ``predict_timelapse``; rank > 0 returns ``{}``.
    So all the analysis / CSV-write / printout work has to be gated on
    rank 0 to avoid the workers fighting over ``sweep_summary.csv`` or
    appending bogus all-empty rows."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    # Lightning sets ``LOCAL_RANK`` / ``RANK`` before the process group
    # is initialised; honour those so the early prints (before the first
    # ``predict_timelapse`` call) are also rank-0-gated.
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0


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

# Skip these runs (chosen after sweep_stardist_analyze.py): the lr=10
# tag (``lr1p0ep1``) blows up early and SGD never catches the leaders.
skip_name_substrings = ("lr1p0ep1", "_sgd_")

# Subset of timepoints to predict / score on, instead of the whole
# timelapse. We take the first ``n_each``, a centred middle ``n_each``,
# and the last ``n_each`` frames; volumes shorter than ~3·n_each fall
# back to the full T-axis. Reduces a sweep from N×T frames down to
# N×min(15, T) so we can rank 14 models by IoU in tractable time.
subset_n_each = 5

# Where the per-model prediction TIFFs land. Lives next to the input
# data (not inside each sweep folder) so all model predictions sit
# together, one folder per model, easy to browse alongside the input.
# Layout::
#
#     <input_dir>/
#     ├── timelapse_fifth_dataset.tif                                    ← input
#     └── predictions/
#         ├── stardist_sweep_adam_lr1p0e-2_noscheduler/
#         │   ├── timelapse_fifth_dataset.tif                            ← prediction
#         │   └── timelapse_fifth_dataset.keras_indices.json             ← sidecar
#         ├── stardist_sweep_lars_lr1p0ep0_noscheduler/
#         │   └── …
#         └── …
predictions_root = input_dir / "predictions"

# Wipe every per-model prediction TIFF + sidecar before predicting, so
# a sweep that uses an updated model arch / threshold / config doesn't
# silently re-use stale cached predictions. Set ``False`` to honour
# existing caches.
force_repredict = True

# Multi-GPU sweep prediction knobs. ``predict_timelapse`` shards the
# T axis across DDP ranks via Lightning's ``DistributedSampler`` — each
# GPU walks a disjoint subset of timepoints (by frame index) and the
# gather happens transparently inside ``predict_timelapse`` before the
# rank-0 worker writes the prediction TIFF. To actually use the GPUs
# you have to launch the script under ``srun`` so SLURM spawns the
# right number of tasks; on a single 4×A100 node the recommended
# launch is::
#
#     srun --ntasks-per-node=4 --gres=gpu:4 python sweep_predict_and_analyze.py
#
# Setting ``devices=-1`` lets Lightning pick up whatever GPUs SLURM /
# CUDA_VISIBLE_DEVICES exposed. Set ``strategy="ddp"`` when devices > 1.
devices = -1  # -1 = all visible GPUs; 1 for single-GPU runs.
accelerator = "auto"
strategy = "ddp"  # "auto" works for single-GPU; explicit DDP for ≥2.
n_tiles = (1, 4, 4)
# Per-tile batch size inside ``predict_volume``. Default of 4 underuses
# a V100; bump to 16 for ~3× wall-clock gain when VRAM allows. Drop back
# to 4 if you OOM mid-frame.
predict_batch_size = 8

# IoU thresholds at which prediction quality is scored.
iou_threshs = (0.3, 0.5, 0.7)

# Primary metric for ranking. Any field that `matching_dataset` returns:
# precision / recall / accuracy / f1 / panoptic_quality /
# mean_matched_score / mean_true_score
primary_metric = "accuracy"
primary_iou = 0.5

# Where to write the summary CSV. Sibling of sweep_stardist_analyze.py's
# stardist_sweeps/ folder under the script dir, so the training-time
# summary and the prediction-quality summary sit next to each other.
script_dir = Path(__file__).resolve().parent
results_folder = script_dir / "stardist_sweeps"
results_folder.mkdir(parents=True, exist_ok=True)
summary_csv = results_folder / "sweep_predict_summary.csv"


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
    # Lightning's CSVLogger writes ``train_loss`` per-batch as
    # ``train_loss_step`` and per-epoch as ``train_loss_epoch`` (because
    # ``log_metrics`` is called with both ``on_step=True`` and
    # ``on_epoch=True`` in our base module). Prefer ``*_epoch``, fall
    # back to bare name, then ``*_step``.
    for tag, key in (
        ("val_loss_final", "val_loss"),
        ("train_loss_final", "train_loss"),
    ):
        for col in (f"{key}_epoch", key, f"{key}_step"):
            if col in df.columns:
                last = df[col].dropna()
                if len(last):
                    out[tag] = float(last.iloc[-1])
                    break
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


def _subset_timepoints(T: int, n_each: int = 5) -> list[int]:
    """Return the indices of the first ``n_each``, middle ``n_each`` and
    last ``n_each`` timepoints of a T-frame timelapse, deduplicated and
    sorted. For ``T <= 3·n_each`` we just return ``range(T)``."""
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


def _save_subset_companions(
    input_path: Path,
    keras_path: Path,
    keras_indices: list[int] | None,
    out_dir: Path,
) -> None:
    """Write subset-T copies of the raw input and the keras reference
    next to the StarDist predictions so the three stacks (raw / keras /
    StarDist) can be loaded side by side in napari.

    Idempotent — both files are written once (skip if present). Both
    are streamed frame by frame so a 192-frame timelapse doesn't pull
    a full 70 GB into memory.

    Layout::

        predictions_root/
          <stem>.raw.tif      ← raw input cropped to keras_indices
          <stem>.keras.tif    ← keras reference labels cropped to keras_indices
          <model_name>/
            <stem>.tif        ← StarDist labels (this model's prediction)
    """
    if keras_indices is None:
        return  # 3D / single-frame input; nothing to subset.

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path.name).stem
    raw_out = out_dir / f"{stem}.raw.tif"
    keras_out = out_dir / f"{stem}.keras.tif"

    # Load + slice via ``imread`` — same access pattern the predict
    # loop already uses (``vol = imread(f); vol = vol[keras_indices]``
    # in the fresh-predict branch). Writing the subset back as a
    # single (T, Z, Y, X) array gives napari/tifffile a real 4D stack
    # rather than N separate 3D slabs.
    if not raw_out.is_file():
        vol = imread(input_path)
        if vol.ndim != 4:
            return
        subset = np.ascontiguousarray(vol[keras_indices])
        imwrite(raw_out, subset, bigtiff=True)
        print(f"   companion → {raw_out.name}  (T={len(keras_indices)})")

    if keras_path.is_file() and not keras_out.is_file():
        ref = imread(keras_path)
        if ref.ndim != 4:
            # Single-frame keras ref — nothing to subset, copy as-is.
            imwrite(keras_out, np.ascontiguousarray(ref), bigtiff=True)
        else:
            subset = np.ascontiguousarray(ref[keras_indices])
            imwrite(keras_out, subset, bigtiff=True)
        print(f"   companion → {keras_out.name}  (T={len(keras_indices)})")


def _score_against_keras(
    pred_path: Path,
    keras_path: Path,
    keras_indices: list[int] | None = None,
) -> dict:
    """Stream both stacks frame by frame through ``matching_dataset``.

    When ``keras_indices`` is given, the prediction TIFF is assumed to
    hold one frame per index (in the order of ``keras_indices``) and
    the keras reference is read at exactly those timepoints. Without
    indices we fall back to the full-stack comparison.

    Returns a flat dict keyed ``<metric>@iou<thr>``.
    """
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
    # Separate dir from full-volume predictions so an existing
    # ``predictions/<name>.tif`` from an earlier run can't shadow the
    # Predictions live next to the input data, under
    # ``<input_dir>/predictions/<model_name>/<input.tif>`` — see
    # ``predictions_root`` at the top of the script. One folder per
    # model so they're easy to browse / clean.
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
        f"n_tiles={n_tiles}"
    )

    # ── predict each input ──
    per_file_scores = []
    for j, f in enumerate(input_files):
        out_path = pred_dir / f.name
        # Sidecar JSON next to the prediction TIFF records the exact
        # T-indices that went into the subset. On a fresh run we write
        # it after the predict; on a resume we read it back so a
        # changed ``subset_n_each`` can't silently misalign frames.
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
                # Every DDP rank participates in ``predict_timelapse`` —
                # Lightning's DistributedSampler hands each rank a
                # disjoint subset of timepoints (by index) and the
                # function gathers all per-frame outputs to rank 0
                # before returning. Non-zero ranks receive ``{}`` so
                # the post-predict scoring / TIFF-write naturally
                # short-circuits on them.
                out = predict_timelapse(
                    star,
                    vol,
                    devices=devices,
                    accelerator=accelerator,
                    strategy=strategy,
                    # Lightning's rich progress bar suffers the same
                    # newline-spam issue under SLURM as our tqdm bars;
                    # gate on the env var. Our own ``bar_desc`` bar
                    # below is the single per-model progress line.
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
                # ``nms_to_labels`` returns uint16 natively; preserve
                # that on disk (16-bit = half the size of uint32, and
                # tifffile can occasionally surface a 32-bit label TIFF
                # as float-like depending on the viewer reading it).
                imwrite(out_path, np.ascontiguousarray(labels_tzyx, dtype=np.uint16))
                # Persist the exact T-indices used so a re-run picks
                # up the same alignment even if ``subset_n_each``
                # was changed in between.
                indices_path.write_text(json.dumps(keras_indices))
            else:
                result = star.predict(
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
            # Recover the index list. Prefer the sidecar JSON if it
            # exists (it was written at predict time, so it's the
            # authoritative record of which T-indices the cached
            # prediction TIFF actually holds). Fall back to recomputing
            # from the input TIFF only when the sidecar is missing
            # (legacy cached runs from before the sidecar was written).
            if indices_path.is_file():
                keras_indices = json.loads(indices_path.read_text())
                source = "sidecar"
            else:
                with TiffFile(f) as tf:
                    shp = tuple(tf.series[0].shape)
                if len(shp) == 4:
                    keras_indices = _subset_timepoints(shp[0], subset_n_each)
                source = "recomputed (no sidecar)"
            # Belt-and-braces: cross-check the cached TIFF's T length
            # against the index list. A mismatch means a stale cache
            # from a different ``subset_n_each`` — skip it to avoid
            # scoring nonsense.
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
                # back up and re-run this iteration
                continue
            print(
                f"   [{j + 1}/{len(input_files)}] {out_path.name} already "
                f"exists — skipping predict (T-indices {keras_indices}, "
                f"from {source})"
            )

        # ── score against keras ref ──
        keras_path = keras_dir / f.name
        # Save subset-T raw + keras companions at predictions_root once
        # (idempotent) so the three stacks — raw input / keras labels /
        # this model's StarDist labels — can be loaded side by side in
        # napari at the exact same T-indices.
        _save_subset_companions(f, keras_path, keras_indices, predictions_root)
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


# %% ─── write summary + rank (rank 0 only) ─────────────────────────
# Workers (rank > 0) have empty ``per_file_scores`` so their ``results``
# list is all-empty rows that would clobber the CSV the rank-0 worker
# is about to write. Gate everything from here on on rank 0 explicitly.
if _is_rank_zero():
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
