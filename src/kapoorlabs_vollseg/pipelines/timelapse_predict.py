"""Distributed timelapse prediction via :class:`lightning.Trainer.predict`.

Wraps any :class:`Pipeline` (singleton or composite — CARE, U-Net,
Mask-UNet, StarDist, ``UNetStarDistPipeline``, ``ROIPipeline``, the
factory output of :class:`VollSeg.from_models`) so its
``.predict(frame)`` is dispatched per-timepoint by a Lightning
``Trainer.predict(...)``. With ``devices > 1`` + ``strategy='ddp'``
Lightning's ``DistributedSampler`` hands each rank a disjoint slice of
the T axis; each GPU walks its assigned timepoints then picks up the
next one as soon as it's free — exactly the work-stealing pattern the
user asked for.

Results are gathered + sorted by their T index + stacked into one
``(T, …)`` array per Result field (``labels`` / ``semantic`` / ``roi``
/ ``denoised`` / ``probability`` — whichever the pipeline actually
populates).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset

from .base import Pipeline


_FIELDS = ("labels", "semantic", "roi", "denoised", "probability")


class _FrameDataset(Dataset):
    """Yield one timepoint per ``__getitem__`` from a TZYX / TYX volume."""

    def __init__(self, volume: np.ndarray):
        if volume.ndim < 3:
            raise ValueError(
                f"_FrameDataset needs at least a (T, *spatial) volume, "
                f"got ndim={volume.ndim}"
            )
        self.volume = volume

    def __len__(self) -> int:
        return int(self.volume.shape[0])

    def __getitem__(self, idx: int):
        return self.volume[idx], idx


def _identity_collate(batch):
    """``batch_size=1`` — pass the single ``(frame, idx)`` through unchanged."""
    return batch[0]


class TimelapsePredictor(LightningModule):
    """Lightning shell whose ``predict_step`` calls a pipeline per frame.

    Adds a Lightning-style ``predict_step`` to every pipeline in the
    codebase by composition — singletons and composites become DDP-
    distributable at the timepoint level without any per-class changes.

    Progress is reported by Lightning's built-in progress bar only
    (gate via ``Trainer(enable_progress_bar=...)``). The phase prints
    inside ``inference.py`` are silenced by default; opt them back in
    with ``KAPOORLABS_VOLLSEG_PROGRESS=1``.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        predict_kwargs: Optional[dict] = None,
        total_frames: Optional[int] = None,
        bar_desc: str = "frames",
        frame_writer: Optional[Callable[[int, dict], None]] = None,
        spill_dir: Optional[os.PathLike] = None,
    ):
        super().__init__()
        # Stored as an attribute, NOT a submodule — pipeline isn't an nn.Module.
        self.pipeline = pipeline
        self.predict_kwargs = dict(predict_kwargs or {})
        # ``total_frames`` / ``bar_desc`` are kept on the signature for
        # backward compatibility but are no longer consulted — there's
        # no per-step tqdm bar anymore. Lightning's own progress bar is
        # the only one shown.
        self._total_frames = total_frames
        self._bar_desc = bar_desc
        self._frame_writer = frame_writer
        # When neither a streaming ``frame_writer`` nor a spill dir is
        # set, ``predict_step`` keeps frame arrays in the returned dict
        # (legacy behaviour). With a spill dir, each frame is dumped to
        # ``<spill_dir>/<t:06d>.npz`` and the returned dict carries
        # only the timepoint index — the offload-and-stitch path
        # described in :func:`predict_timelapse`.
        self._spill_dir = Path(spill_dir) if spill_dir is not None else None

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        frame, t_idx = batch
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        result = self.pipeline.predict(frame, **self.predict_kwargs)

        # Streaming sink: the caller wrote the frame to disk via
        # ``frame_writer``. Drop every array from the dict we return so
        # Lightning's per-batch buffer + the downstream gather/stack
        # don't hold onto ~250 MB × T worth of float32 — that's what
        # OOM-killed the 192-frame timelapse before this path existed.
        if self._frame_writer is not None:
            self._frame_writer(
                int(t_idx),
                {
                    "labels": result.labels,
                    "semantic": result.semantic,
                    "roi": result.roi,
                    "denoised": result.denoised,
                    "probability": result.probability,
                },
            )
            return {"t": int(t_idx)}

        # Offload-and-stitch path: spill this frame's arrays to disk so
        # nothing timelapse-sized accumulates in Python's per-batch
        # buffer. ``predict_timelapse`` stitches the per-T spill files
        # back into stacked arrays on rank 0 after Lightning's predict
        # loop returns.
        if self._spill_dir is not None:
            payload: dict[str, np.ndarray] = {}
            if result.labels is not None:
                payload["labels"] = np.asarray(result.labels)
            if result.semantic is not None:
                payload["semantic"] = np.asarray(result.semantic)
            if result.roi is not None:
                payload["roi"] = np.asarray(result.roi)
            if result.denoised is not None:
                payload["denoised"] = np.asarray(result.denoised)
            if result.probability is not None:
                payload["probability"] = np.asarray(result.probability)
            np.savez(self._spill_dir / f"{int(t_idx):06d}.npz", **payload)
            return {"t": int(t_idx)}

        return {
            "t": int(t_idx),
            "labels": result.labels,
            "semantic": result.semantic,
            "roi": result.roi,
            "denoised": result.denoised,
            "probability": result.probability,
        }


def predict_timelapse(
    pipeline: Pipeline,
    volume: np.ndarray,
    *,
    devices: Any = 1,
    accelerator: str = "auto",
    strategy: str = "auto",
    enable_progress_bar: bool = True,
    num_workers=0,
    bar_desc: str = "frames",
    frame_writer: Optional[Callable[[int, dict], None]] = None,
    offload_to_disk: bool = True,
    spill_dir: Optional[os.PathLike] = None,
    **predict_kwargs,
) -> dict[str, Optional[np.ndarray]]:
    """Run ``pipeline.predict(...)`` over every timepoint of ``volume``.

    Parameters
    ----------
    pipeline
        Any :class:`Pipeline` — a singleton, a Layer-2 composite, or the
        output of :func:`VollSeg.from_models`.
    volume
        ``(T, *spatial)`` numpy array. T is the axis sharded across GPUs.
    devices
        Lightning ``devices`` value. ``1`` (single GPU) / ``-1`` (all
        visible GPUs) / ``N`` (first N) / list of GPU ids.
    accelerator, strategy
        Forwarded to :class:`lightning.Trainer`. Set ``strategy='ddp'``
        for multi-GPU T-sharding.
    enable_progress_bar
        Lightning's per-batch bar — set False if you have an outer tqdm.
    frame_writer
        Optional ``(t_idx, result_dict) -> None`` callback invoked from
        inside ``predict_step`` immediately after the pipeline returns.
        When set, each frame's arrays are handed to the writer and
        dropped from the per-batch return value, so nothing
        timelapse-sized accumulates in Python. The caller's writer is
        responsible for streaming to disk (typically a single
        ``TiffWriter`` opened around the call). The returned dict will
        be empty — disk is the source of truth. Use this when you want
        the prediction to flow straight to a final on-disk format
        without ever materializing the stacked timelapse in RAM.
    offload_to_disk
        Default ``True``. When set (and ``frame_writer`` is ``None``),
        each frame's arrays are spilled to ``spill_dir / <t:06d>.npz``
        inside ``predict_step`` and the per-batch dict carries only
        ``{"t": idx}``. After Lightning's predict loop returns, the
        per-T spill files are stitched into ``np.stack``-style output
        arrays. Memory peak drops from ~3× stack size (Python list +
        ``np.stack`` copy + result) to ~1× stack size (only the
        stitched output is held at once). Caller still gets the same
        ``{"labels": (T, …), "denoised": (T, …)}`` shape it always
        did. Set ``False`` to restore the legacy in-memory path —
        useful for tiny T where the disk round-trip costs more than it
        saves.
    spill_dir
        Optional directory used for the spill files. ``None`` (default)
        creates a fresh ``TemporaryDirectory`` that's wiped at the end
        of the call. Pass an explicit path if you want to inspect the
        per-T arrays after the fact (e.g. for debugging a flaky
        downstream stitch).
    **predict_kwargs
        Forwarded to every ``pipeline.predict(frame, **kwargs)`` call —
        e.g. ``prob_thresh`` / ``nms_thresh`` / ``n_tiles``.

    Returns
    -------
    dict
        Without ``frame_writer``: the fields the pipeline populates,
        each stacked along T (``labels`` / ``semantic`` / ``roi`` /
        ``denoised`` / ``probability``). Missing fields are absent.

        With ``frame_writer``: empty dict — the writer was the only
        consumer of the per-frame arrays.
    """
    # Resolve spill directory: explicit path > tempdir > no offload.
    spill_ctx: Optional[tempfile.TemporaryDirectory] = None
    spill_path: Optional[Path] = None
    use_offload = offload_to_disk and frame_writer is None
    if use_offload:
        if spill_dir is None:
            spill_ctx = tempfile.TemporaryDirectory(prefix="vollseg_spill_")
            spill_path = Path(spill_ctx.name)
        else:
            spill_path = Path(spill_dir)
            spill_path.mkdir(parents=True, exist_ok=True)

    predictor = TimelapsePredictor(
        pipeline,
        predict_kwargs,
        total_frames=int(volume.shape[0]),
        bar_desc=bar_desc,
        frame_writer=frame_writer,
        spill_dir=spill_path,
    )
    loader = DataLoader(
        _FrameDataset(volume),
        batch_size=1,
        shuffle=False,
        collate_fn=_identity_collate,
        num_workers=num_workers,
    )
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=enable_progress_bar,
    )
    per_frame = trainer.predict(predictor, loader)

    # Other ranks in DDP get None back — let the caller short-circuit.
    if per_frame is None:
        return {}

    # Streaming sink path: ``predict_step`` already handed every frame's
    # arrays to ``frame_writer``; the returned dicts hold only ``{"t":
    # idx}``. Skip the gather / stack — there's nothing to stack and the
    # caller is already done with disk-side writes.
    if frame_writer is not None:
        return {}

    # Lightning returns a list-of-batch-outputs; each batch is one frame
    # so each entry is a dict. Each rank only sees the slice the
    # DistributedSampler handed it, so under DDP we must gather across
    # ranks before stacking — otherwise rank 0 writes T/world_size
    # frames and the other ranks clobber that same file with their own
    # shard.
    per_frame = [d for d in per_frame if d is not None]

    # ``gather_object`` serialises the entire ``per_frame`` list into a
    # byte tensor and ships it through the **current CUDA device**, which
    # OOMs on any non-trivial timelapse (192 frames of (19, 1560, 1560)
    # ≈ 70 GiB of pickle bytes). With a single rank there's nothing to
    # gather, so skip it. With multiple ranks we still call it — the
    # multi-rank-OOM fix is a separate piece of work (streaming each
    # frame to a per-T scratch file in ``predict_step`` so the gather
    # only ships file paths).
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        gather_list = [None] * world_size if rank == 0 else None
        dist.gather_object(per_frame, gather_list, dst=0)
        if rank != 0:
            return {}
        per_frame = [d for sub in gather_list for d in (sub or [])]

    # DistributedSampler pads to be divisible by world_size by repeating
    # samples — dedupe by T before stacking so duplicates from padding
    # don't end up in the output.
    seen: set[int] = set()
    unique = []
    for d in per_frame:
        t = int(d["t"])
        if t in seen:
            continue
        seen.add(t)
        unique.append(d)
    per_frame = sorted(unique, key=lambda d: int(d["t"]))

    out: dict[str, Optional[np.ndarray]] = {}
    if use_offload:
        # Stitch: each ``per_frame`` entry holds only its T index; the
        # arrays live in ``<spill_path>/<t:06d>.npz``. Allocate one
        # output buffer per field (sized to T × frame-shape) and copy
        # each frame's array into the matching slot. Memory peak ≈ one
        # output buffer + one read frame, instead of the prior
        # ``list-of-T-arrays + np.stack`` pile.
        if not per_frame:
            if spill_ctx is not None:
                spill_ctx.cleanup()
            return out
        first_npz = np.load(spill_path / f"{int(per_frame[0]['t']):06d}.npz")
        field_specs: dict[str, tuple] = {
            field: (first_npz[field].shape, first_npz[field].dtype)
            for field in first_npz.files
        }
        first_npz.close()
        T = len(per_frame)
        for field, (frame_shape, frame_dtype) in field_specs.items():
            out[field] = np.empty((T, *frame_shape), dtype=frame_dtype)
        for i, d in enumerate(per_frame):
            t = int(d["t"])
            npz = np.load(spill_path / f"{t:06d}.npz")
            for field in field_specs:
                out[field][i] = npz[field]
            npz.close()
        if spill_ctx is not None:
            spill_ctx.cleanup()
        return out

    # Legacy in-memory path: frames carry their arrays in ``per_frame``
    # itself. Stacks twice the timelapse size (list + np.stack output)
    # at peak — only safe when T × frame_size fits 2× in RAM.
    for key in _FIELDS:
        chunks = [d[key] for d in per_frame if d.get(key) is not None]
        if chunks:
            out[key] = np.stack(chunks, axis=0)
    return out
