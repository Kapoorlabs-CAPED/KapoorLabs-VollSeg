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

from typing import Any, Optional

import numpy as np
import torch
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset

from .base import Pipeline


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
    """

    def __init__(self, pipeline: Pipeline, predict_kwargs: Optional[dict] = None):
        super().__init__()
        # Stored as an attribute, NOT a submodule — pipeline isn't an nn.Module.
        self.pipeline = pipeline
        self.predict_kwargs = dict(predict_kwargs or {})

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        frame, t_idx = batch
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        result = self.pipeline.predict(frame, **self.predict_kwargs)
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
    **predict_kwargs
        Forwarded to every ``pipeline.predict(frame, **kwargs)`` call —
        e.g. ``prob_thresh`` / ``nms_thresh`` / ``n_tiles``.

    Returns
    -------
    dict
        Fields that the pipeline populates, each stacked along T:
        ``labels`` / ``semantic`` / ``roi`` / ``denoised`` / ``probability``.
        Missing fields are absent from the dict (not None-stuffed).
    """
    predictor = TimelapsePredictor(pipeline, predict_kwargs)
    loader = DataLoader(
        _FrameDataset(volume),
        batch_size=1,
        shuffle=False,
        collate_fn=_identity_collate,
        num_workers=0,
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

    # Lightning returns a list-of-batch-outputs; each batch is one frame
    # so each entry is a dict. Each rank only sees the slice the
    # DistributedSampler handed it, so under DDP we must gather across
    # ranks before stacking — otherwise rank 0 writes T/world_size
    # frames and the other ranks clobber that same file with their own
    # shard.
    per_frame = [d for d in per_frame if d is not None]

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
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
    for key in ("labels", "semantic", "roi", "denoised", "probability"):
        chunks = [d[key] for d in per_frame if d.get(key) is not None]
        if chunks:
            out[key] = np.stack(chunks, axis=0)
    return out
