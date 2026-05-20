"""Modular, stackable training pipeline.

Same shape as ``lightning_kietzmannlab.lightning_trainer.TrainingPipeline``
— a single class whose ``setup_*`` methods each do one well-defined
thing and write into ``self``. yaml drives which ``setup_*`` methods
get called and in what order; the final :meth:`train` ties everything
together via a Lightning ``Trainer.fit`` call.

Why a single class instead of many trainers
-------------------------------------------
The old per-task trainers (``CARETrainer``, ``UNetTrainer``,
``StarDistTrainer``, ``MaskUNetTrainer``) duplicated 80 % of their
config plumbing (optimizer factory, scheduler factory, checkpoint
callback, CSV logger, save-paths, augmentation). The duplication kept
drifting between them; adding a cosine scheduler meant editing four
files. This pipeline absorbs that plumbing once. Each task-specific
trainer becomes a 30-line façade that just sequences the right
``setup_*`` calls.

The façades are still importable from ``kapoorlabs_vollseg`` for
backward compatibility — their ``__init__`` signatures + checkpoint
shapes are unchanged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from .._backbones.care import _build_unet
from .._lightning.registry import get_optimizer_factory, get_scheduler_factory
from ..care_lightning.module import CareModule
from ..stardist.lightning_module import StarDistModule
from ..stardist.model import StarDistUNet


@dataclass
class _PipelineState:
    """Just the bag of fields ``setup_*`` writes into."""

    # ── model + lightning shell ──
    network: Optional[nn.Module] = None
    module: Optional[LightningModule] = None
    loss_func: Optional[nn.Module] = None

    # ── optimizer / scheduler factories ──
    optim_factory: Optional[Callable[[Any], Any]] = None
    scheduler_factory: Optional[Callable[[Any], Any]] = None

    # ── data ──
    train_dataloader: Any = None
    val_dataloader: Any = None
    datamodule: Any = None

    # ── lightning runtime knobs ──
    epochs: int = 100
    accelerator: str = "auto"
    devices: Any = "auto"
    precision: str = "32-true"
    strategy: str = "auto"
    gradient_clip_val: Optional[float] = None

    # ── callbacks / logger ──
    callbacks: list = field(default_factory=list)
    logger: Any = None

    # ── arch knobs we record for the sidecar JSON ──
    arch_hparams: dict = field(default_factory=dict)


class TrainingPipeline:
    """Stackable training pipeline. Call ``setup_*`` methods to build up
    state, then :meth:`train` to run Lightning's fit loop.

    Method ordering is loosely enforced: :meth:`train` raises if the
    network or dataloaders aren't set up, but otherwise ``setup_*``
    methods can be called in any order so yaml-driven configs can omit
    pieces (defaults take over).

    Parameters
    ----------
    experiment_name, log_path
        Where checkpoints + the ``{experiment_name}.json`` /
        ``training_config.json`` sidecars land.
    epochs, learning_rate
        Trainer-wide defaults — individual ``setup_*`` methods can
        override.
    accelerator, devices, precision, strategy, gradient_clip_val
        Forwarded to :class:`lightning.Trainer` at :meth:`train` time.
    """

    def __init__(
        self,
        *,
        experiment_name: str,
        log_path: Union[str, Path],
        epochs: int = 100,
        learning_rate: float = 4e-4,
        accelerator: str = "auto",
        devices: Any = "auto",
        precision: str = "32-true",
        strategy: str = "auto",
        gradient_clip_val: Optional[float] = None,
    ):
        self.experiment_name = experiment_name
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.learning_rate = float(learning_rate)
        self.state = _PipelineState(
            epochs=int(epochs),
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            strategy=strategy,
            gradient_clip_val=gradient_clip_val,
        )

    # ─────────────────────────────────────────────────────── model setup

    def setup_unet_model(
        self,
        *,
        conv_dims: int = 3,
        in_channels: int = 1,
        num_classes: int = 1,
        unet_depth: int = 3,
        num_channels_init: int = 64,
        use_batch_norm: bool = True,
    ) -> TrainingPipeline:
        """Build a CAREamics UNet trunk — used for CARE denoising, U-Net
        semantic segmentation, ROI mask-UNet."""
        self.state.network = _build_unet(
            conv_dims=conv_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            depth=unet_depth,
            num_channels_init=num_channels_init,
            use_batch_norm=use_batch_norm,
        )
        self.state.arch_hparams.update(
            conv_dims=conv_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            unet_depth=unet_depth,
            num_channels_init=num_channels_init,
            use_batch_norm=use_batch_norm,
        )
        return self

    def setup_stardist_model(
        self,
        *,
        rays: np.ndarray,
        conv_dims: int = 3,
        in_channels: int = 1,
        unet_depth: int = 3,
        num_channels_init: int = 64,
        use_batch_norm: bool = True,
    ) -> TrainingPipeline:
        """Build StarDist's UNet + (prob_head, dist_head)."""
        rays = np.ascontiguousarray(rays, dtype=np.float32)
        if rays.ndim != 2:
            raise ValueError(f"rays must be 2D (N, ndim), got {rays.shape}")
        self.state.network = StarDistUNet(
            n_rays=rays.shape[0],
            conv_dims=conv_dims,
            in_channels=in_channels,
            depth=unet_depth,
            num_channels_init=num_channels_init,
            use_batch_norm=use_batch_norm,
        )
        self.state.arch_hparams.update(
            conv_dims=conv_dims,
            in_channels=in_channels,
            unet_depth=unet_depth,
            num_channels_init=num_channels_init,
            use_batch_norm=use_batch_norm,
            n_rays=int(rays.shape[0]),
        )
        # Stash rays so trainers can persist them next to the .ckpt.
        self._rays = rays
        return self

    def setup_custom_model(self, network: nn.Module) -> TrainingPipeline:
        """Plug an already-built ``nn.Module`` directly (escape hatch
        for non-UNet experiments)."""
        self.state.network = network
        return self

    # ─────────────────────────────────────────────────────── loss

    def setup_loss(self, loss_func: nn.Module) -> TrainingPipeline:
        self.state.loss_func = loss_func
        return self

    # ─────────────────────────────────────────── optimizer / scheduler

    def setup_optimizer(
        self,
        name: str = "adam",
        *,
        lr: Optional[float] = None,
        **kwargs,
    ) -> TrainingPipeline:
        """Resolve a string-keyed optimizer choice via the registry.

        ``name`` is one of ``adam`` / ``adamw`` / ``adamw_clip`` /
        ``sgd`` / ``rmsprop`` / ``rprop`` / ``lars``. ``lr`` defaults to
        the pipeline's ``learning_rate``; any other kwargs (``momentum``,
        ``weight_decay``, ``betas``, …) pass through to the optimizer.
        """
        if lr is None:
            lr = self.learning_rate
        self.state.optim_factory = get_optimizer_factory(name, lr=lr, **kwargs)
        return self

    # Explicit, kietzmann-style setup_<optimizer>() methods. Use these
    # when the choice is hardcoded in the script. The string-keyed
    # ``setup_optimizer(name)`` above is the equivalent yaml-driven
    # entry point — sweep scripts use that one.

    def setup_adam(self, **kwargs) -> TrainingPipeline:
        return self.setup_optimizer("adam", **kwargs)

    def setup_adamw(self, **kwargs) -> TrainingPipeline:
        return self.setup_optimizer("adamw", **kwargs)

    def setup_sgd(self, **kwargs) -> TrainingPipeline:
        return self.setup_optimizer("sgd", **kwargs)

    def setup_lars(self, **kwargs) -> TrainingPipeline:
        return self.setup_optimizer("lars", **kwargs)

    def setup_rmsprop(self, **kwargs) -> TrainingPipeline:
        return self.setup_optimizer("rmsprop", **kwargs)

    def setup_scheduler(
        self,
        name: Optional[str] = None,
        **kwargs,
    ) -> TrainingPipeline:
        """Resolve a string-keyed LR scheduler choice via the registry.

        ``name=None`` (default) → no scheduler. ``"cosine"`` /
        ``"warm_cosine"`` / ``"cosine_restart"`` / ``"multistep"`` /
        ``"plateau"`` / ``"linear"`` / ``"constant"`` / ``"exponential"``
        / ``"same"``. kwargs map to the scheduler's ``__init__``
        (``t_max``, ``eta_min``, ``milestones``, ``gamma``, …).
        """
        self.state.scheduler_factory = get_scheduler_factory(name, **kwargs)
        return self

    def setup_learning_rate_scheduler(self, **kwargs) -> TrainingPipeline:
        """Kietzmann-compatible alias for :meth:`setup_scheduler` — same
        name as ``CareInception.setup_learning_rate_scheduler`` so
        existing scripts that call this exact method name keep working."""
        return self.setup_scheduler(**kwargs)

    # ─────────────────────────────────────────── lightning module wrap

    def setup_care_module(
        self,
        *,
        n_tiles: Optional[list[int]] = None,
        tile_overlap: float = 0.125,
        loss_func: Optional[nn.Module] = None,
    ) -> TrainingPipeline:
        """Wrap the configured network in :class:`CareModule` (used by
        CARE denoising + U-Net segmentation + ROI mask-UNet)."""
        if self.state.network is None:
            raise RuntimeError(
                "setup_care_module called before a network is set up; "
                "call setup_unet_model first."
            )
        loss = loss_func or self.state.loss_func or nn.MSELoss()
        self.state.module = CareModule(
            network=self.state.network,
            loss_func=loss,
            optim_func=self.state.optim_factory,
            scheduler=self.state.scheduler_factory,
            n_tiles=n_tiles or [1, 4, 4],
            tile_overlap=float(tile_overlap),
        )
        return self

    def setup_stardist_module(
        self,
        *,
        n_tiles: Optional[list[int]] = None,
        tile_overlap: float = 0.125,
        loss_lam: float = 0.2,
    ) -> TrainingPipeline:
        """Wrap the configured network in :class:`StarDistModule`."""
        if self.state.network is None:
            raise RuntimeError(
                "setup_stardist_module called before a network is set up; "
                "call setup_stardist_model first."
            )
        self.state.module = StarDistModule(
            network=self.state.network,
            optim_func=self.state.optim_factory,
            scheduler=self.state.scheduler_factory,
            n_tiles=n_tiles or [1, 4, 4],
            tile_overlap=float(tile_overlap),
            loss_lam=float(loss_lam),
        )
        return self

    # ─────────────────────────────────────────────────────── data

    def set_dataloaders(
        self,
        train_dataloader: Any,
        val_dataloader: Optional[Any] = None,
    ) -> TrainingPipeline:
        """Escape hatch — attach already-built train + val dataloaders.
        Use this only when the dataset / collate is too task-specific
        for the ``setup_*_h5_datasets`` helpers below."""
        self.state.train_dataloader = train_dataloader
        self.state.val_dataloader = val_dataloader
        return self

    def set_datamodule(self, datamodule: Any) -> TrainingPipeline:
        self.state.datamodule = datamodule
        return self

    # ─── transforms ─────────────────────────────────────────────────

    def setup_stardist_transforms(
        self,
        *,
        pmin: float = 0.1,
        pmax: float = 99.9,
        augment: bool = True,
        flip_p: float = 0.5,
        rotation_p: float = 0.5,
        gaussian_noise_std: float = 0.0,
        gaussian_noise_p: float = 0.3,
    ) -> TrainingPipeline:
        """Build the StarDist train/val transform pair: percentile
        normalize + optional flip / rot90 / gaussian noise. Stored on
        ``self._stardist_train_tf`` and ``self._stardist_val_tf`` for
        :meth:`setup_stardist_h5_datasets` to pick up."""
        from ..stardist import (
            Compose,
            InputGaussianNoise,
            InputPercentileNormalize,
            RandomFlip,
            RandomRot90,
        )

        train = [InputPercentileNormalize(pmin=pmin, pmax=pmax)]
        if augment:
            train.append(RandomFlip(p=flip_p))
            train.append(RandomRot90(p=rotation_p))
            if gaussian_noise_std > 0:
                train.append(
                    InputGaussianNoise(std=gaussian_noise_std, p=gaussian_noise_p)
                )
        self._stardist_train_tf = Compose(train)
        self._stardist_val_tf = InputPercentileNormalize(pmin=pmin, pmax=pmax)
        return self

    def setup_unet_transforms(
        self,
        *,
        pmin: float = 0.1,
        pmax: float = 99.9,
        augment: bool = True,
        gaussian_noise_std: float = 0.0,
    ) -> TrainingPipeline:
        """Build the U-Net train/val transform pair — same shape as the
        StarDist one but operates on ``(raw, mask)`` pairs so flips and
        rotations are applied jointly to both tensors."""

        def _percentile_normalize(raw):
            flat = raw.flatten()
            lo = torch.quantile(flat, pmin / 100.0)
            hi = torch.quantile(flat, pmax / 100.0)
            return (raw - lo) / (hi - lo + 1e-8)

        def _train(raw, mask):
            raw = _percentile_normalize(raw)
            if augment:
                for axis in range(raw.dim()):
                    if torch.rand(1).item() < 0.5:
                        raw = torch.flip(raw, dims=[axis])
                        mask = torch.flip(mask, dims=[axis])
                if raw.dim() >= 2 and torch.rand(1).item() < 0.5:
                    k = int(torch.randint(0, 4, (1,)).item())
                    if k > 0:
                        raw = torch.rot90(raw, k=k, dims=[-2, -1])
                        mask = torch.rot90(mask, k=k, dims=[-2, -1])
                if gaussian_noise_std > 0 and torch.rand(1).item() < 0.3:
                    raw = raw + torch.randn_like(raw) * gaussian_noise_std
            return raw, mask

        def _val(raw, mask):
            return _percentile_normalize(raw), mask

        self._unet_train_tf = _train
        self._unet_val_tf = _val
        return self

    # ─── H5 datasets + dataloaders ─────────────────────────────────

    def setup_stardist_h5_datasets(
        self,
        *,
        h5_file: str,
        rays: np.ndarray,
        train_split: str = "train",
        val_split: str = "val",
        batch_size: int = 4,
        num_workers: int = 0,
    ) -> TrainingPipeline:
        """Build StarDist train + val H5 datasets and their dataloaders.

        Pulls the train transform from ``self._stardist_train_tf`` set
        by :meth:`setup_stardist_transforms` (call that first); falls
        back to percentile-normalize-only when neither is set."""
        from ..stardist import StarDistH5Dataset, stardist_collate

        train_tf = getattr(self, "_stardist_train_tf", None)
        val_tf = getattr(self, "_stardist_val_tf", None)
        if train_tf is None:
            self.setup_stardist_transforms(augment=False)
            train_tf = self._stardist_train_tf
            val_tf = self._stardist_val_tf

        train_ds = StarDistH5Dataset(
            h5_file, split=train_split, rays=rays, transform=train_tf
        )
        val_ds = StarDistH5Dataset(
            h5_file, split=val_split, rays=rays, transform=val_tf
        )
        self.state.train_dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=stardist_collate,
            persistent_workers=num_workers > 0,
        )
        self.state.val_dataloader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=stardist_collate,
            persistent_workers=num_workers > 0,
        )
        return self

    def setup_unet_h5_datasets(
        self,
        *,
        h5_file: str,
        train_split: str = "train",
        val_split: str = "val",
        batch_size: int = 4,
        num_workers: int = 0,
    ) -> TrainingPipeline:
        """Build U-Net train + val H5 datasets and their dataloaders.

        Pulls the transform pair from :meth:`setup_unet_transforms`;
        falls back to percentile-normalize-only when not set."""
        from .._lightning.unet_dataset import H5UNetDataset, unet_collate

        train_tf = getattr(self, "_unet_train_tf", None)
        val_tf = getattr(self, "_unet_val_tf", None)
        if train_tf is None:
            self.setup_unet_transforms(augment=False)
            train_tf = self._unet_train_tf
            val_tf = self._unet_val_tf

        train_ds = H5UNetDataset(h5_file, split=train_split, transform=train_tf)
        val_ds = H5UNetDataset(h5_file, split=val_split, transform=val_tf)
        self.state.train_dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=unet_collate,
            persistent_workers=num_workers > 0,
        )
        self.state.val_dataloader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=unet_collate,
            persistent_workers=num_workers > 0,
        )
        return self

    def setup_care_h5_datasets(
        self,
        *,
        h5_file: str,
        train_split: str = "train",
        val_split: str = "val",
        batch_size: int = 16,
        num_workers: int = 0,
        input_key: str = "low",
        target_key: str = "high",
    ) -> TrainingPipeline:
        """Build CARE train + val H5 datasets and their dataloaders.
        Uses :class:`H5CareDataset` from the lightning-port subpackage,
        so checkpoints trained with this path are byte-compatible with
        the ``xenopus_edge_enhancement`` weights."""
        from ..care_lightning.dataset import H5CareDataset

        train_ds = H5CareDataset(
            h5_file, split=train_split, input_key=input_key, target_key=target_key
        )
        val_ds = H5CareDataset(
            h5_file, split=val_split, input_key=input_key, target_key=target_key
        )
        self.state.train_dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        self.state.val_dataloader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        return self

    # ─────────────────────────────────────────── callbacks / logger

    def setup_checkpointing(
        self,
        *,
        save_top_k: int = -1,
        every_n_epochs: int = 1,
        save_last: bool = True,
    ) -> TrainingPipeline:
        """Add a flat :class:`ModelCheckpoint` that writes directly into
        ``log_path/`` (no ``lightning_logs/version_X/checkpoints/``
        nesting)."""
        self.state.callbacks.append(
            ModelCheckpoint(
                dirpath=os.fspath(self.log_path),
                filename=f"{self.experiment_name}-{{epoch:03d}}",
                save_last=save_last,
                save_top_k=save_top_k,
                every_n_epochs=every_n_epochs,
            )
        )
        return self

    def setup_csv_logger(self) -> TrainingPipeline:
        """Flat :class:`CSVLogger` writing ``log_path/metrics.csv`` (no
        version subfolder). Wipes stale rows so the column schema can
        change between runs without crashing Lightning's header
        rewriter."""
        for stale in ("metrics.csv", "hparams.yaml"):
            p = self.log_path / stale
            if p.exists():
                p.unlink()
        self.state.logger = CSVLogger(
            save_dir=os.fspath(self.log_path),
            name="",
            version="",
        )
        return self

    def add_callback(self, callback: Callback) -> TrainingPipeline:
        self.state.callbacks.append(callback)
        return self

    # ─────────────────────────────────────────────────────── train

    def save_hparams(self, extra: Optional[dict] = None) -> None:
        """Dump the recorded arch knobs to ``log_path/{experiment_name}.json``
        AND ``log_path/training_config.json`` (the second one is what
        ``from_folder`` reads at inference time)."""
        blob = {
            "experiment_name": self.experiment_name,
            "parameters": {**self.state.arch_hparams, **(extra or {})},
        }
        (self.log_path / f"{self.experiment_name}.json").write_text(
            json.dumps(blob["parameters"], indent=2)
        )
        (self.log_path / "training_config.json").write_text(json.dumps(blob, indent=2))

    def train(self) -> Trainer:
        """Run ``Trainer.fit``. Returns the :class:`lightning.Trainer`
        instance so the caller can read ``trainer.checkpoint_callback``
        / ``trainer.callback_metrics`` afterwards.

        Raises
        ------
        RuntimeError
            If neither a datamodule nor a train_dataloader was set up.
        """
        if self.state.module is None:
            raise RuntimeError(
                "Pipeline has no LightningModule — call setup_care_module "
                "or setup_stardist_module before train()."
            )
        if self.state.datamodule is None and self.state.train_dataloader is None:
            raise RuntimeError(
                "Pipeline has no data — call set_dataloaders or set_datamodule "
                "before train()."
            )

        # Default callbacks/logger if the user didn't add them.
        if not any(isinstance(cb, ModelCheckpoint) for cb in self.state.callbacks):
            self.setup_checkpointing()
        if self.state.logger is None:
            self.setup_csv_logger()

        trainer = Trainer(
            max_epochs=self.state.epochs,
            accelerator=self.state.accelerator,
            devices=self.state.devices,
            precision=self.state.precision,
            strategy=self.state.strategy,
            gradient_clip_val=self.state.gradient_clip_val,
            default_root_dir=os.fspath(self.log_path),
            callbacks=list(self.state.callbacks),
            logger=self.state.logger,
        )

        if self.state.datamodule is not None:
            trainer.fit(self.state.module, datamodule=self.state.datamodule)
        else:
            trainer.fit(
                self.state.module,
                train_dataloaders=self.state.train_dataloader,
                val_dataloaders=self.state.val_dataloader,
            )
        return trainer


__all__ = ["TrainingPipeline"]
