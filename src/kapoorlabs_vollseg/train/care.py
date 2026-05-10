"""CARE denoising trainer — first-class PyTorch implementation.

A leaner reflection of the kapoorlabs-lightning ``CareInception``
orchestrator, but self-contained: builds the careamics UNet, wires up
:class:`CareModule`, and hands them to a Lightning ``Trainer``. The user
brings their own ``LightningDataModule`` (or a pair of DataLoaders) so
that this trainer doesn't lock anyone into a specific data layout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from .._backbones.care import _build_unet
from .._lightning.care_module import CareModule


class CARETrainer:
    """Train a CARE denoising model under PyTorch Lightning.

    Parameters
    ----------
    model_name, model_dir
        Where checkpoints + the hyperparameter JSON are written.
    epochs, batch_size, learning_rate
        Standard training knobs.
    unet_depth, num_channels_init, use_batch_norm
        UNet architecture (passed to ``careamics.models.unet.UNet``).
    n_tiles, tile_overlap
        Forwarded to :class:`CareModule` for inference-time tiling.
    accelerator, devices, precision, strategy
        Forwarded to ``lightning.Trainer``.
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_dir: Union[str, Path],
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 4e-4,
        # UNet architecture
        conv_dims: int = 3,
        in_channels: int = 1,
        num_classes: int = 1,
        unet_depth: int = 3,
        num_channels_init: int = 64,
        use_batch_norm: bool = True,
        # CareModule prediction-time defaults
        n_tiles: Optional[list] = None,
        tile_overlap: float = 0.125,
        # Lightning runtime
        accelerator: str = "auto",
        devices: Any = "auto",
        precision: str = "32-true",
        strategy: str = "auto",
        loss_func: Optional[nn.Module] = None,
        optim_factory: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.conv_dims = conv_dims
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.unet_depth = unet_depth
        self.num_channels_init = num_channels_init
        self.use_batch_norm = use_batch_norm

        self.n_tiles = list(n_tiles) if n_tiles is not None else [1, 4, 4]
        self.tile_overlap = tile_overlap
        self.accelerator = accelerator
        self.devices = devices
        self.precision = precision
        self.strategy = strategy

        self.loss_func = loss_func or nn.MSELoss()
        self.optim_factory = optim_factory or (
            lambda params: torch.optim.Adam(params, lr=self.learning_rate)
        )

    def fit(
        self,
        datamodule=None,
        *,
        train_dataloader=None,
        val_dataloader=None,
        callbacks=None,
        logger=None,
    ):
        """Train the model. Provide either ``datamodule`` or the two dataloaders."""
        from lightning import Trainer

        unet = _build_unet(
            conv_dims=self.conv_dims,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            depth=self.unet_depth,
            num_channels_init=self.num_channels_init,
            use_batch_norm=self.use_batch_norm,
        )
        module = CareModule(
            network=unet,
            loss_func=self.loss_func,
            optim_func=self.optim_factory,
            n_tiles=self.n_tiles,
            tile_overlap=self.tile_overlap,
        )

        self._save_hparams()

        trainer = Trainer(
            max_epochs=self.epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            precision=self.precision,
            strategy=self.strategy,
            default_root_dir=os.fspath(self.model_dir),
            callbacks=callbacks or [],
            logger=logger,
        )
        if datamodule is not None:
            trainer.fit(module, datamodule=datamodule)
        else:
            trainer.fit(
                module,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
        return trainer, module

    def _save_hparams(self) -> None:
        out = {
            "conv_dims": self.conv_dims,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "unet_depth": self.unet_depth,
            "num_channels_init": self.num_channels_init,
            "use_batch_norm": self.use_batch_norm,
            "n_tiles": self.n_tiles,
            "tile_overlap": self.tile_overlap,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "model_name": self.model_name,
        }
        with open(self.model_dir / f"{self.model_name}.json", "w") as f:
            json.dump(out, f, indent=2)
