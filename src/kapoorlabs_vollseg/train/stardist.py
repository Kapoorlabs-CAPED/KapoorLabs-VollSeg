"""StarDist trainer — first-class PyTorch implementation.

Builds :class:`StarDistUNet`, wraps it in :class:`StarDistModule`, and
hands the pair to ``lightning.Trainer``. Saves the rays array as a
sidecar ``{model_name}.rays.npy`` next to the checkpoint so prediction
can rebuild the matching backbone.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from ..stardist.lightning_module import StarDistModule
from ..stardist.model import StarDistUNet


class StarDistTrainer:
    """Train a StarDist model under PyTorch Lightning."""

    def __init__(
        self,
        *,
        model_name: str,
        model_dir: Union[str, Path],
        rays: np.ndarray,
        epochs: int = 100,
        batch_size: int = 4,
        learning_rate: float = 4e-4,
        # UNet architecture
        conv_dims: int = 3,
        in_channels: int = 1,
        depth: int = 3,
        num_channels_init: int = 64,
        use_batch_norm: bool = True,
        # StarDist-specific
        loss_lam: float = 0.2,
        # Inference defaults stored on the module
        n_tiles: Optional[list] = None,
        tile_overlap: float = 0.125,
        # Lightning runtime
        accelerator: str = "auto",
        devices: Any = "auto",
        precision: str = "32-true",
        strategy: str = "auto",
        optim_factory: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.rays = np.ascontiguousarray(rays, dtype=np.float32)
        if self.rays.ndim != 2:
            raise ValueError(f"rays must be 2D (N, ndim), got {self.rays.shape}")

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.conv_dims = conv_dims
        self.in_channels = in_channels
        self.depth = depth
        self.num_channels_init = num_channels_init
        self.use_batch_norm = use_batch_norm
        self.loss_lam = loss_lam
        self.n_tiles = list(n_tiles) if n_tiles is not None else [1, 4, 4]
        self.tile_overlap = tile_overlap
        self.accelerator = accelerator
        self.devices = devices
        self.precision = precision
        self.strategy = strategy
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
        """Train. Provide either ``datamodule`` or train/val dataloaders."""
        from lightning import Trainer

        unet = StarDistUNet(
            n_rays=self.rays.shape[0],
            conv_dims=self.conv_dims,
            in_channels=self.in_channels,
            depth=self.depth,
            num_channels_init=self.num_channels_init,
            use_batch_norm=self.use_batch_norm,
        )
        module = StarDistModule(
            network=unet,
            optim_func=self.optim_factory,
            n_tiles=self.n_tiles,
            tile_overlap=self.tile_overlap,
            loss_lam=self.loss_lam,
        )

        # Persist rays + arch knobs alongside the ckpt so prediction can rebuild.
        rays_path = self.model_dir / f"{self.model_name}.rays.npy"
        np.save(rays_path, self.rays)
        self._save_hparams(rays_path)

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

    def _save_hparams(self, rays_path: Path) -> None:
        out = {
            "model_name": self.model_name,
            "rays_file": rays_path.name,
            "n_rays": int(self.rays.shape[0]),
            "conv_dims": self.conv_dims,
            "in_channels": self.in_channels,
            "depth": self.depth,
            "num_channels_init": self.num_channels_init,
            "use_batch_norm": self.use_batch_norm,
            "loss_lam": self.loss_lam,
            "n_tiles": self.n_tiles,
            "tile_overlap": self.tile_overlap,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }
        with open(self.model_dir / f"{self.model_name}.json", "w") as f:
            json.dump(out, f, indent=2)
