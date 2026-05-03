"""StarDist trainer (keras / stardist) — currently the only StarDist trainer.

A PyTorch StarDist trainer is planned. Keeping the ``Keras`` suffix for
parity with other backbones; switch to a bare ``StarDistTrainer`` when
the PyTorch port lands.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from stardist import Rays_GoldenSpiral, calculate_extents
from stardist.models import Config2D, Config3D

from .._backbones.stardist_keras import StarDist2DBackboneKeras, StarDist3DBackboneKeras
from ._checkpoint import load_latest_checkpoint


class StarDistTrainerKeras:
    """Train a StarDist 2D or 3D model."""

    def __init__(
        self,
        *,
        ndim: int,
        model_name: str,
        model_dir: Union[str, Path],
        backbone: str = "resnet",
        n_rays: int = 16,
        grid: Sequence[int] = (1, 1, 1),
        anisotropy: Optional[Tuple[float, ...]] = None,
        epochs: int = 400,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        depth: int = 3,
        kern_size: int = 3,
        startfilter: int = 48,
        patch_size: Tuple[int, ...] = (16, 256, 256),
        n_channel_in: int = 1,
        use_gpu: bool = True,
        train_dist_loss: str = "mse",
    ):
        if ndim not in (2, 3):
            raise ValueError(f"ndim must be 2 or 3, got {ndim}")
        self.ndim = ndim
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.backbone_choice = backbone
        self.n_rays = n_rays
        self.grid = tuple(grid)
        self.anisotropy = anisotropy
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.depth = depth
        self.kern_size = kern_size
        self.startfilter = startfilter
        self.patch_size = tuple(patch_size)
        self.n_channel_in = n_channel_in
        self.use_gpu = use_gpu
        self.train_dist_loss = train_dist_loss

    def fit(self, X_trn, Y_trn, *, validation_data: Optional[Tuple] = None):
        anisotropy, rays = self._anisotropy_and_rays(Y_trn)
        config = self._build_config(anisotropy, rays)

        bb_cls = StarDist3DBackboneKeras if self.ndim == 3 else StarDist2DBackboneKeras
        model = bb_cls(config, name=self.model_name, basedir=os.fspath(self.model_dir))
        load_latest_checkpoint(model, self.model_dir, self.model_name)

        history = model.train(
            X_trn, Y_trn, validation_data=validation_data, epochs=self.epochs
        )
        return history, model

    # --------------------------------------------------------- helpers

    def _anisotropy_and_rays(self, Y):
        if self.anisotropy is not None:
            return self.anisotropy, Rays_GoldenSpiral(self.n_rays, anisotropy=self.anisotropy)
        if self.ndim == 3:
            try:
                extents = calculate_extents(list(Y))
                aniso = tuple(extents.max() / extents)
                return aniso, Rays_GoldenSpiral(self.n_rays, anisotropy=aniso)
            except Exception:
                return None, self.n_rays
        return None, self.n_rays

    def _build_config(self, anisotropy, rays):
        common = dict(
            train_epochs=self.epochs,
            train_learning_rate=self.learning_rate,
            train_patch_size=self.patch_size,
            train_batch_size=self.batch_size,
            train_dist_loss=self.train_dist_loss,
            n_channel_in=self.n_channel_in,
            train_checkpoint=os.fspath(self.model_dir / f"{self.model_name}.h5"),
            grid=self.grid,
            use_gpu=self.use_gpu,
            backbone=self.backbone_choice,
        )
        if self.backbone_choice == "resnet":
            arch = dict(
                resnet_n_blocks=self.depth,
                resnet_kernel_size=(self.kern_size,) * self.ndim,
                resnet_n_filter_base=self.startfilter,
            )
        else:
            arch = dict(
                unet_n_depth=self.depth,
                unet_kernel_size=(self.kern_size,) * self.ndim,
                unet_n_filter_base=self.startfilter,
            )
        cfg_cls = Config3D if self.ndim == 3 else Config2D
        kwargs = dict(rays=rays, **common, **arch)
        if self.ndim == 3:
            kwargs["anisotropy"] = anisotropy
        return cfg_cls(**kwargs)
