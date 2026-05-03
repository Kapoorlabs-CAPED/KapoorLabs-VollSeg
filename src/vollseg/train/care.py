"""CARE denoising trainer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

from csbdeep.models import Config

from .._backbones import CAREBackbone
from ._checkpoint import load_latest_checkpoint


class CARETrainer:
    """Train a CARE denoising network.

    Parameters
    ----------
    model_name, model_dir
        Where the trained model is saved.
    axes, n_channel_in, n_channel_out
        Axes string and channel counts for csbdeep ``Config``.
    epochs, batch_size, learning_rate
        Standard training knobs.
    unet_n_depth, unet_n_first, unet_kern_size
        U-Net architecture.
    train_loss
        Loss name accepted by csbdeep (default: ``mae``).
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_dir: Union[str, Path],
        axes: str = "ZYXC",
        n_channel_in: int = 1,
        n_channel_out: int = 1,
        epochs: int = 400,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        unet_n_depth: int = 3,
        unet_n_first: int = 48,
        unet_kern_size: int = 3,
        train_loss: str = "mae",
        train_reduce_lr: dict = None,
    ):
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.config = Config(
            axes,
            n_channel_in,
            n_channel_out,
            unet_n_depth=unet_n_depth,
            train_epochs=epochs,
            train_batch_size=batch_size,
            unet_n_first=unet_n_first,
            train_loss=train_loss,
            unet_kern_size=unet_kern_size,
            train_learning_rate=learning_rate,
            train_reduce_lr=train_reduce_lr or {"patience": 5, "factor": 0.5},
        )

    def fit(
        self,
        X,
        Y=None,
        *,
        validation_data: Optional[Tuple] = None,
        load_data_sequence: bool = False,
    ):
        """Train and return the keras history.

        ``X`` and ``Y`` are paired arrays for the in-memory path; pass a
        keras ``Sequence`` as ``X`` and set ``load_data_sequence=True`` for
        the streaming path.
        """
        model = CAREBackbone(self.config, name=self.model_name, basedir=os.fspath(self.model_dir))
        load_latest_checkpoint(model, self.model_dir, self.model_name)
        history = model.train(
            X, Y,
            validation_data=validation_data,
            load_data_sequence=load_data_sequence,
        )
        return history, model
