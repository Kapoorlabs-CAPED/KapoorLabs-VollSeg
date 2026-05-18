"""Lightning base module — minimal port of ``kapoorlabs_lightning.BaseModule``.

Only the pieces the CARE / UNet inference path actually needs:
``forward``, ``loss``, ``log_metrics``, ``configure_optimizers``. Scheduler
restoration logic isn't reproduced here — trainers can do that in their
own ``setup`` hooks.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional

import torch
import torch.nn as nn
from lightning import LightningModule


class BaseModule(LightningModule):
    """Wrap a network + loss + optimizer factory under a Lightning shell."""

    def __init__(
        self,
        network: nn.Module,
        loss_func: Optional[nn.Module] = None,
        optim_func: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        *,
        on_step: bool = False,
        on_epoch: bool = True,
        sync_dist: bool = True,
        rank_zero_only: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["network", "loss_func", "optim_func", "scheduler"],
        )
        self.network = network
        self.loss_func = loss_func
        self.optim_func = optim_func
        self.scheduler = scheduler
        self._on_step = on_step
        self._on_epoch = on_epoch
        self._sync_dist = sync_dist
        self._rank_zero_only = rank_zero_only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_func(y_hat, y)

    def log_metrics(
        self,
        name: str,
        value: torch.Tensor,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
    ) -> None:
        self.log(
            name,
            value,
            on_step=self._on_step if on_step is None else on_step,
            on_epoch=self._on_epoch if on_epoch is None else on_epoch,
            prog_bar=True,
            logger=True,
            sync_dist=self._sync_dist,
            rank_zero_only=self._rank_zero_only,
        )

    def configure_optimizers(self):
        if self.optim_func is None:
            return None
        optimizer = self.optim_func(self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return OrderedDict(
                {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                        "frequency": 1,
                    },
                }
            )
        return {"optimizer": optimizer}
