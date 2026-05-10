"""CARE Lightning module — supervised denoising on 3D volumes.

Mirrors the upstream ``CareModule`` shape (``unsqueeze(1)`` for the
channel dim, paired ``(low, high)`` batches, PSNR logging) so checkpoints
trained with kapoorlabs-lightning load here without translation.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base_module import BaseModule


class CareModule(BaseModule):
    """Predict ``high`` from ``low`` SNR 3D patches."""

    def __init__(
        self,
        network: torch.nn.Module,
        loss_func: Optional[torch.nn.Module] = None,
        optim_func=None,
        scheduler=None,
        *,
        n_tiles: Optional[list[int]] = None,
        tile_overlap: float = 0.125,
        eval_transforms=None,
    ):
        super().__init__(
            network=network,
            loss_func=loss_func,
            optim_func=optim_func,
            scheduler=scheduler,
        )
        self.n_tiles = list(n_tiles) if n_tiles is not None else [1, 4, 4]
        self.tile_overlap = float(tile_overlap)
        self.eval_transforms = eval_transforms

    # --------------------------------------------------------- training

    def training_step(self, batch, batch_idx):
        low, high = batch
        low = low.unsqueeze(1)
        high = high.unsqueeze(1)
        predicted = self(low)
        loss = self.loss_func(predicted, high)
        self.log_metrics("train_loss", loss)
        self.log_metrics("train_psnr", _psnr(predicted, high))
        try:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log_metrics("learning_rate", torch.tensor(lr))
        except Exception:
            pass
        return loss

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")

    def _eval_step(self, batch, prefix: str):
        low, high = batch
        low = low.unsqueeze(1)
        high = high.unsqueeze(1)
        predicted = self(low)
        loss = self.loss_func(predicted, high)
        self.log_metrics(f"{prefix}_loss", loss)
        self.log_metrics(f"{prefix}_psnr", _psnr(predicted, high))

    # ------------------------------------------------------- prediction

    def predict_step(self, batch, batch_idx):
        """Return ``(predicted_tile, coords)`` for stitching downstream."""
        tiles, coords = batch
        tiles = tiles.unsqueeze(1)
        with torch.no_grad():
            predicted = self(tiles)
        predicted = predicted.squeeze(1)
        return predicted.cpu(), coords.cpu()


def _psnr(
    pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0
) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"))
    return 10 * torch.log10(max_val**2 / mse)
