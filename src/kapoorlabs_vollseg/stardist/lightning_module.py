"""Lightning module for StarDist training + tiled prediction.

Wraps :class:`StarDistUNet` and :func:`stardist_loss`. Mirrors the shape
of :class:`kapoorlabs_vollseg._lightning.CareModule` (same ``BaseModule`` parent,
same ``predict_step`` contract) so the rest of the codebase doesn't have
to special-case StarDist anywhere — the same DataLoader-based tiling
pipeline works.
"""

from __future__ import annotations

from typing import Optional

import torch

from .._lightning.base_module import BaseModule
from .losses import stardist_loss
from .model import StarDistUNet


class StarDistModule(BaseModule):
    """Predict ``(object_prob, ray_distances)`` from raw 3D / 2D patches."""

    def __init__(
        self,
        network: StarDistUNet,
        loss_func=None,  # unused; stardist_loss is fixed
        optim_func=None,
        scheduler=None,
        *,
        n_tiles: Optional[list[int]] = None,
        tile_overlap: float = 0.125,
        loss_lam: float = 0.2,
    ):
        super().__init__(
            network=network,
            loss_func=loss_func,
            optim_func=optim_func,
            scheduler=scheduler,
        )
        if not isinstance(network, StarDistUNet):
            raise TypeError(
                f"StarDistModule.network must be a StarDistUNet, "
                f"got {type(network).__name__}"
            )
        self.n_rays = network.n_rays
        self.conv_dims = network.conv_dims
        self.n_tiles = list(n_tiles) if n_tiles is not None else [1, 4, 4]
        self.tile_overlap = float(tile_overlap)
        self.loss_lam = float(loss_lam)

    # --------------------------------------------------------- training

    def training_step(self, batch, batch_idx):
        raw, prob_target, dist_target = batch
        prob_logits, dists = self(raw)
        total, p_term, d_term = stardist_loss(
            prob_logits, dists, prob_target, dist_target, lam=self.loss_lam
        )
        self.log_metrics("train_loss", total)
        self.log_metrics("train_prob_loss", p_term)
        self.log_metrics("train_dist_loss", d_term)
        try:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log_metrics("learning_rate", torch.tensor(lr))
        except Exception:
            pass
        return total

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")

    def _eval_step(self, batch, prefix: str):
        raw, prob_target, dist_target = batch
        prob_logits, dists = self(raw)
        total, p_term, d_term = stardist_loss(
            prob_logits, dists, prob_target, dist_target, lam=self.loss_lam
        )
        self.log_metrics(f"{prefix}_loss", total)
        self.log_metrics(f"{prefix}_prob_loss", p_term)
        self.log_metrics(f"{prefix}_dist_loss", d_term)

    # ------------------------------------------------------- prediction

    def predict_step(self, batch, batch_idx):
        """Return ``(prob_sigmoid, dists, coords)`` for stitching downstream.

        ``batch`` matches the shape produced by
        :class:`kapoorlabs_vollseg._lightning.CarePredictionDataset` — one tile
        tensor and one coords tensor — but the tile is *not* unsqueezed
        here (CarePredictionDataset returns ``(tile, coords)`` where
        tile is ``(B, *spatial)``; we add the channel dim ourselves).

        The returned tensors:

        - ``prob_sigmoid`` — ``(B, 1, *spatial)`` in [0, 1]
        - ``dists``        — ``(B, n_rays, *spatial)`` raw distances
        - ``coords``       — ``(B, 6)`` per-tile origin + size, untouched
        """
        tiles, coords = batch
        if tiles.dim() == self.conv_dims + 1:  # (B, *spatial) — add channel
            tiles = tiles.unsqueeze(1)
        with torch.no_grad():
            prob_logits, dists = self(tiles)
        prob = torch.sigmoid(prob_logits)
        return prob.cpu(), dists.cpu(), coords.cpu()
