"""StarDist training losses.

Two terms summed:

- ``prob_loss`` — BCE-with-logits between the predicted object-probability
  logits and the EDT-normalized target (from
  :func:`vollseg.stardist.foreground_probability_map`).
- ``dist_loss`` — masked L1 between predicted and target ray distances.
  Only foreground pixels (where prob_target > 0) contribute, with each
  pixel weighted by its prob target so points near the object center
  drive the gradient (matches upstream stardist's behavior).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def prob_loss(prob_logits: torch.Tensor, prob_target: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits between predicted and target object probability.

    Both inputs are ``(B, 1, *spatial)``. Target should be in [0, 1] —
    ``foreground_probability_map`` produces this shape.
    """
    return F.binary_cross_entropy_with_logits(prob_logits, prob_target)


def dist_loss(
    dists: torch.Tensor,
    dist_target: torch.Tensor,
    prob_target: torch.Tensor,
    *,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Foreground-masked weighted L1 on radial distances.

    Parameters
    ----------
    dists
        ``(B, n_rays, *spatial)`` predicted distances.
    dist_target
        ``(B, n_rays, *spatial)`` target distances.
    prob_target
        ``(B, 1, *spatial)`` — used both as foreground mask
        (``> 0``) and as a per-pixel weight (so center voxels matter
        more than boundary voxels).
    eps
        Numerical floor for the normalizer.
    """
    weights = prob_target.expand_as(dists)            # broadcast (B,1,*) -> (B,N,*)
    abs_err = (dists - dist_target).abs()
    return (abs_err * weights).sum() / (weights.sum() + eps)


def stardist_loss(
    prob_logits: torch.Tensor,
    dists: torch.Tensor,
    prob_target: torch.Tensor,
    dist_target: torch.Tensor,
    *,
    lam: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Composite loss = ``prob_loss + lam * dist_loss``.

    Returns the total loss plus the two component losses (so the
    Lightning module can log each separately).

    The ``lam`` default of ``0.2`` matches the upstream stardist
    ``train_loss_weights`` heuristic for 3D models with ~96 rays.
    """
    p = prob_loss(prob_logits, prob_target)
    d = dist_loss(dists, dist_target, prob_target)
    return p + lam * d, p, d
