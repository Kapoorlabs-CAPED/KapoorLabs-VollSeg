"""StarDist model — careamics UNet with two output heads.

The trunk is the same `careamics.models.unet.UNet` we use everywhere
else. We override its single output projection with two heads:

- ``prob_head``  → 1 channel (object probability; sigmoid in inference)
- ``dist_head``  → ``n_rays`` channels (radial distances; raw / linear)

The wrapping :class:`StarDistUNet` is a plain ``nn.Module`` with a clean
``forward(x) -> (prob_logits, dists)`` shape. The Lightning training
module (next commit) wraps it.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _build_trunk(
    *,
    conv_dims: int,
    in_channels: int,
    depth: int,
    num_channels_init: int,
    use_batch_norm: bool,
) -> Tuple[nn.Module, int]:
    """Build the careamics UNet trunk and report its final feature width.

    careamics' UNet ends in a 1×1×1 conv to ``num_classes`` channels — we
    force that to ``num_channels_init`` so we can attach our own heads
    afterwards rather than chaining off a 1-channel bottleneck.
    """
    from careamics.models.unet import UNet
    trunk = UNet(
        conv_dims=conv_dims,
        in_channels=in_channels,
        num_classes=num_channels_init,    # treat the "head" channels as features
        depth=depth,
        num_channels_init=num_channels_init,
        use_batch_norm=use_batch_norm,
    )
    return trunk, num_channels_init


class StarDistUNet(nn.Module):
    """Trunk + (prob head, dist head). 2D or 3D, dispatched by ``conv_dims``.

    Parameters
    ----------
    n_rays
        Number of radial-distance channels.
    conv_dims
        ``2`` or ``3`` — picks the trunk's convolution dimensionality.
    in_channels
        Input image channels (typically 1 for grayscale microscopy).
    depth, num_channels_init, use_batch_norm
        Standard careamics UNet knobs.
    """

    def __init__(
        self,
        *,
        n_rays: int,
        conv_dims: int = 3,
        in_channels: int = 1,
        depth: int = 3,
        num_channels_init: int = 64,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        if conv_dims not in (2, 3):
            raise ValueError(f"conv_dims must be 2 or 3, got {conv_dims}")
        self.n_rays = int(n_rays)
        self.conv_dims = conv_dims

        self.trunk, feat = _build_trunk(
            conv_dims=conv_dims,
            in_channels=in_channels,
            depth=depth,
            num_channels_init=num_channels_init,
            use_batch_norm=use_batch_norm,
        )

        ConvNd = nn.Conv3d if conv_dims == 3 else nn.Conv2d
        # 1×1(×1) projection from trunk features to each head.
        self.prob_head = ConvNd(feat, 1, kernel_size=1)
        self.dist_head = ConvNd(feat, self.n_rays, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the trunk and both heads.

        Returns
        -------
        prob_logits : torch.Tensor
            ``(B, 1, *spatial)`` raw logits — apply ``sigmoid`` for
            probabilities. Trained against the EDT-derived target with
            BCE-with-logits.
        dists : torch.Tensor
            ``(B, n_rays, *spatial)`` predicted distances, in voxels.
            Trained against the ray-march target with masked L1.
        """
        feats = self.trunk(x)
        return self.prob_head(feats), self.dist_head(feats)


def split_outputs(out: torch.Tensor, n_rays: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a single ``(B, 1+n_rays, *spatial)`` tensor into ``(prob, dists)``.

    Convenience for code paths that prefer concatenated outputs (e.g.
    when serializing the head outputs as one tensor for caching).
    """
    if out.shape[1] != 1 + n_rays:
        raise ValueError(
            f"Expected channel dim = 1 + n_rays = {1 + n_rays}, got {out.shape[1]}"
        )
    return out[:, :1], out[:, 1:]
