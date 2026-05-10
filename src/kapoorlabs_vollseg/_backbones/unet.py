"""PyTorch U-Net backbone — same careamics UNet wired for binary segmentation.

The architecture is the same shape as :class:`CAREBackbone`'s — that's
the whole point of moving to a single PyTorch backbone — but the head is
interpreted as logits for a binary mask, and the wrapping CareModule's
loss can be swapped to BCE at training time.

Inference stays in the ``Result.semantic`` / ``Result.labels`` shape so
:class:`kapoorlabs_vollseg.UNetSegmenter` and the Layer 2 composites don't need to
care which backbone they have.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from .._lightning.care_module import CareModule
from .care import _build_unet


class UNetBackbone:
    """Hold a CareModule whose network is interpreted as a binary segmenter."""

    def __init__(self, module: CareModule):
        self.module = module
        self.module.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        *,
        conv_dims: int = 3,
        in_channels: int = 1,
        num_classes: int = 1,
        depth: int = 3,
        num_channels_init: int = 64,
        use_batch_norm: bool = True,
        map_location: Optional[str] = None,
    ) -> UNetBackbone:
        unet = _build_unet(
            conv_dims=conv_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            depth=depth,
            num_channels_init=num_channels_init,
            use_batch_norm=use_batch_norm,
        )
        module = CareModule.load_from_checkpoint(
            checkpoint_path=str(checkpoint),
            network=unet,
            loss_func=torch.nn.BCEWithLogitsLoss(),
            optim_func=None,
            map_location=map_location,
        )
        return cls(module)
