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

from ..care_lightning.module import CareModule
from .care import _build_unet, infer_arch_from_checkpoint


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
        conv_dims: Optional[int] = None,
        in_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
        depth: Optional[int] = None,
        num_channels_init: Optional[int] = None,
        use_batch_norm: Optional[bool] = None,
        map_location: Optional[str] = None,
        weights_only: bool = False,
    ) -> UNetBackbone:
        """As :meth:`CAREBackbone.from_checkpoint` — see that docstring for
        the auto-detection contract. ROI Mask-UNet (2D) and nuclei /
        membrane U-Net (3D) both flow through here unchanged."""
        arch = infer_arch_from_checkpoint(checkpoint, weights_only=weights_only)
        conv_dims = conv_dims if conv_dims is not None else arch["conv_dims"]
        in_channels = in_channels if in_channels is not None else arch["in_channels"]
        num_classes = num_classes if num_classes is not None else arch["num_classes"]
        depth = depth if depth is not None else arch["depth"]
        num_channels_init = (
            num_channels_init
            if num_channels_init is not None
            else arch["num_channels_init"]
        )
        use_batch_norm = (
            use_batch_norm if use_batch_norm is not None else arch["use_batch_norm"]
        )
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
            weights_only=weights_only,
        )
        return cls(module)
