"""MaskUNet singleton — first-class PyTorch implementation.

Operationally identical to :class:`UNetSegmenter` for inference (same
careamics UNet under the hood); kept as its own class so MaskUNet
checkpoints can be referenced by intent in user code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .._backbones.maskunet import MaskUNetBackbone
from .unet import UNetSegmenter


class MaskUNetSegmenter(UNetSegmenter):
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        **kwargs,
    ) -> MaskUNetSegmenter:
        backbone_kwargs = {
            k: kwargs.pop(k)
            for k in (
                "conv_dims",
                "in_channels",
                "num_classes",
                "depth",
                "num_channels_init",
                "use_batch_norm",
                "map_location",
            )
            if k in kwargs
        }
        return cls(
            MaskUNetBackbone.from_checkpoint(checkpoint, **backbone_kwargs), **kwargs
        )
