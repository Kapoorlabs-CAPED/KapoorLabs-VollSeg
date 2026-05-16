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

    @classmethod
    def from_folder(cls, folder: Union[str, Path], **kwargs) -> MaskUNetSegmenter:
        """Build from a model folder containing the ROI ``.ckpt`` plus
        ``training_config.json`` (or fallback ``{experiment_name}.json``).
        Per ``KapoorLabs-Lightning/scripts/conf/parameters/roi.yaml``,
        ROI Mask-UNet trains with ``conv_dims=2``, so the JSON puts
        that in ``parameters.conv_dims`` and the loader auto-builds a
        2D backbone — no manual override required."""
        from .._backbones._config import find_checkpoint, read_training_config

        ckpt = find_checkpoint(folder)
        arch = read_training_config(folder)
        arch.update(kwargs)
        return cls.from_checkpoint(ckpt, **arch)
