"""MaskUNet singleton — U-Net trained to predict both segmentation + mask.

Operationally identical to :class:`UNetSegmenter` for inference (the
distinction is only meaningful for trained weights), so this is a thin
subclass that swaps in the :class:`MaskUNetBackbone`.
"""

from __future__ import annotations

from .._backbones import MaskUNetBackbone
from .unet import UNetSegmenter


class MaskUNetSegmenter(UNetSegmenter):
    @classmethod
    def from_pretrained(cls, name_or_alias: str, **kwargs) -> "MaskUNetSegmenter":
        from ..pretrained import get_model_instance
        return cls(get_model_instance(MaskUNetBackbone, name_or_alias), **kwargs)
