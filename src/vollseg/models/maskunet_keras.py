"""MaskUNet singleton (keras / csbdeep) — legacy.

New code should use :class:`vollseg.MaskUNetSegmenter` (PyTorch + Lightning).
"""

from __future__ import annotations

from .._backbones.maskunet_keras import MaskUNetBackboneKeras
from .unet_keras import UNetSegmenterKeras


class MaskUNetSegmenterKeras(UNetSegmenterKeras):
    @classmethod
    def from_pretrained(cls, name_or_alias: str, **kwargs) -> "MaskUNetSegmenterKeras":
        from ..pretrained import get_model_instance
        return cls(get_model_instance(MaskUNetBackboneKeras, name_or_alias), **kwargs)
