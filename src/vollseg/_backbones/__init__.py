"""Backbone classes — thin subclasses of csbdeep / stardist models.

These exist because csbdeep/stardist need a concrete class on disk to load
weights from. They are not the user-facing API: see :mod:`vollseg.models`
for the inference singletons that wrap them.
"""

from .care import CAREBackbone
from .maskunet import MaskUNetBackbone
from .stardist import StarDist2DBackbone, StarDist3DBackbone
from .unet import UNetBackbone

__all__ = [
    "CAREBackbone",
    "UNetBackbone",
    "MaskUNetBackbone",
    "StarDist2DBackbone",
    "StarDist3DBackbone",
]
