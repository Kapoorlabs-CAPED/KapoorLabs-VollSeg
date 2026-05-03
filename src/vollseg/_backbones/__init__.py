"""Backbones — first-class PyTorch + legacy keras siblings.

The bare-named classes (``CAREBackbone``, ``UNetBackbone``,
``MaskUNetBackbone``) are PyTorch + careamics + Lightning. The
``*Keras`` variants wrap csbdeep / stardist for backwards compatibility
with already-trained weights.
"""

from .care import CAREBackbone
from .care_keras import CAREBackboneKeras
from .cellpose import CellPoseBackbone
from .maskunet import MaskUNetBackbone
from .maskunet_keras import MaskUNetBackboneKeras
from .stardist_keras import StarDist2DBackboneKeras, StarDist3DBackboneKeras
from .unet import UNetBackbone
from .unet_keras import UNetBackboneKeras

__all__ = [
    # PyTorch first-class
    "CAREBackbone",
    "UNetBackbone",
    "MaskUNetBackbone",
    "CellPoseBackbone",
    # Keras legacy
    "CAREBackboneKeras",
    "UNetBackboneKeras",
    "MaskUNetBackboneKeras",
    "StarDist2DBackboneKeras",
    "StarDist3DBackboneKeras",
]
