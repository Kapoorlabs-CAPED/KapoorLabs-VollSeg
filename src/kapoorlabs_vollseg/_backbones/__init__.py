"""Backbones — first-class PyTorch + legacy keras siblings.

Bare names (``CAREBackbone``, ``UNetBackbone``, ``MaskUNetBackbone``,
``StarDistBackbone``, ``CellPoseBackbone``) are PyTorch.
``*Keras`` variants wrap csbdeep / stardist for already-trained weights.
"""

from .care import CAREBackbone
from .care_keras import CAREBackboneKeras
from .cellpose import CellPoseBackbone
from .maskunet import MaskUNetBackbone
from .maskunet_keras import MaskUNetBackboneKeras
from .stardist import StarDistBackbone
from .stardist_keras import StarDist2DBackboneKeras, StarDist3DBackboneKeras
from .unet import UNetBackbone
from .unet_keras import UNetBackboneKeras

__all__ = [
    # PyTorch first-class
    "CAREBackbone",
    "UNetBackbone",
    "MaskUNetBackbone",
    "StarDistBackbone",
    "CellPoseBackbone",
    # Keras legacy
    "CAREBackboneKeras",
    "UNetBackboneKeras",
    "MaskUNetBackboneKeras",
    "StarDist2DBackboneKeras",
    "StarDist3DBackboneKeras",
]
