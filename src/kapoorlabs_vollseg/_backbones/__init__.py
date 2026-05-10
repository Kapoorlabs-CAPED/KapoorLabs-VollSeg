"""Backbones — first-class PyTorch + (optional) legacy keras siblings.

Bare names (``CAREBackbone``, ``UNetBackbone``, ``MaskUNetBackbone``,
``StarDistBackbone``, ``CellPoseBackbone``) are PyTorch and always
available.

The ``*Keras`` variants require the optional ``[keras]`` extra
(``pip install kapoorlabs-vollseg[keras]``); if csbdeep / stardist
aren't installed, those names simply aren't exposed here — importing
the package on a PyTorch-only env still works.
"""

# PyTorch first-class — always available.
from .care import CAREBackbone
from .cellpose import CellPoseBackbone
from .maskunet import MaskUNetBackbone
from .stardist import StarDistBackbone
from .unet import UNetBackbone

__all__ = [
    "CAREBackbone",
    "UNetBackbone",
    "MaskUNetBackbone",
    "StarDistBackbone",
    "CellPoseBackbone",
]

# Optional keras backbones — present only when [keras] extras are installed.
HAS_KERAS = False
try:
    HAS_KERAS = True
    __all__.extend(
        [
            "CAREBackboneKeras",
            "UNetBackboneKeras",
            "MaskUNetBackboneKeras",
            "StarDist2DBackboneKeras",
            "StarDist3DBackboneKeras",
        ]
    )
except ImportError:
    pass
