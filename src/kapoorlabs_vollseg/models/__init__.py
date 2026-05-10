"""Layer 1 — singleton inference models.

Bare names (PyTorch / Lightning / careamics, plus PyTorch CellPose) are
always available. ``*Keras`` variants are only present when the
optional ``[keras]`` extra is installed.
"""

from .care import CAREDenoiser
from .cellpose import CellPoseSegmenter
from .maskunet import MaskUNetSegmenter
from .stardist import StarDistSegmenter
from .unet import UNetSegmenter

__all__ = [
    "CAREDenoiser",
    "UNetSegmenter",
    "MaskUNetSegmenter",
    "StarDistSegmenter",
    "CellPoseSegmenter",
]

HAS_KERAS = False
try:
    HAS_KERAS = True
    __all__.extend(
        [
            "CAREDenoiserKeras",
            "UNetSegmenterKeras",
            "MaskUNetSegmenterKeras",
            "StarDistSegmenterKeras",
        ]
    )
except ImportError:
    pass
