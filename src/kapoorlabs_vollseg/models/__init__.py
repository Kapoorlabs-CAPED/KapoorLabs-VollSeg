"""Layer 1 — singleton inference models.

PyTorch / Lightning is the first-class backend; the bare-named classes
(``CAREDenoiser``, ``UNetSegmenter``, ``MaskUNetSegmenter``,
``StarDistSegmenter``) wrap careamics-based PyTorch backbones.
``CellPoseSegmenter`` is PyTorch via the cellpose package.

The ``*Keras`` variants wrap csbdeep / stardist and remain available for
existing trained weights.
"""

from .care import CAREDenoiser
from .care_keras import CAREDenoiserKeras
from .cellpose import CellPoseSegmenter
from .maskunet import MaskUNetSegmenter
from .maskunet_keras import MaskUNetSegmenterKeras
from .stardist import StarDistSegmenter
from .stardist_keras import StarDistSegmenterKeras
from .unet import UNetSegmenter
from .unet_keras import UNetSegmenterKeras

__all__ = [
    # PyTorch first-class
    "CAREDenoiser",
    "UNetSegmenter",
    "MaskUNetSegmenter",
    "StarDistSegmenter",
    "CellPoseSegmenter",
    # Keras legacy
    "CAREDenoiserKeras",
    "UNetSegmenterKeras",
    "MaskUNetSegmenterKeras",
    "StarDistSegmenterKeras",
]
