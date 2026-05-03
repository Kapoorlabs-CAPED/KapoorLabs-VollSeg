"""Layer 1 — singleton inference models.

PyTorch / Lightning is the first-class backend; the bare-named classes
(``CAREDenoiser``, ``UNetSegmenter``, ``MaskUNetSegmenter``) wrap the
careamics UNet via the inlined :class:`vollseg._lightning.CareModule`.

The ``*Keras`` variants wrap csbdeep / stardist and remain available for
existing trained weights. ``StarDistSegmenterKeras`` has no PyTorch
counterpart yet — it remains the sole way to use StarDist.
``CellPoseSegmenter`` is PyTorch by way of the cellpose package.
"""

from .care import CAREDenoiser
from .care_keras import CAREDenoiserKeras
from .cellpose import CellPoseSegmenter
from .maskunet import MaskUNetSegmenter
from .maskunet_keras import MaskUNetSegmenterKeras
from .stardist_keras import StarDistSegmenterKeras
from .unet import UNetSegmenter
from .unet_keras import UNetSegmenterKeras

__all__ = [
    # PyTorch first-class
    "CAREDenoiser",
    "UNetSegmenter",
    "MaskUNetSegmenter",
    "CellPoseSegmenter",
    # Keras legacy / no-pytorch-counterpart-yet
    "CAREDenoiserKeras",
    "UNetSegmenterKeras",
    "MaskUNetSegmenterKeras",
    "StarDistSegmenterKeras",
]
