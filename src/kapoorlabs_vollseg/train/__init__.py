"""Training harnesses — PyTorch first-class, keras optional.

Bare names are always available. ``*Keras`` trainers require the
optional ``[keras]`` extra.
"""

from .care import CARETrainer
from .cellpose import CellPoseTrainer
from .maskunet import MaskUNetTrainer
from .stardist import StarDistTrainer
from .unet import UNetTrainer

__all__ = [
    "CARETrainer",
    "UNetTrainer",
    "MaskUNetTrainer",
    "StarDistTrainer",
    "CellPoseTrainer",
]

HAS_KERAS = False
try:
    HAS_KERAS = True
    __all__.extend(
        [
            "CARETrainerKeras",
            "UNetTrainerKeras",
            "StarDistTrainerKeras",
        ]
    )
except ImportError:
    pass
