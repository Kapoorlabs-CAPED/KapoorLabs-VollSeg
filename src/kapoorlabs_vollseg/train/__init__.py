"""Training harnesses — PyTorch first-class, keras legacy.

Bare-named: PyTorch Lightning trainers built on the careamics UNet (and
the StarDist UNet on top of it).
``*Keras`` suffix: original csbdeep / stardist trainers, kept around
for existing pipelines.
"""

from .care import CARETrainer
from .care_keras import CARETrainerKeras
from .cellpose import CellPoseTrainer
from .maskunet import MaskUNetTrainer
from .stardist import StarDistTrainer
from .stardist_keras import StarDistTrainerKeras
from .unet import UNetTrainer
from .unet_keras import UNetTrainerKeras

__all__ = [
    # PyTorch first-class
    "CARETrainer",
    "UNetTrainer",
    "MaskUNetTrainer",
    "StarDistTrainer",
    "CellPoseTrainer",
    # Keras legacy
    "CARETrainerKeras",
    "UNetTrainerKeras",
    "StarDistTrainerKeras",
]
