"""Training harnesses — PyTorch first-class, keras legacy.

Bare-named: PyTorch Lightning trainers built on the careamics UNet.
``*Keras`` suffix: original csbdeep / stardist trainers, kept around for
existing pipelines.
"""

from .care import CARETrainer
from .care_keras import CARETrainerKeras
from .cellpose import CellPoseTrainer
from .maskunet import MaskUNetTrainer
from .smartseeds_keras import SmartSeedsKeras
from .stardist_keras import StarDistTrainerKeras
from .unet import UNetTrainer
from .unet_keras import UNetTrainerKeras

__all__ = [
    # PyTorch first-class
    "CARETrainer",
    "UNetTrainer",
    "MaskUNetTrainer",
    "CellPoseTrainer",
    # Keras legacy / no-pytorch-counterpart-yet
    "CARETrainerKeras",
    "UNetTrainerKeras",
    "StarDistTrainerKeras",
    "SmartSeedsKeras",
]
