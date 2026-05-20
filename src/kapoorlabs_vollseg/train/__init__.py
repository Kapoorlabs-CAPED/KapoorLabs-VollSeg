"""Training harnesses.

The unified :class:`~kapoorlabs_vollseg.training.TrainingPipeline` is
the supported entry point for CARE / U-Net / StarDist / MaskUNet
training — call its stackable ``setup_*`` methods directly. The old
per-task trainer façades (``CARETrainer`` / ``UNetTrainer`` /
``MaskUNetTrainer`` / ``StarDistTrainer``) have been removed.

What stays here:

* :class:`CellPoseTrainer` — CellPose has its own end-to-end training
  loop that doesn't fit the Lightning ``configure_optimizers`` shape
  the pipeline assumes.
* The legacy ``*Keras`` trainers (csbdeep stack) for already-trained
  ``.h5`` weights.
"""

from .cellpose import CellPoseTrainer

__all__ = ["CellPoseTrainer"]

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
