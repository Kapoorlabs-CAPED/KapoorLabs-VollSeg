"""Inlined PyTorch-Lightning support — no runtime dep on kapoorlabs-lightning.

Mirrors just enough of the kapoorlabs-lightning CARE module shape that
loading checkpoints trained with that package "just works", while keeping
this repo independent. Code style and key class names match upstream so
porting future trainers stays mechanical.

This subpackage also hosts the kietzmann-style optimizer + scheduler
classes and the string-keyed registry that :class:`TrainingPipeline`
uses to resolve ``optimizer:`` / ``scheduler:`` yaml entries.
"""

from . import optimizers, schedulers
from .base_module import BaseModule
from .dataset import CarePredictionDataset, compute_tile_shape
from .registry import (
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    get_optimizer_factory,
    get_scheduler_factory,
)
from .stitch import stitch_tiles
from .transforms import PercentileNormalize, ToFloat32
from .unet_dataset import H5UNetDataset, unet_collate

__all__ = [
    "BaseModule",
    "CarePredictionDataset",
    "compute_tile_shape",
    "stitch_tiles",
    "PercentileNormalize",
    "ToFloat32",
    "H5UNetDataset",
    "unet_collate",
    "optimizers",
    "schedulers",
    "OPTIMIZER_REGISTRY",
    "SCHEDULER_REGISTRY",
    "get_optimizer_factory",
    "get_scheduler_factory",
]
