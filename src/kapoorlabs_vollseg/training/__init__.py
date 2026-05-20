"""Modular training-pipeline subpackage.

The :class:`TrainingPipeline` here is the stackable orchestrator every
training script in ``scripts/model_training/`` instantiates and drives
through ``setup_*`` calls — same shape as
``lightning_kietzmannlab.lightning_trainer.TrainingPipeline``.
"""

from .pipeline import TrainingPipeline

__all__ = ["TrainingPipeline"]
