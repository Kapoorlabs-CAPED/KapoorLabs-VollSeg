"""PyTorch StarDist — clean rewrite of the keras implementation.

Architecturally, StarDist is a U-Net that produces ``1 + n_rays`` output
channels:

- Channel 0 — per-pixel object probability (sigmoid in inference).
- Channels 1..n_rays — per-pixel distance to the object boundary along
  each of ``n_rays`` ray directions.

This package builds that on top of the careamics U-Net so the same
backbone serves CARE / U-Net / MaskUNet / StarDist. Training is via
PyTorch Lightning with H5-backed paired patches.

The current scope is the modeling + label-encoding foundation:

- :func:`rays_2d` / :func:`rays_3d_golden_spiral` — ray geometry
- :func:`compute_distance_map` — turn a label image into the
  ``(n_rays, ...)`` distance target
- :class:`StarDistUNet` — UNet with prob + dist heads
- :func:`stardist_loss` — composite BCE + masked-L1

Training data generation, the Lightning module, and inference (NMS +
polygon decoding) land in subsequent commits.
"""

from .dataset import StarDistH5Dataset, stardist_collate
from .distance import compute_distance_map, foreground_probability_map
from .inference import StarDistResult, predict_volume
from .lightning_module import StarDistModule
from .losses import dist_loss, prob_loss, stardist_loss
from .model import StarDistUNet, split_outputs
from .rays import compute_faces, rays_2d, rays_3d_golden_spiral
from .transforms import (
    Compose,
    InputGaussianNoise,
    InputPercentileNormalize,
    RandomFlip,
    RandomRot90,
)

__all__ = [
    # rays
    "rays_2d",
    "rays_3d_golden_spiral",
    "compute_faces",
    # label encoding
    "compute_distance_map",
    "foreground_probability_map",
    # model
    "StarDistUNet",
    "split_outputs",
    # losses
    "stardist_loss",
    "prob_loss",
    "dist_loss",
    # dataset
    "StarDistH5Dataset",
    "stardist_collate",
    # training
    "StarDistModule",
    # inference
    "predict_volume",
    "StarDistResult",
    # augmentation
    "Compose",
    "InputGaussianNoise",
    "InputPercentileNormalize",
    "RandomFlip",
    "RandomRot90",
]
