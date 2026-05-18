"""Hydra schema for StarDist Lightning training."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StarDistTrainParams:
    # UNet trunk architecture
    conv_dims: int
    in_channels: int
    unet_depth: int
    num_channels_init: int
    use_batch_norm: bool

    # StarDist-specific
    n_rays: int
    anisotropy: Optional[list[float]]  # 3D only, e.g. [2.0, 1.0, 1.0]
    loss_lam: float

    # Training
    epochs: int
    batch_size: int
    learning_rate: float
    num_workers: int
    devices: int
    accelerator: str
    train_precision: str
    strategy: str

    slurm_auto_requeue: bool

    # Augmentation
    pmin: float
    pmax: float
    augment: bool
    gaussian_noise_std: float

    # Inference defaults
    n_tiles: list[int]
    tile_overlap: float


@dataclass
class StarDistDataPaths:
    base_data_dir: str
    h5_file: str
    log_path: str
    experiment_name: str


@dataclass
class StarDistTrainScenario:
    parameters: StarDistTrainParams
    train_data_paths: StarDistDataPaths
