"""Hydra schema for U-Net Lightning training."""

from dataclasses import dataclass


@dataclass
class UNetTrainParams:
    # UNet architecture
    conv_dims: int
    in_channels: int
    num_classes: int
    unet_depth: int
    num_channels_init: int
    use_batch_norm: bool

    # Training
    epochs: int
    batch_size: int
    learning_rate: float
    num_workers: int
    devices: int
    accelerator: str
    train_precision: str
    strategy: str
    gradient_clip_val: float
    gradient_clip_algorithm: str
    slurm_auto_requeue: bool

    # Augmentation knobs
    pmin: float
    pmax: float
    augment: bool
    gaussian_noise_std: float

    # Inference defaults stored on the module
    n_tiles: list[int]
    tile_overlap: float


@dataclass
class UNetDataPaths:
    base_data_dir: str
    h5_file: str  # input H5 (relative to base_data_dir)
    log_path: str  # output checkpoints + logs
    experiment_name: str


@dataclass
class UNetTrainScenario:
    parameters: UNetTrainParams
    train_data_paths: UNetDataPaths
