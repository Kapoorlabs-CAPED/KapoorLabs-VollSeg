"""Hydra schema for the StarDist threshold-optimisation script."""

from dataclasses import dataclass


@dataclass
class OptimizeThresholdsParams:
    # UNet trunk knobs — needed to rebuild the backbone via from_folder().
    conv_dims: int
    in_channels: int

    # StarDist knob — fallback if rays.npy is missing in log_path.
    n_rays: int

    # Search grid
    nms_threshs: list[float]
    iou_threshs: list[float]
    measure: str  # "accuracy" / "precision" / "recall" / "f1"

    # Inference runtime
    n_tiles: list[int]

    # H5 split + how many samples to use (-1 = all).
    split: str
    max_samples: int

    # Normalisation
    normalize_inputs: bool
    norm_axes: list[int]


@dataclass
class StarDistDataPaths:
    base_data_dir: str
    h5_file: str
    log_path: str
    experiment_name: str


@dataclass
class OptimizeThresholdsScenario:
    parameters: OptimizeThresholdsParams
    train_data_paths: StarDistDataPaths
