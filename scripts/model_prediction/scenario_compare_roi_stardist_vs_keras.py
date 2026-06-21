"""Hydra schema for the ROI Mask-UNet → StarDist vs Keras comparison.

Same as ``CompareStarDistVsKerasScenario`` but adds a single
``mask_unet_log_path`` field on the train_data_paths block — the
folder of the trained ROI Mask-UNet that gates StarDist predictions
via :class:`ROIPipeline`. Useful for the early embryo timepoints
where most of the volume is empty space; the ROI mask restricts
percentile-normalisation and StarDist's peak detection to the
foreground crop.
"""

from dataclasses import dataclass

from scenario_compare_stardist_vs_keras import (
    CompareDataPaths,
    CompareParams,
)


@dataclass
class RoiStarDistDataPaths:
    base_data_dir: str
    h5_file: str
    log_path: str  # StarDist model folder
    experiment_name: str
    mask_unet_log_path: str  # ROI Mask-UNet folder (frozen across the comparison)


@dataclass
class CompareRoiStarDistVsKerasScenario:
    parameters: CompareParams
    train_data_paths: RoiStarDistDataPaths
    compare_data_paths: CompareDataPaths
