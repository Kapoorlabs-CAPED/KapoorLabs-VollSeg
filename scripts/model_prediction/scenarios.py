"""Hydra config schemas for the segmentation-model prediction scripts.

One dataclass per model so each ``predict-<model>.py`` carries only
the knobs it actually consumes; everything they share lives in
``ExperimentDataPaths``.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentDataPaths:
    base_data_dir: str
    input_dir: str  # under base_data_dir
    output_dir: str  # under base_data_dir; created if missing
    log_path: str  # absolute path to the trained-model folder


@dataclass
class CarePredictParams:
    file_type: str
    n_tiles: list[int]
    tile_overlap: float
    batch_size: int


@dataclass
class RoiPredictParams:
    file_type: str
    n_tiles: list[int]
    tile_overlap: float
    batch_size: int
    min_size_mask: int  # remove_small_objects floor (px)


@dataclass
class UNetPredictParams:
    file_type: str
    n_tiles: list[int]
    tile_overlap: float
    batch_size: int
    min_size: int  # remove_small_objects floor (px)


@dataclass
class StarDistPredictParams:
    file_type: str
    n_tiles: list[int]
    tile_overlap: float
    batch_size: int
    n_rays: int  # fallback when rays.npy is missing from the folder
    prob_thresh: Optional[float]  # null → use the model default
    nms_thresh: Optional[float]  # null → use the model default


# Scenario wrappers Hydra uses for type-checking.
@dataclass
class CarePredictScenario:
    parameters: CarePredictParams
    experiment_data_paths: ExperimentDataPaths


@dataclass
class RoiPredictScenario:
    parameters: RoiPredictParams
    experiment_data_paths: ExperimentDataPaths


@dataclass
class UNetPredictScenario:
    parameters: UNetPredictParams
    experiment_data_paths: ExperimentDataPaths


@dataclass
class StarDistPredictScenario:
    parameters: StarDistPredictParams
    experiment_data_paths: ExperimentDataPaths
