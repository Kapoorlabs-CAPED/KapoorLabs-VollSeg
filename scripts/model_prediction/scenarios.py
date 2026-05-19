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
    log_path: str  # local trained-model folder (used when hf_repo_id is null)
    hf_repo_id: Optional[str] = None  # e.g. "KapoorLabs/xenopus-stardist-pytorch"
    hf_model_dir: str = ""  # local cache root for HF; falls back to log_path


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


# -------------------------------------------------- combo predict


@dataclass
class ModelRef:
    """Where one role's model lives. Either fields-set means 'skip this role'."""

    log_path: str = ""  # local trained-model folder
    hf_repo_id: Optional[str] = None  # e.g. "KapoorLabs/xenopus-stardist-pytorch"
    hf_model_dir: str = ""  # local cache root for HF; falls back to log_path


@dataclass
class ComboExperimentPaths:
    base_data_dir: str
    input_dir: str
    output_dir: str
    stardist: ModelRef
    unet: ModelRef
    maskunet: ModelRef  # 2D Mask-UNet (ROI), MIP-on-3D in-singleton


@dataclass
class ComboPredictParams:
    file_type: str  # glob, e.g. "*.tif"
    n_tiles: list[int]  # 3D, applied per-frame
    tile_overlap: float
    batch_size: int
    # StarDist runtime
    n_rays: int  # fallback when rays.npy is missing
    prob_thresh: Optional[float]
    nms_thresh: Optional[float]
    # Pipeline shape
    seedpool: bool  # only honoured when both stardist+unet are set
    # ROI postproc — same as predict-roi.py
    min_size_mask: int
    # U-Net postproc — same as predict-unet.py
    min_size: int


@dataclass
class ComboPredictScenario:
    parameters: ComboPredictParams
    experiment_data_paths: ComboExperimentPaths
