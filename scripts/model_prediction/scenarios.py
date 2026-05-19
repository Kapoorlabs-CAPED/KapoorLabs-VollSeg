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


# Lightning multi-GPU knobs shared by every predict scenario. Set
# `devices > 1` + `strategy='ddp'` to shard the T axis across GPUs
# inside :func:`predict_timelapse`.
@dataclass
class _MultiGPU:
    devices: int = 1  # 1, -1 (all), or a positive count
    accelerator: str = "auto"  # "cuda" / "cpu" / "auto"
    strategy: str = "auto"  # "ddp" / "auto"


@dataclass
class CarePredictParams(_MultiGPU):
    file_type: str = "*.tif"
    n_tiles: list[int] = None
    tile_overlap: float = 0.125
    batch_size: int = 4


@dataclass
class RoiPredictParams(_MultiGPU):
    file_type: str = "*.tif"
    n_tiles: list[int] = None
    tile_overlap: float = 0.125
    batch_size: int = 4
    min_size_mask: int = 100  # remove_small_objects floor (px)


@dataclass
class UNetPredictParams(_MultiGPU):
    file_type: str = "*.tif"
    n_tiles: list[int] = None
    tile_overlap: float = 0.125
    batch_size: int = 4
    min_size: int = 10  # remove_small_objects floor (px)


@dataclass
class StarDistPredictParams(_MultiGPU):
    file_type: str = "*.tif"
    n_tiles: list[int] = None
    tile_overlap: float = 0.125
    batch_size: int = 4
    n_rays: int = 96  # fallback when rays.npy is missing
    prob_thresh: Optional[float] = None
    nms_thresh: Optional[float] = None


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
class ComboPredictParams(_MultiGPU):
    file_type: str = "*.tif"  # glob
    n_tiles: list[int] = None  # 3D, applied per-frame
    tile_overlap: float = 0.125
    batch_size: int = 4
    # StarDist runtime
    n_rays: int = 96
    prob_thresh: Optional[float] = None
    nms_thresh: Optional[float] = None
    # Pipeline shape
    seedpool: bool = False  # only honoured when stardist+unet both set
    # Postproc floors
    min_size_mask: int = 100
    min_size: int = 10


@dataclass
class ComboPredictScenario:
    parameters: ComboPredictParams
    experiment_data_paths: ComboExperimentPaths
