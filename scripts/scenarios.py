"""Hydra config schemas for the segmentation scripts.

Trimmed-down version of CopenhagenWorkflow's ``scenario_segment_star_cellpose.py``
— only the fields the new vollseg-based scripts actually consume. Add
fields here when a new script needs them; don't add fields preemptively.
"""

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class Parameters:
    # IO / dispatch
    file_type: str           # glob, e.g. "*.tif"
    axes: str                # csbdeep / stardist axes string
    n_tiles: List[int]       # per-axis tiling for predict()
    channel_nuclei: int
    channel_membrane: int

    # Pipeline assembly (nuclei side)
    use_roi_unet: bool
    use_seedpool: bool
    use_care_denoise: bool   # apply CARE denoise before nuclei seg

    # Sizing / thresholds
    min_size: int
    min_size_mask: int
    max_size: int
    prob_thresh: Optional[float]
    nms_thresh: Optional[float]

    # CellPose runtime
    cellpose_diameter: float
    cellpose_flow_threshold: float
    cellpose_cellprob_threshold: float
    cellpose_stitch_threshold: float
    cellpose_anisotropy: Optional[float]
    cellpose_do_3d: bool
    cellpose_gpu: bool
    cellpose_channels: List[int]      # [cyto, nuclei] per CellPose convention
    cellpose_bsize: int

    # PyTorch backbone architecture (must match training-time values).
    # Used by CAREBackbone.from_checkpoint / UNetBackbone.from_checkpoint to
    # rebuild the careamics UNet before loading the .ckpt weights.
    pt_conv_dims: int
    pt_in_channels: int
    pt_num_classes: int
    pt_unet_depth: int
    pt_num_channels_init: int
    pt_use_batch_norm: bool
    pt_tile_overlap: float

    # Output naming
    save_name_prefix: str    # prefix for per-frame output files


@dataclass
class ModelPaths:
    base_dir: str
    star_model_dir: str
    unet_model_dir: str
    roi_model_dir: str
    care_model_dir: str
    cellpose_model_dir: str
    star_nuclei_model_name: str
    unet_nuclei_model_name: str
    roi_nuclei_model_name: str
    care_membrane_model_name: str
    # Lightning .ckpt files for the PyTorch backbones — set to null to keep
    # using the keras .h5 path under {care|unet|roi}_model_dir/{name}.
    care_membrane_checkpoint: Optional[str] = None
    unet_nuclei_checkpoint: Optional[str] = None
    roi_nuclei_checkpoint: Optional[str] = None
    cellpose_membrane_model_name: Optional[str] = None    # local cellpose checkpoint
    cellpose_membrane_model_type: Optional[str] = None    # built-in (e.g. cyto3)


@dataclass
class ExperimentDataPaths:
    base_directory: str
    timelapse_nuclei_directory: str
    timelapse_membrane_directory: str
    timelapse_seg_nuclei_directory: str
    timelapse_seg_membrane_directory: str
    timelapse_seg_vollcell_directory: str
    membrane_enhanced_directory: str
    voxel_size_xyz: List[float]
    metrics_ground_truth_directory: Optional[str] = None
    metrics_results_directory: Optional[str] = None


@dataclass
class SegmentScenario:
    parameters: Parameters
    model_paths: ModelPaths
    experiment_data_paths: ExperimentDataPaths
