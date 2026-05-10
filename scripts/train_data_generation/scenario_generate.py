"""Hydra schema for unified SmartPatches H5 generation.

Same H5 trains both U-Net and StarDist — UNet reads ``raw + mask``,
StarDist reads ``raw + label`` (and derives prob/dist on the fly).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenParams:
    patch_shape: list[int]  # (Y X) for 2D, (Z Y X) for 3D
    val_files: int  # last N files reserved for validation
    lower_ratio_fore_to_back: float  # SmartPatches veto lower bound
    upper_ratio_fore_to_back: float  # SmartPatches veto upper bound
    erosion_iterations: int
    max_foreground_patches_per_image: float
    paste_augmentation: bool  # composite cells onto bg patches
    max_paste_patches_per_image: float
    seed: int
    overwrite: bool
    file_type: str  # glob, e.g. "*.tif"


@dataclass
class GenDataPaths:
    base_data_dir: str
    raw_dir: str
    label_dir: str  # integer instance masks (required)
    h5_file: str  # output H5 path (relative to base_data_dir)
    binary_mask_dir: Optional[str] = None  # optional pre-computed binary masks; if
    # set, written into /split/mask in the H5


@dataclass
class GenScenario:
    parameters: GenParams
    train_data_paths: GenDataPaths
