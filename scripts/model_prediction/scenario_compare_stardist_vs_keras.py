"""Hydra schema for the StarDist-vs-keras per-frame comparison script.

Same shape as ``scenario_optimize_stardist_thresholds`` — re-uses
``train_data_paths`` so the winner model folder + experiment name come
from the same yaml the threshold optimiser ran against. Adds a
``compare_data_paths`` block for the input timelapse, the keras
reference labels, and the output CSV location.
"""

from dataclasses import dataclass


@dataclass
class CompareParams:
    # Inference runtime
    n_tiles: list[int]
    tile_overlap: float
    batch_size: int
    pmin: float
    pmax: float

    # Multi-GPU knobs forwarded to ``predict_timelapse``.
    devices: int  # -1 = all visible, N = first N, 1 = single GPU
    accelerator: str  # "auto" | "cuda" | "cpu"
    strategy: str  # "auto" | "ddp"

    # Which T-indices to score. ``subset_n_each`` reuses the sweep's
    # first/mid/last triple selection so the comparison lines up with
    # the sweep predictions on disk.
    subset_n_each: int

    # If True, every metric is recomputed even if the CSV exists.
    force: bool


@dataclass
class CompareDataPaths:
    # Input timelapse to predict on. Same TIFF the sweep ran against.
    input_dir: str
    input_pattern: str  # glob, e.g. "*.tif"

    # Keras reference labels folder. Must contain a TIFF with the same
    # basename as the input — that's the reference the sweep already
    # scored against.
    keras_dir: str

    # Output directory for the per-frame stats CSV. Each input TIFF
    # produces one CSV at ``out_dir/<stem>.compare.csv``.
    out_dir: str


@dataclass
class StarDistDataPaths:
    base_data_dir: str
    h5_file: str
    log_path: str
    experiment_name: str


@dataclass
class CompareStarDistVsKerasScenario:
    parameters: CompareParams
    train_data_paths: StarDistDataPaths
    compare_data_paths: CompareDataPaths
