"""Hydra config schemas for the curvature-physics scripts."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CurvatureParameters:
    # IO
    file_type: str  # glob, e.g. "*.tif"

    # Geometry knobs (forwarded to compute_curvature / _timelapse)
    spacing: list[float]  # (dz, dy, dx) or (dy, dx)
    n_window: int
    stride: int
    geodesic: bool

    # Optional physics (set to null to disable each term)
    surface_tension: Optional[float]  # γ (N/m)
    bending_modulus: Optional[float]  # κ_b (J)
    spontaneous_curvature: float  # C₀ (1/length)
    saddle_splay_modulus: Optional[float]  # κ_G (J)

    # Distribution histogram
    field: str  # "kappa", "pressure", ...
    n_bins: int
    value_range: Optional[list[float]]  # (lo, hi), or null = auto

    # Timelapse axis
    timelapse: bool  # True → expects (T, *spatial)
    spatial_ndim: int  # 2 or 3 (per-frame)

    # Tracking (timelapse only — passed straight to link_labels_timelapse)
    max_link_distance: Optional[float]


@dataclass
class CurvatureDataPaths:
    base_data_dir: str
    label_dir: str  # per-file label TIFFs
    output_dir: str  # CSV + heatmap TIFFs go here


@dataclass
class CurvatureScenario:
    parameters: CurvatureParameters
    experiment_data_paths: CurvatureDataPaths
