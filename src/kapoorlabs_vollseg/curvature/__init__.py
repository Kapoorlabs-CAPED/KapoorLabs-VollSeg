"""Segmentation-based curvature toolkit.

Given a label image, for each labeled region:

1. Extract its boundary (2D contour) or surface (3D triangle mesh).
2. Slide a window of ``n_window`` neighboring boundary points along the
   boundary with stride ``stride``.
3. Fit a **circle** (2D) or **sphere** (3D) to each window's points and
   record the local curvature κ = 1/r, signed by whether the fitted
   centre lies on the inside (convex) or outside (concave) of the
   surface.
4. Optionally map curvature to a **force / pressure profile** via
   Young-Laplace (``surface_tension`` supplied) or Helfrich bending
   (``bending_modulus`` supplied), or both.

Public API::

    from kapoorlabs_vollseg.curvature import compute_curvature

    profiles = compute_curvature(
        labels,
        spacing=(2.0, 0.6918, 0.6918),   # (dz, dy, dx) μm
        n_window=21, stride=5,
        surface_tension=1e-3,            # N/m — optional
        bending_modulus=2e-20,           # J — optional
    )
    for label_id, profile in profiles.items():
        ...                              # profile.kappa, .pressure, ...
"""

from .api import compute_curvature
from .boundary import (
    build_vertex_adjacency,
    extract_boundary_2d,
    extract_surface_3d,
)
from .distribution import CurvatureDistribution, compute_curvature_distribution
from .fit_2d import fit_circle_2d
from .fit_3d import fit_sphere_3d
from .physics import helfrich_bending_density, young_laplace_pressure
from .profile import CurvatureProfile
from .render import (
    process_label_folder,
    render_curvature_volume,
    save_curvature_tiffs,
)
from .timelapse import (
    CurvatureTimelapse,
    compute_curvature_timelapse,
    process_timelapse_folder,
    save_curvature_timelapse_tiffs,
)
from .tracking import (
    FEATURE_DEFAULT_WEIGHTS,
    available_features,
    link_labels_timelapse,
)
from .windows import bfs_geodesic_neighbors, euclidean_neighbors

__all__ = [
    "compute_curvature",
    "CurvatureProfile",
    "extract_boundary_2d",
    "extract_surface_3d",
    "build_vertex_adjacency",
    "fit_circle_2d",
    "fit_sphere_3d",
    "young_laplace_pressure",
    "helfrich_bending_density",
    "bfs_geodesic_neighbors",
    "euclidean_neighbors",
    "render_curvature_volume",
    "save_curvature_tiffs",
    "process_label_folder",
    # timelapse
    "link_labels_timelapse",
    "FEATURE_DEFAULT_WEIGHTS",
    "available_features",
    "CurvatureTimelapse",
    "compute_curvature_timelapse",
    "save_curvature_timelapse_tiffs",
    "process_timelapse_folder",
    # distribution
    "CurvatureDistribution",
    "compute_curvature_distribution",
]
