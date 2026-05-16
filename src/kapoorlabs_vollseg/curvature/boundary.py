"""Extract boundary contours (2D) or triangulated surfaces (3D) per label.

Both routines respect anisotropic voxel ``spacing`` so downstream
curvatures end up in physical units (1/μm) rather than 1/voxel.
"""

from __future__ import annotations

import numpy as np
from skimage.measure import find_contours, marching_cubes


def extract_boundary_2d(
    labels: np.ndarray,
    label_id: int,
    *,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> list[np.ndarray]:
    """Return one or more closed sub-pixel contours for ``label_id``.

    Parameters
    ----------
    labels
        ``(Y, X)`` integer label image.
    label_id
        The label to extract.
    spacing
        ``(dy, dx)`` voxel size. Contour coordinates are rescaled so
        downstream distances are in physical units.

    Returns
    -------
    list of (P, 2) arrays
        Each array is one closed contour, ordered along the boundary.
        Coordinate order is ``(y, x)``.
    """
    if labels.ndim != 2:
        raise ValueError(
            f"extract_boundary_2d expects 2D labels, got ndim={labels.ndim}"
        )
    mask = (labels == label_id).astype(np.uint8)
    if mask.sum() == 0:
        return []
    contours = find_contours(mask, level=0.5)
    spacing_arr = np.asarray(spacing, dtype=np.float32)
    return [c * spacing_arr for c in contours if len(c) >= 3]


def extract_surface_3d(
    labels: np.ndarray,
    label_id: int,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    step_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Run marching cubes on ``label_id`` and return (vertices, faces, normals).

    Returns ``None`` if the label has no voxels or the mesh is degenerate.
    Vertices are in physical units. Normals are per-vertex outward
    normals from marching cubes' analytic gradient.
    """
    if labels.ndim != 3:
        raise ValueError(
            f"extract_surface_3d expects 3D labels, got ndim={labels.ndim}"
        )
    mask = (labels == label_id).astype(np.uint8)
    if mask.sum() == 0:
        return None
    try:
        verts, faces, normals, _ = marching_cubes(
            mask,
            level=0.5,
            spacing=spacing,
            step_size=step_size,
            allow_degenerate=False,
        )
    except (ValueError, RuntimeError):
        return None
    if verts.shape[0] < 4 or faces.shape[0] < 1:
        return None
    return verts, faces, normals


def build_vertex_adjacency(
    faces: np.ndarray,
    n_vertices: int,
) -> list[set[int]]:
    """Build the vertex-vertex adjacency graph from a triangle list.

    Two vertices are adjacent iff they share a triangle. Used by
    :func:`bfs_geodesic_neighbors` to compute mesh-aware neighborhoods.
    """
    adjacency: list[set[int]] = [set() for _ in range(n_vertices)]
    for a, b, c in faces:
        adjacency[a].update((int(b), int(c)))
        adjacency[b].update((int(a), int(c)))
        adjacency[c].update((int(a), int(b)))
    return adjacency
