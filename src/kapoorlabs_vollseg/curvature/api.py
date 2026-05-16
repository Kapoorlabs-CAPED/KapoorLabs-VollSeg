"""Top-level orchestrator: label image → per-region :class:`CurvatureProfile`.

Glues together boundary extraction (``boundary.py``), sliding-window
neighbour selection (``windows.py``), circle/sphere fitting
(``fit_2d.py`` / ``fit_3d.py``), and the optional physics mapping
(``physics.py``).
"""

from __future__ import annotations

from typing import Optional
from collections.abc import Iterable

import numpy as np

from .boundary import (
    build_vertex_adjacency,
    extract_boundary_2d,
    extract_surface_3d,
)
from .fit_2d import fit_circle_2d
from .fit_3d import fit_sphere_3d
from .physics import helfrich_bending_density, young_laplace_pressure
from .profile import CurvatureProfile
from .windows import slide_windows_2d, slide_windows_3d


def compute_curvature(
    labels: np.ndarray,
    *,
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
    n_window: int = 21,
    stride: int = 5,
    label_ids: Optional[Iterable[int]] = None,
    # 3D neighbour-selection knobs
    geodesic: bool = True,
    geodesic_method: str = "bfs",
    # Optional physics knobs
    surface_tension: Optional[float] = None,
    bending_modulus: Optional[float] = None,
    spontaneous_curvature: float = 0.0,
    saddle_splay_modulus: Optional[float] = None,
) -> dict[int, CurvatureProfile]:
    """Compute one :class:`CurvatureProfile` per labeled region.

    Parameters
    ----------
    labels
        Integer label image — 2D ``(Y, X)`` or 3D ``(Z, Y, X)``.
        Background must be 0.
    spacing
        Per-axis physical voxel size. 2 entries for 2D ``(dy, dx)``,
        3 for 3D ``(dz, dy, dx)``. Curvatures and radii will be in
        the *inverse* of these units (so pass μm to get 1/μm).
    n_window, stride
        Sliding-window length and step. ``n_window`` must be ≥ 3 (2D) /
        ≥ 4 (3D); ``stride`` ≥ 1.
    label_ids
        Restrict to specific labels. ``None`` means all non-zero
        labels present in ``labels``.
    geodesic, geodesic_method
        3D only. When ``geodesic=True`` (default), window neighbours
        are selected along the mesh; ``geodesic_method`` is ``"bfs"``
        (hop-count, fast) or ``"dijkstra"`` (edge-length-weighted,
        slower but exact).
    surface_tension
        If supplied, Young-Laplace pressure ``ΔP = γκ`` (2D) /
        ``2γH`` (3D) is added to each profile.
    bending_modulus, spontaneous_curvature, saddle_splay_modulus
        If ``bending_modulus`` is supplied, the Helfrich bending-energy
        density is added.

    Returns
    -------
    dict[int, CurvatureProfile]
        One profile per processed label. Labels with no extractable
        boundary / too few points are silently dropped.
    """
    if labels.ndim not in (2, 3):
        raise ValueError(f"labels must be 2D or 3D, got ndim={labels.ndim}")
    spacing = tuple(spacing[: labels.ndim])
    if len(spacing) != labels.ndim:
        raise ValueError(f"spacing must have {labels.ndim} entries, got {len(spacing)}")

    if label_ids is None:
        uniq = np.unique(labels)
        label_ids = [int(v) for v in uniq if v != 0]

    profiles: dict[int, CurvatureProfile] = {}
    for lid in label_ids:
        profile = (
            _compute_region_2d(
                labels,
                int(lid),
                spacing,
                n_window,
                stride,
                surface_tension,
                bending_modulus,
                spontaneous_curvature,
                saddle_splay_modulus,
            )
            if labels.ndim == 2
            else _compute_region_3d(
                labels,
                int(lid),
                spacing,
                n_window,
                stride,
                geodesic,
                geodesic_method,
                surface_tension,
                bending_modulus,
                spontaneous_curvature,
                saddle_splay_modulus,
            )
        )
        if profile is not None:
            profiles[int(lid)] = profile
    return profiles


# =========================================================== 2D per-region


def _compute_region_2d(
    labels,
    label_id,
    spacing,
    n_window,
    stride,
    surface_tension,
    bending_modulus,
    c0,
    kG,
) -> Optional[CurvatureProfile]:
    contours = extract_boundary_2d(labels, label_id, spacing=spacing)
    if not contours:
        return None

    centres, fit_centres, radii, kappas, normals = [], [], [], [], []
    for contour in contours:
        if len(contour) < n_window:
            continue
        for centre_idx, win_idx in slide_windows_2d(
            contour,
            n_window=n_window,
            stride=stride,
        ):
            pts = contour[win_idx]
            fc, r = fit_circle_2d(pts)
            if r <= 0:
                continue
            centre_pt = contour[centre_idx]
            # Tangent ≈ vector between window endpoints.
            tangent = contour[win_idx[-1]] - contour[win_idx[0]]
            tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
            # Normal is the in-plane perpendicular of the tangent
            # (rotate by +90°); sign-flip below if needed.
            normal = np.array([-tangent[1], tangent[0]])
            # Convex (positive κ) when the fitted centre is on the
            # *inside* — i.e. the radius vector points opposite to the
            # outward normal.
            radius_vec = fc - centre_pt
            sign = -1.0 if np.dot(radius_vec, normal) > 0 else +1.0
            centres.append(centre_pt)
            fit_centres.append(fc)
            radii.append(r)
            kappas.append(sign / r)
            normals.append(normal if sign > 0 else -normal)

    if not centres:
        return None
    return _wrap_profile(
        label_id,
        ndim=2,
        centres=centres,
        fit_centres=fit_centres,
        radii=radii,
        kappas=kappas,
        normals=normals,
        surface_tension=surface_tension,
        bending_modulus=bending_modulus,
        c0=c0,
        kG=kG,
        metadata={
            "spacing": spacing,
            "n_window": n_window,
            "stride": stride,
            "n_contours": len(contours),
        },
    )


# =========================================================== 3D per-region


def _compute_region_3d(
    labels,
    label_id,
    spacing,
    n_window,
    stride,
    geodesic,
    geodesic_method,
    surface_tension,
    bending_modulus,
    c0,
    kG,
) -> Optional[CurvatureProfile]:
    surface = extract_surface_3d(labels, label_id, spacing=spacing)
    if surface is None:
        return None
    verts, faces, vert_normals = surface
    adjacency = build_vertex_adjacency(faces, len(verts))

    centres, fit_centres, radii, kappas, normals = [], [], [], [], []
    for centre_idx, win_idx in slide_windows_3d(
        verts,
        adjacency,
        n_window=n_window,
        stride=stride,
        geodesic=geodesic,
        geodesic_method=geodesic_method,
    ):
        pts = verts[win_idx]
        fc, r = fit_sphere_3d(pts)
        if r <= 0:
            continue
        centre_pt = verts[centre_idx]
        normal = vert_normals[centre_idx]
        radius_vec = fc - centre_pt
        sign = -1.0 if np.dot(radius_vec, normal) > 0 else +1.0
        centres.append(centre_pt)
        fit_centres.append(fc)
        radii.append(r)
        kappas.append(sign / r)
        normals.append(normal)

    if not centres:
        return None
    return _wrap_profile(
        label_id,
        ndim=3,
        centres=centres,
        fit_centres=fit_centres,
        radii=radii,
        kappas=kappas,
        normals=normals,
        surface_tension=surface_tension,
        bending_modulus=bending_modulus,
        c0=c0,
        kG=kG,
        metadata={
            "spacing": spacing,
            "n_window": n_window,
            "stride": stride,
            "geodesic": geodesic,
            "geodesic_method": geodesic_method,
            "n_vertices": int(len(verts)),
            "n_faces": int(len(faces)),
        },
    )


# =========================================================== shared


def _wrap_profile(
    label_id,
    *,
    ndim,
    centres,
    fit_centres,
    radii,
    kappas,
    normals,
    surface_tension,
    bending_modulus,
    c0,
    kG,
    metadata,
) -> CurvatureProfile:
    kappa_arr = np.asarray(kappas, dtype=np.float64)
    profile = CurvatureProfile(
        label_id=int(label_id),
        ndim=int(ndim),
        centers=np.asarray(centres, dtype=np.float64),
        fit_centers=np.asarray(fit_centres, dtype=np.float64),
        radii=np.asarray(radii, dtype=np.float64),
        kappa=kappa_arr,
        normals=np.asarray(normals, dtype=np.float64),
        metadata=metadata,
    )
    if surface_tension is not None:
        profile.surface_tension = float(surface_tension)
        profile.pressure = young_laplace_pressure(
            kappa_arr,
            surface_tension=float(surface_tension),
            ndim=ndim,
        )
    if bending_modulus is not None:
        profile.bending_modulus = float(bending_modulus)
        profile.spontaneous_curvature = float(c0)
        if kG is not None:
            profile.saddle_splay_modulus = float(kG)
        profile.bending_density = helfrich_bending_density(
            kappa_arr,
            bending_modulus=float(bending_modulus),
            spontaneous_curvature=float(c0),
            saddle_splay_modulus=kG,
        )
    return profile
