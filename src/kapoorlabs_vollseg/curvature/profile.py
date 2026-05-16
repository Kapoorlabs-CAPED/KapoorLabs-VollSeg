"""Per-region curvature profile container.

Fields are NumPy arrays of shape ``(W, ...)`` where ``W`` is the
number of sliding-window samples along the boundary or surface. The
optional ``pressure`` and ``bending_density`` columns are filled in by
:mod:`physics` when the relevant material constants are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CurvatureProfile:
    """One sliding-window curvature trace for a single labeled region."""

    label_id: int
    ndim: int  # 2 or 3

    # Geometry per window.
    centers: np.ndarray  # (W, ndim) boundary-point coords
    fit_centers: np.ndarray  # (W, ndim) fitted circle/sphere centres
    radii: np.ndarray  # (W,) fitted radius (length units)
    kappa: np.ndarray  # (W,) signed curvature (1/length)
    normals: np.ndarray  # (W, ndim) outward normals

    # Optional physics derived from kappa.
    surface_tension: Optional[float] = None
    pressure: Optional[np.ndarray] = None  # (W,) Young-Laplace ΔP

    bending_modulus: Optional[float] = None
    spontaneous_curvature: float = 0.0
    saddle_splay_modulus: Optional[float] = None
    bending_density: Optional[np.ndarray] = None  # (W,) Helfrich f

    # Free-form per-profile metadata (e.g. spacing, fit method).
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------ summaries

    @property
    def n_windows(self) -> int:
        return int(self.centers.shape[0])

    def summary(self) -> dict[str, float]:
        """Median / IQR statistics over the profile, useful for QC."""
        out = {
            "label_id": int(self.label_id),
            "n_windows": self.n_windows,
            "kappa_median": float(np.median(self.kappa)),
            "kappa_p25": float(np.percentile(self.kappa, 25)),
            "kappa_p75": float(np.percentile(self.kappa, 75)),
            "radius_median": float(np.median(self.radii)),
        }
        if self.pressure is not None:
            out["pressure_median"] = float(np.median(self.pressure))
        if self.bending_density is not None:
            out["bending_density_median"] = float(np.median(self.bending_density))
        return out

    def to_dict(self) -> dict[str, np.ndarray]:
        """Plain-Python representation suitable for pandas / H5 / JSON."""
        d: dict[str, np.ndarray] = {
            "centers": self.centers,
            "fit_centers": self.fit_centers,
            "radii": self.radii,
            "kappa": self.kappa,
            "normals": self.normals,
        }
        if self.pressure is not None:
            d["pressure"] = self.pressure
        if self.bending_density is not None:
            d["bending_density"] = self.bending_density
        return d
