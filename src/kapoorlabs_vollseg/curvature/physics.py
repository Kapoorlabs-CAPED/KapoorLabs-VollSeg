r"""Map a curvature profile to a mechanical force / pressure profile.

Two complementary models, gated on whether the user supplies the
material constants:

- :func:`young_laplace_pressure` — surface tension \gamma; output is the
  pressure differential ΔP across the curved interface
  (``ΔP = \gammaκ`` in 2D, ``ΔP = 2\gammaH`` in 3D where ``H`` is the mean
  curvature, equal to ``κ`` for an isotropic sphere fit).

- :func:`helfrich_bending_density` — bending modulus ``κ_b`` (and
  optional saddle-splay modulus ``κ_G`` for the Gaussian-curvature
  term); output is the bending energy density per unit area, useful
  when the user is modelling membrane mechanics in the Helfrich
  framework.

Both functions assume the curvature input is signed (positive when
the surface bulges outward, see ``CurvatureProfile.kappa``) and are
purely numpy.
"""

from __future__ import annotations

import numpy as np


def young_laplace_pressure(
    kappa: np.ndarray,
    *,
    surface_tension: float,
    ndim: int,
) -> np.ndarray:
    r"""Young-Laplace pressure differential ΔP across a curved surface.

    - 2D contour: ``ΔP = \gamma · κ``  (one principal curvature)
    - 3D surface: ``ΔP = 2\gamma · H``  where ``H = κ`` for a sphere fit
      (both principal curvatures equal ``1/r``, so mean curvature
      ``H = (k₁+k₂)/2 = 1/r = κ``).

    Parameters
    ----------
    kappa
        Signed curvature, in units of 1/length (whatever ``spacing``
        units were used when extracting the boundary).
    surface_tension
        \gamma, in force/length (N/m). The output is then in pressure
        units (N/m² = Pa).
    ndim
        2 or 3.
    """
    if ndim == 2:
        return surface_tension * np.asarray(kappa, dtype=np.float64)
    if ndim == 3:
        return 2.0 * surface_tension * np.asarray(kappa, dtype=np.float64)
    raise ValueError(f"ndim must be 2 or 3, got {ndim}")


def helfrich_bending_density(
    kappa: np.ndarray,
    *,
    bending_modulus: float,
    spontaneous_curvature: float = 0.0,
    saddle_splay_modulus: float | None = None,
) -> np.ndarray:
    """Helfrich bending-energy density (per unit area).

    For an isotropic sphere fit, both principal curvatures equal ``κ``,
    so ``H = κ`` and ``K = κ²``. The Helfrich free-energy density is

        ``f = κ_b · (2H - C₀)²  + κ_G · K``

    with ``κ_b`` the bending modulus, ``C₀`` the spontaneous curvature,
    and ``κ_G`` the saddle-splay (Gaussian) modulus.

    Parameters
    ----------
    kappa
        Signed curvature in 1/length.
    bending_modulus
        ``κ_b``, in energy units (J).
    spontaneous_curvature
        ``C₀``, in 1/length. Default 0 (a flat membrane is the energy
        minimum).
    saddle_splay_modulus
        ``κ_G``, optional. When supplied, the Gaussian-curvature term
        ``κ_G · κ²`` is added.
    """
    k = np.asarray(kappa, dtype=np.float64)
    f = bending_modulus * (2.0 * k - spontaneous_curvature) ** 2
    if saddle_splay_modulus is not None:
        f = f + saddle_splay_modulus * k * k
    return f
