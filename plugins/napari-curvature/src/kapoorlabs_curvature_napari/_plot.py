"""Matplotlib canvas embedded in a Qt widget.

Two plot modes:

- ``draw_profile`` — single 1D line plot: curvature (and optionally
  intensity on a second y-axis) vs. distance along the polyline at
  the current time-point.
- ``draw_kymograph`` — 2D heatmap (length × time) for the currently
  selected line + channel; used to preview what a record-to-disk
  would produce.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class ProfileCanvas(FigureCanvasQTAgg):
    """Qt-embeddable matplotlib canvas with two helper draw methods."""

    def __init__(self):
        self._fig = Figure(figsize=(5, 3), tight_layout=True)
        super().__init__(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._ax_right: Optional[object] = None

    # ----------------------------------------------------- single time-point

    def draw_profile(
        self,
        curvature: np.ndarray,
        intensity: Optional[np.ndarray] = None,
        *,
        title: str = "",
    ) -> None:
        """Plot κ (and intensity on a twin axis) vs. distance along a line."""
        self._ax.clear()
        if self._ax_right is not None:
            self._ax_right.remove()
            self._ax_right = None

        x = np.arange(len(curvature))
        self._ax.plot(x, curvature, color="tab:blue", label="curvature")
        self._ax.set_xlabel("distance along line (samples)")
        self._ax.set_ylabel("curvature", color="tab:blue")
        self._ax.tick_params(axis="y", labelcolor="tab:blue")

        if intensity is not None and len(intensity) == len(curvature):
            self._ax_right = self._ax.twinx()
            self._ax_right.plot(
                x,
                intensity,
                color="tab:orange",
                alpha=0.7,
                label="intensity",
            )
            self._ax_right.set_ylabel("intensity", color="tab:orange")
            self._ax_right.tick_params(axis="y", labelcolor="tab:orange")

        self._ax.set_title(title)
        self.draw_idle()

    # ----------------------------------------------------- kymograph

    def draw_kymograph(
        self,
        kymograph: np.ndarray,
        *,
        title: str = "",
        cmap: str = "viridis",
    ) -> None:
        """Heatmap of length (x) × time (y). ``kymograph`` is ``(T, L)``."""
        self._ax.clear()
        if self._ax_right is not None:
            self._ax_right.remove()
            self._ax_right = None
        im = self._ax.imshow(
            kymograph,
            aspect="auto",
            origin="lower",
            cmap=cmap,
        )
        self._fig.colorbar(im, ax=self._ax, fraction=0.05)
        self._ax.set_xlabel("distance along line (samples)")
        self._ax.set_ylabel("frame (t)")
        self._ax.set_title(title)
        self.draw_idle()
