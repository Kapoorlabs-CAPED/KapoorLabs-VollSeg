"""Napari plugin for KapoorLabs curvature timelapse measurement.

Pairs a raw timelapse with a curvature timelapse (output of
:mod:`kapoorlabs_vollseg.curvature`) and lets the user draw arbitrary
line profiles, watch the curvature / intensity along each line update
as napari's T slider moves, and record kymographs (length × time) for
each line into TIFFs + CSV.
"""

from ._widget import CurvatureWidget

__all__ = ["CurvatureWidget"]
