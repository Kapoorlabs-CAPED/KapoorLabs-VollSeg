"""Seed-pooling primitives: tiny geometric checks used during fusion.

Both classes answer the same shape of question — "is this point inside this
bounding box?" — but with opposite intent:

- :class:`SeedPool` returns ``True`` (include the candidate seed) when the
  point lies *outside* every existing instance box.
- :class:`UnetStarMask` returns ``True`` (include the StarDist instance into
  the U-Net mask) when the U-Net seed lies *outside* the StarDist box.

The fusion logic that uses them lives in :mod:`kapoorlabs_vollseg.fusion`.
"""

from collections.abc import Sequence


class SeedPool:
    """Test whether a candidate point falls outside a given bounding box."""

    def __init__(self, box: Sequence[int], point: Sequence[float]):
        self.box = box
        self.point = point
        self.ndim = len(point)

    def pooling(self) -> bool:
        # Include the seed iff it is *outside* the box on at least one axis.
        return any(self._outside_on_axis(p) for p in range(self.ndim))

    def _outside_on_axis(self, axis: int) -> bool:
        lo = self.box[axis]
        hi = self.box[axis + self.ndim]
        return not (lo <= self.point[axis] <= hi)


class UnetStarMask:
    """Test whether a U-Net seed falls outside a StarDist instance box."""

    def __init__(self, box: Sequence[int], point: Sequence[float]):
        self.box = box
        self.point = point
        self.ndim = len(point)

    def masking(self) -> bool:
        return any(self._outside_on_axis(p) for p in range(self.ndim))

    def semi_masking(self) -> bool:
        # Skip axis 0 (typically Z): only require XY separation.
        return any(self._outside_on_axis(p) for p in range(1, self.ndim))

    def _outside_on_axis(self, axis: int) -> bool:
        lo = self.box[axis]
        hi = self.box[axis + self.ndim]
        return not (lo <= self.point[axis] <= hi)
