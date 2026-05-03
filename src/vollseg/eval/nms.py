"""Non-max suppression on instance label images.

Two operations:

- :meth:`NMSLabel.suppress_overlapping` — drop instances whose bounding boxes
  overlap (IoU above threshold) or contain each other.
- :meth:`NMSLabel.suppress_thin_z` — drop 3D instances thinner than
  ``z_thresh`` slices.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from skimage import measure
from skimage.util import map_array


class NMSLabel:
    """Suppress instances in a label image by bounding-box NMS.

    Parameters
    ----------
    image
        Integer label image (2D or 3D).
    nms_thresh
        IoU threshold above which an overlapping pair is collapsed.
    z_thresh
        Minimum Z-extent (in slices) for a 3D instance to survive
        :meth:`suppress_thin_z`.
    """

    def __init__(self, image: np.ndarray, nms_thresh: float, z_thresh: int = 1):
        self.image = image
        self.nms_thresh = float(nms_thresh)
        self.z_thresh = int(z_thresh)
        self.ndim = image.ndim

    # ------------------------------------------------------------------ NMS
    def suppress_overlapping(self) -> np.ndarray:
        """Collapse pairs of bounding boxes whose IoU >= ``nms_thresh``."""
        props = measure.regionprops(self.image)
        bboxes = [p.bbox for p in props]
        labels = [p.label for p in props]

        remap: dict = {}
        for last in range(len(labels) - 1, 0, -1):
            for pos in range(last):
                self._merge_if_overlapping(
                    bboxes[last], bboxes[pos], labels[last], labels[pos], remap
                )

        out = self.image.copy()
        for src, dst in remap.items():
            out = np.where(out == src, dst, out)
        return out

    def _merge_if_overlapping(self, a, b, la, lb, remap):
        if self.ndim == 2:
            a_inside_b = b[0] <= a[0] and b[2] >= a[2] and b[1] <= a[1] and b[3] >= a[3]
            b_inside_a = a[0] <= b[0] and a[2] >= b[2] and a[1] <= b[1] and a[3] >= b[3]
            if b_inside_a:
                remap[lb] = la
                return
            if a_inside_b:
                remap[la] = lb
                return
            if _iou_2d(a, b) >= self.nms_thresh:
                remap[la] = lb
        elif self.ndim == 3:
            a_inside_b = (
                b[1] <= a[1] and b[4] >= a[4] and b[2] <= a[2] and b[5] >= a[5]
            )
            b_inside_a = (
                a[1] <= b[1] and a[4] >= b[4] and a[2] <= b[2] and a[5] >= b[5]
            )
            if b_inside_a:
                remap[lb] = la
                return
            if a_inside_b:
                remap[la] = lb
                return
            if _iou_3d(a, b) >= self.nms_thresh:
                remap[la] = lb

    # ------------------------------------------------------------ thin-Z
    def suppress_thin_z(self) -> np.ndarray:
        """Drop 3D instances whose Z-extent is smaller than ``z_thresh``."""
        if self.ndim != 3:
            return self.image

        props = measure.regionprops(self.image)
        src: List[int] = []
        dst: List[int] = []
        for p in props:
            z_extent = abs(p.bbox[0] - p.bbox[3])
            src.append(p.label)
            dst.append(0 if z_extent < self.z_thresh else p.label)

        if not src:
            return self.image
        return map_array(self.image, np.asarray(src), np.asarray(dst))


# ---------------------------------------------------------------- IoU helpers

def _iou_2d(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    aA = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    aB = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / float(aA + aB - inter)


def _iou_3d(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    zA, yA, xA = max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2])
    zB, yB, xB = min(a[3], b[3]), min(a[4], b[4]), min(a[5], b[5])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1) * max(0, zB - zA + 1)
    aA = (a[3] - a[0] + 1) * (a[4] - a[1] + 1) * (a[5] - a[2] + 1)
    aB = (b[3] - b[0] + 1) * (b[4] - b[1] + 1) * (b[5] - b[2] + 1)
    return inter / float(aA + aB - inter)
