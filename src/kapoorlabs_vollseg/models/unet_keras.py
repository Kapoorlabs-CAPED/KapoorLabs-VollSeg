"""U-Net singleton (keras / csbdeep) — legacy.

New code should use :class:`kapoorlabs_vollseg.UNetSegmenter` (PyTorch + Lightning).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.filters import threshold_multiotsu
from skimage.morphology import label as cc_label
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential

from .._backbones.unet_keras import UNetBackboneKeras
from ..pipelines.base import Result, infer_axes


class UNetSegmenterKeras:
    """Run a csbdeep U-Net and turn its probability map into a binary mask + CC labels."""

    def __init__(
        self,
        backbone: UNetBackboneKeras,
        *,
        min_size: int = 10,
        morph_iterations: int = 0,
    ):
        self.backbone = backbone
        self.min_size = min_size
        self.morph_iterations = morph_iterations

    @classmethod
    def from_pretrained(cls, name_or_alias: str, **kwargs) -> UNetSegmenterKeras:
        from ..pretrained import get_model_instance

        return cls(get_model_instance(UNetBackboneKeras, name_or_alias), **kwargs)

    def predict(
        self,
        image: np.ndarray,
        *,
        axes: Optional[str] = None,
        n_tiles: Optional[tuple] = None,
        **_ignored,
    ) -> Result:
        if axes is None:
            axes = infer_axes(image)
        prob = self.backbone.predict(
            image.astype("float32"), axes=axes, n_tiles=n_tiles
        )

        try:
            thresholds = threshold_multiotsu(prob, classes=2)
            binary = np.digitize(prob, bins=thresholds) > 0
        except ValueError:
            binary = prob > 0

        if self.morph_iterations > 0 and binary.ndim == 3:
            for z in range(binary.shape[0]):
                binary[z] = binary_dilation(binary[z], iterations=self.morph_iterations)
                binary[z] = binary_erosion(binary[z], iterations=self.morph_iterations)

        labels = cc_label(binary)
        if self.min_size > 0:
            labels = remove_small_objects(
                labels.astype(np.int64), min_size=self.min_size
            )
        labels = relabel_sequential(labels.astype(np.uint32))[0]

        return Result(labels=labels, semantic=binary, probability=prob)
