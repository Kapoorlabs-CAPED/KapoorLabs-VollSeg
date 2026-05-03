"""U-Net semantic segmentation singleton — first-class PyTorch implementation.

Same tiled inference path as :class:`CAREDenoiser`; the difference is in
post-processing — sigmoid → multi-Otsu → connected-components → size
filter — to produce the ``Result.semantic`` / ``Result.labels``
shape that the Layer 2 composites expect.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.filters import threshold_multiotsu
from skimage.morphology import label as cc_label
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential
from torch.utils.data import DataLoader

from .._backbones.unet import UNetBackbone
from .._lightning.dataset import CarePredictionDataset, compute_tile_shape
from .._lightning.stitch import stitch_tiles
from .._lightning.transforms import PercentileNormalize
from ..pipelines.base import Result


class UNetSegmenter:
    """Run a PyTorch U-Net to produce a binary mask + CC instance labels.

    Parameters
    ----------
    backbone
        A :class:`UNetBackbone` with weights loaded.
    n_tiles, tile_overlap, batch_size, num_workers, pmin, pmax, device
        Same meanings as :class:`CAREDenoiser`.
    min_size
        Drop connected components smaller than this many voxels.
    morph_iterations
        Iterations of dilation-then-erosion (per Z-slice for 3D) used to
        close small holes. ``0`` disables.
    sigmoid
        Apply ``sigmoid`` to the network output before thresholding (the
        loss-during-training was ``BCEWithLogits``, so logits come out).
    """

    def __init__(
        self,
        backbone: UNetBackbone,
        *,
        n_tiles: Optional[List[int]] = None,
        tile_overlap: float = 0.125,
        batch_size: int = 4,
        num_workers: int = 0,
        pmin: Optional[float] = 0.1,
        pmax: Optional[float] = 99.9,
        device: Optional[str] = None,
        min_size: int = 10,
        morph_iterations: int = 0,
        sigmoid: bool = True,
    ):
        self.backbone = backbone
        self.n_tiles = list(n_tiles) if n_tiles is not None else [1, 4, 4]
        self.tile_overlap = float(tile_overlap)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self._normalizer = (
            PercentileNormalize(pmin=pmin, pmax=pmax)
            if pmin is not None and pmax is not None
            else None
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.module.to(self.device)
        self.min_size = int(min_size)
        self.morph_iterations = int(morph_iterations)
        self.sigmoid = bool(sigmoid)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        **kwargs,
    ) -> "UNetSegmenter":
        backbone_kwargs = {
            k: kwargs.pop(k)
            for k in (
                "conv_dims", "in_channels", "num_classes",
                "depth", "num_channels_init", "use_batch_norm",
                "map_location",
            )
            if k in kwargs
        }
        return cls(UNetBackbone.from_checkpoint(checkpoint, **backbone_kwargs), **kwargs)

    def predict(
        self,
        image: np.ndarray,
        *,
        n_tiles: Optional[Tuple[int, int, int]] = None,
        **_ignored,
    ) -> Result:
        if image.ndim != 3:
            raise ValueError(f"UNetSegmenter.predict expects a 3D volume, got ndim={image.ndim}")

        n = tuple(n_tiles) if n_tiles is not None else tuple(self.n_tiles)
        tile_shape = compute_tile_shape(image.shape, n)

        dataset = CarePredictionDataset(
            volume=image.astype(np.float32),
            tile_shape=tile_shape,
            overlap=self.tile_overlap,
            normalizer=self._normalizer,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        predictions = []
        self.backbone.module.eval()
        with torch.no_grad():
            for tiles, coords in loader:
                tiles = tiles.to(self.device)
                pred, coords_out = self.backbone.module.predict_step(
                    (tiles, coords), batch_idx=0
                )
                if self.sigmoid:
                    pred = torch.sigmoid(pred)
                predictions.append((pred, coords_out))

        prob = stitch_tiles(predictions, image.shape, overlap_fraction=self.tile_overlap)

        try:
            thresholds = threshold_multiotsu(prob, classes=2)
            binary = np.digitize(prob, bins=thresholds) > 0
        except ValueError:
            binary = prob > 0.5

        if self.morph_iterations > 0 and binary.ndim == 3:
            for z in range(binary.shape[0]):
                binary[z] = binary_dilation(binary[z], iterations=self.morph_iterations)
                binary[z] = binary_erosion(binary[z], iterations=self.morph_iterations)

        labels = cc_label(binary)
        if self.min_size > 0:
            labels = remove_small_objects(labels.astype(np.int64), min_size=self.min_size)
        labels = relabel_sequential(labels.astype(np.uint32))[0]

        return Result(labels=labels, semantic=binary, probability=prob)
