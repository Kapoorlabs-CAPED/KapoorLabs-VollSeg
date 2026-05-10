"""CARE denoising singleton — first-class PyTorch implementation.

Tiled inference: a 3D volume is sliced via :class:`CarePredictionDataset`,
each tile is run through the wrapped :class:`CareModule`, and the tiles
are blended back via :func:`stitch_tiles`. The Layer-2 composites
(:class:`DenoisedPipeline` etc.) consume :class:`Result` exactly as they
did with the keras singleton — no composite needs to know which backend
produced the denoised output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .._backbones.care import CAREBackbone
from .._lightning.dataset import CarePredictionDataset, compute_tile_shape
from .._lightning.stitch import stitch_tiles
from .._lightning.transforms import PercentileNormalize
from ..pipelines.base import Result


class CAREDenoiser:
    """Run a PyTorch CARE network on a 3D volume.

    Parameters
    ----------
    backbone
        A :class:`CAREBackbone` with weights loaded.
    n_tiles
        Per-axis tile count for inference. Defaults to ``[1, 4, 4]``,
        matching kapoorlabs-lightning's CareInception.
    tile_overlap
        Fraction of overlap between adjacent tiles for the linear blend.
    batch_size, num_workers
        DataLoader knobs for the tile inference pass.
    pmin, pmax
        Percentile-normalize input volumes to ``[0, 1]`` using these
        percentiles. Set ``pmin=pmax=None`` to skip normalization.
    device
        Override the inference device; defaults to ``cuda`` if available.
    """

    def __init__(
        self,
        backbone: CAREBackbone,
        *,
        n_tiles: Optional[list[int]] = None,
        tile_overlap: float = 0.125,
        batch_size: int = 4,
        num_workers: int = 0,
        pmin: Optional[float] = 0.1,
        pmax: Optional[float] = 99.9,
        device: Optional[str] = None,
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

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        **kwargs,
    ) -> CAREDenoiser:
        """Convenience constructor that builds the backbone from a ``.ckpt``.

        Architecture knobs not consumed by the singleton are forwarded to
        :meth:`CAREBackbone.from_checkpoint`.
        """
        backbone_kwargs = {
            k: kwargs.pop(k)
            for k in (
                "conv_dims",
                "in_channels",
                "num_classes",
                "depth",
                "num_channels_init",
                "use_batch_norm",
                "map_location",
            )
            if k in kwargs
        }
        return cls(
            CAREBackbone.from_checkpoint(checkpoint, **backbone_kwargs), **kwargs
        )

    def predict(
        self,
        image: np.ndarray,
        *,
        n_tiles: Optional[tuple[int, int, int]] = None,
        **_ignored,
    ) -> Result:
        if image.ndim != 3:
            raise ValueError(
                f"CAREDenoiser.predict expects a 3D volume, got ndim={image.ndim}"
            )

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
                predictions.append((pred, coords_out))

        denoised = stitch_tiles(
            predictions, image.shape, overlap_fraction=self.tile_overlap
        )
        return Result(denoised=denoised)
