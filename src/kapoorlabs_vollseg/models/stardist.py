"""StarDist Layer-1 singleton — first-class PyTorch implementation.

Wraps :class:`StarDistBackbone` and exposes the same
:meth:`Pipeline.predict(image) -> Result` contract every other singleton
implements, so the Layer-2 composites (``UNetStarDistPipeline``,
``ROIPipeline``, ``DenoisedPipeline``, …) and the
:class:`kapoorlabs_vollseg.VollSeg.from_models` factory work unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from .._backbones.stardist import StarDistBackbone
from ..pipelines.base import Result
from ..stardist.inference import predict_volume


class StarDistSegmenter:
    """Run a PyTorch StarDist model and return instance labels.

    Parameters
    ----------
    backbone
        A :class:`StarDistBackbone` with weights loaded.
    prob_thresh, nms_thresh, min_distance
        Thresholds passed to :func:`kapoorlabs_vollseg.stardist.predict_volume`.
    n_tiles, tile_overlap, batch_size, num_workers, pmin, pmax, device
        Forwarded to the inference loop; same meanings as in
        :class:`kapoorlabs_vollseg.CAREDenoiser`.
    """

    def __init__(
        self,
        backbone: StarDistBackbone,
        *,
        prob_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        min_distance: int = 2,
        n_tiles: Optional[list[int]] = None,
        tile_overlap: float = 0.125,
        batch_size: int = 4,
        num_workers: int = 0,
        pmin: Optional[float] = 0.1,
        pmax: Optional[float] = 99.9,
        device: Optional[str] = None,
    ):
        self.backbone = backbone
        self.prob_thresh = float(prob_thresh)
        self.nms_thresh = float(nms_thresh)
        self.min_distance = int(min_distance)
        self.n_tiles = list(n_tiles) if n_tiles is not None else [1, 4, 4]
        self.tile_overlap = float(tile_overlap)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pmin = pmin
        self.pmax = pmax
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        *,
        rays: np.ndarray,
        **kwargs,
    ) -> StarDistSegmenter:
        """Build directly from a Lightning ``.ckpt`` plus the rays array."""
        backbone_kwargs = {
            k: kwargs.pop(k)
            for k in (
                "conv_dims",
                "in_channels",
                "depth",
                "num_channels_init",
                "use_batch_norm",
                "map_location",
            )
            if k in kwargs
        }
        backbone = StarDistBackbone.from_checkpoint(
            checkpoint, rays=rays, **backbone_kwargs
        )
        return cls(backbone, **kwargs)

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, Path],
        *,
        rays: Optional[np.ndarray] = None,
        n_rays: int = 96,
        **kwargs,
    ) -> StarDistSegmenter:
        """Build from a folder holding the ``.ckpt``, optional ``rays.npy``,
        and ``training_config.json`` (or fallback JSON). When ``rays`` is
        not passed, the loader looks for ``rays.npy`` / ``*rays*.npy`` in
        the folder; if neither exists it generates a fresh golden-spiral
        set of length ``n_rays``."""
        from .._backbones._config import (
            find_checkpoint,
            find_rays,
            read_training_config,
        )
        from ..stardist.rays import rays_3d_golden_spiral

        ckpt = find_checkpoint(folder)
        arch = read_training_config(folder)
        arch.update(kwargs)

        if rays is None:
            rays_path = find_rays(folder)
            rays = (
                np.load(rays_path)
                if rays_path is not None
                else rays_3d_golden_spiral(n_rays)
            )
        return cls.from_checkpoint(ckpt, rays=rays, **arch)

    def predict(
        self,
        image: np.ndarray,
        *,
        prob_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
        n_tiles: Optional[tuple[int, ...]] = None,
        **_ignored,
    ) -> Result:
        sd = predict_volume(
            self.backbone.module,
            image,
            self.backbone.rays,
            prob_thresh=prob_thresh if prob_thresh is not None else self.prob_thresh,
            nms_thresh=nms_thresh if nms_thresh is not None else self.nms_thresh,
            min_distance=self.min_distance,
            n_tiles=n_tiles if n_tiles is not None else self.n_tiles,
            tile_overlap=self.tile_overlap,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pmin=self.pmin,
            pmax=self.pmax,
            device=self.device,
        )
        return Result(labels=sd.labels, probability=sd.prob_map)
