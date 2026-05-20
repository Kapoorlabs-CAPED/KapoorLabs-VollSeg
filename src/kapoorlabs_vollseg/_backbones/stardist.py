"""PyTorch StarDist backbone — wraps :class:`StarDistModule` + ray geometry.

Loaded weights live inside the Lightning module; the rays themselves are
stored alongside the backbone because they're a property of the trained
model (fixed at training time, must be reused for inference).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..stardist.lightning_module import StarDistModule
from ..stardist.model import StarDistUNet
from ..stardist.rays import compute_faces


class StarDistBackbone:
    """Hold a trained :class:`StarDistModule` together with its ray geometry."""

    def __init__(self, module: StarDistModule, rays: np.ndarray):
        if rays.ndim != 2:
            raise ValueError(f"rays must be 2D (N, ndim), got shape {rays.shape}")
        if rays.shape[0] != module.n_rays:
            raise ValueError(
                f"rays.shape[0]={rays.shape[0]} doesn't match module.n_rays={module.n_rays}"
            )
        self.module = module
        self.module.eval()
        self.rays = np.ascontiguousarray(rays, dtype=np.float32)
        # Triangulated faces of the polyhedron (3D only; empty for 2D) —
        # the inference rasteriser uses these to build the actual
        # star-convex polyhedron, instead of a nearest-ray cone union.
        self.faces = compute_faces(self.rays)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        *,
        rays: np.ndarray,
        conv_dims: int = 3,
        in_channels: int = 1,
        depth: int = 3,
        num_channels_init: int = 64,
        use_batch_norm: bool = True,
        map_location: Optional[str] = None,
        weights_only: bool = False,
    ) -> StarDistBackbone:
        """Build a StarDistBackbone from a Lightning ``.ckpt``.

        ``rays`` must be the same array passed to the trainer — it can't
        be inferred from the checkpoint. Pair the checkpoint with a
        sidecar ``.npy`` (see :class:`kapoorlabs_vollseg.train.StarDistTrainer`).
        """
        n_rays = rays.shape[0]
        unet = StarDistUNet(
            n_rays=n_rays,
            conv_dims=conv_dims,
            in_channels=in_channels,
            depth=depth,
            num_channels_init=num_channels_init,
            use_batch_norm=use_batch_norm,
        )
        module = StarDistModule.load_from_checkpoint(
            checkpoint_path=str(checkpoint),
            network=unet,
            optim_func=None,
            map_location=map_location,
            weights_only=weights_only,
        )
        return cls(module, rays)
