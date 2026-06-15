"""U-Net semantic segmentation singleton — first-class PyTorch implementation.

Same tiled inference path as :class:`CAREDenoiser`; the difference is in
post-processing — multi-Otsu → connected-components → size filter — to
produce the ``Result.semantic`` / ``Result.labels`` shape that the
Layer 2 composites expect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

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
    """

    def __init__(
        self,
        backbone: UNetBackbone,
        *,
        n_tiles: Optional[list[int]] = None,
        tile_overlap: float = 0.125,
        batch_size: int = 4,
        num_workers: int = 0,
        pmin: Optional[float] = 0.1,
        pmax: Optional[float] = 99.9,
        device: Optional[str] = None,
        min_size: int = 10,
        morph_iterations: int = 0,
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

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        **kwargs,
    ) -> UNetSegmenter:
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
            UNetBackbone.from_checkpoint(checkpoint, **backbone_kwargs), **kwargs
        )

    @classmethod
    def from_folder(cls, folder: Union[str, Path], **kwargs) -> UNetSegmenter:
        """Build from a model folder containing the ``.ckpt`` plus a
        ``training_config.json`` (or fallback ``{experiment_name}.json``).
        See :meth:`CAREDenoiser.from_folder` for the full contract."""
        from .._backbones._config import find_checkpoint, read_training_config

        ckpt = find_checkpoint(folder)
        arch = read_training_config(folder)
        arch.update(kwargs)
        return cls.from_checkpoint(ckpt, **arch)

    @property
    def model_dim(self) -> int:
        """Spatial dimensionality the loaded network was trained for (2 or 3).

        Inferred from the first conv layer's weight tensor — same logic
        as :func:`kapoorlabs_vollseg._backbones.care.infer_arch_from_checkpoint`
        but applied to the live ``nn.Module``.
        """
        for m in self.backbone.module.network.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)):
                return m.weight.ndim - 2  # (out, in, *spatial)
        return 3

    def predict(
        self,
        image: np.ndarray,
        *,
        n_tiles: Optional[tuple[int, ...]] = None,
        **_ignored,
    ) -> Result:
        """Dispatch on (model_dim, image.ndim):

        - 3D model, 3D image — direct tiled prediction (original path).
        - 2D model, 2D image — direct 2D tiled prediction.
        - 2D model, 3D image — max-Z projection → 2D prediction → broadcast
          the 2D mask back to ZYX. Matches the original VollSeg_unet flow
          for ROI Mask-UNet (``conv_dims=2`` per ``roi.yaml``).
        - 3D model, 2D image — error: requires a Z dimension.
        """
        model_dim = self.model_dim
        if image.ndim not in (2, 3):
            raise ValueError(
                f"{type(self).__name__}.predict expects a 2D or 3D image, "
                f"got ndim={image.ndim}"
            )
        if model_dim == 3 and image.ndim == 2:
            raise ValueError(
                f"{type(self).__name__}: model is 3D but input is 2D; "
                f"cannot stretch a 3D backbone over a single slice."
            )

        # 2D-on-3D: MIP first, predict on the projection, broadcast at the end.
        original_shape = image.shape
        was_mip = model_dim == 2 and image.ndim == 3
        if was_mip:
            image = np.amax(image, axis=0)

        n = self._resolve_n_tiles(n_tiles, image.ndim)

        # Percentile-normalise the WHOLE input once, before tiling.
        # Per-tile normalisation (the old behaviour) stretches pure-
        # background tiles' noise to [0, 1] and the model hallucinates
        # foreground there. See the StarDist note in
        # ``stardist/inference.py::_predict_and_stitch``; same fix
        # applied here so U-Net inference matches the original
        # CSBDeep convention (``normalize(img)`` before prediction).
        image = np.ascontiguousarray(image, dtype=np.float32)
        if self._normalizer is not None:
            pmin = self._normalizer.pmin
            pmax = self._normalizer.pmax
            flat = image.ravel()
            lo = float(np.percentile(flat, pmin))
            hi = float(np.percentile(flat, pmax))
            image = (image - lo) / (hi - lo + 1e-8)

        # careamics' U-Net halves each spatial axis at every pool and
        # then ``torch.cat([upsample, skip])``s — sizes diverge unless
        # the input is a multiple of ``2**depth``. Pad both the whole
        # volume AND each tile up to that multiple, predict, then crop
        # back to ``pre_pad_shape`` before stitching.
        multiple = self._required_multiple()
        pre_pad_shape = image.shape
        pad_widths = tuple((0, (-s) % multiple) for s in image.shape)
        if any(after > 0 for _, after in pad_widths):
            image = np.pad(image, pad_widths, mode="reflect")

        # Compute the tile shape on the padded image and round each
        # axis up to the same ``multiple`` so per-tile sizes also pass.
        raw_tile_shape = compute_tile_shape(image.shape, n)
        tile_shape = tuple(
            int(np.ceil(t / multiple)) * multiple for t in raw_tile_shape
        )

        dataset = CarePredictionDataset(
            volume=image,
            tile_shape=tile_shape,
            overlap=self.tile_overlap,
            normalizer=None,  # already normalised whole-volume above
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

        prob = stitch_tiles(
            predictions, image.shape, overlap_fraction=self.tile_overlap
        )
        # Crop the stitched probability map back to the unpadded shape.
        if prob.shape != pre_pad_shape:
            prob = prob[tuple(slice(0, s) for s in pre_pad_shape)]

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
            labels = remove_small_objects(
                labels.astype(np.int64), min_size=self.min_size
            )
        labels = relabel_sequential(labels.astype(np.uint32))[0]

        # Broadcast 2D-model output back to the original 3D shape — same
        # gate-the-whole-stack semantics as the original VollSeg_unet.
        if was_mip:
            labels = np.broadcast_to(labels, original_shape).copy()
            binary = np.broadcast_to(binary, original_shape).copy()
            prob = np.broadcast_to(prob, original_shape).copy()

        return Result(labels=labels, semantic=binary, probability=prob)

    def _required_multiple(self) -> int:
        """Return ``2**depth`` for the loaded careamics U-Net.

        careamics builds the encoder as ``[Conv_Block, pool] * depth``
        but reuses ONE ``self.pooling`` instance across all levels, so
        ``network.modules()`` only sees that instance once (PyTorch
        dedupes by identity). Counting ``Conv_Block`` modules in the
        encoder ``ModuleList`` instead gives the real depth: the encoder
        has ``depth`` ``Conv_Block``s, and each level halves every
        spatial axis, so the input must be a multiple of ``2**depth``.
        """
        encoder = getattr(self.backbone.module.network, "encoder", None)
        encoder_blocks = getattr(encoder, "encoder_blocks", None)
        if encoder_blocks is not None:
            depth = sum(1 for m in encoder_blocks if type(m).__name__ == "Conv_Block")
        else:
            depth = 0
        return 1 << max(depth, 0)

    def _resolve_n_tiles(
        self,
        n_tiles: Optional[tuple[int, ...]],
        image_ndim: int,
    ) -> tuple[int, ...]:
        """Coerce the user-supplied ``n_tiles`` to match the (post-MIP) image
        dimensionality. A 3-tuple ``(z, y, x)`` collapses to ``(y, x)`` for
        a 2D image; a missing value falls back to ``self.n_tiles``."""
        candidate = tuple(n_tiles) if n_tiles is not None else tuple(self.n_tiles)
        if len(candidate) == image_ndim:
            return candidate
        if len(candidate) > image_ndim:
            return candidate[-image_ndim:]
        return (1,) * (image_ndim - len(candidate)) + candidate
