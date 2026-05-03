"""Joint U-Net + StarDist training orchestrator.

The original ``SmartSeeds2D`` / ``SmartSeeds3D`` god-classes did seven
things at once: scan dirs, generate NPZ, derive binary masks from instance
masks, train U-Net, train StarDist, manage checkpoints, plot history.
This rewrite delegates each concern:

- IO and pairing → :func:`vollseg.data.iter_image_files`
- binary↔instance conversion → :mod:`vollseg.data.labels`
- streaming loaders → :mod:`vollseg.data.sequencer`
- training → :class:`UNetTrainer` + :class:`StarDistTrainer`

:class:`SmartSeeds` is just the chain that runs them in order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from tifffile import imread, imwrite

from ..data.io import iter_image_files
from ..data.labels import binary_to_labels, erode_labels, labels_to_binary
from ..data.sequencer import StarDistSequencer, UNetSequencer
from .stardist import StarDistTrainer
from .unet import UNetTrainer


class SmartSeeds:
    """End-to-end U-Net + StarDist training from a directory of paired files.

    Expected layout under ``base_dir``::

        raw/                   raw images (float)
        binary_mask/           binary masks (auto-generated if missing)
        real_mask/             instance label images (auto-generated if missing)
        val_raw/               validation raw images
        val_real_mask/         validation instance labels

    Either ``binary_mask`` or ``real_mask`` may be initially empty: this
    class will derive one from the other.

    Parameters
    ----------
    train_unet, train_star
        Stages to run; both default to True.
    erosion_iterations
        How much to erode each instance before deriving the binary mask
        (helps the U-Net learn separation between touching cells).
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        *,
        unet_trainer: UNetTrainer,
        stardist_trainer: StarDistTrainer,
        raw_dir: str = "raw",
        real_mask_dir: str = "real_mask",
        binary_mask_dir: str = "binary_mask",
        val_raw_dir: str = "val_raw",
        val_real_mask_dir: str = "val_real_mask",
        train_unet: bool = True,
        train_star: bool = True,
        erosion_iterations: int = 2,
        batch_size: int = 4,
        patch_size: Tuple[int, ...] = (16, 256, 256),
        axis_norm: Tuple[int, ...] = (0, 1, 2),
    ):
        self.base = Path(base_dir)
        self.unet_trainer = unet_trainer
        self.stardist_trainer = stardist_trainer
        self.raw_dir = self.base / raw_dir
        self.real_mask_dir = self.base / real_mask_dir
        self.binary_mask_dir = self.base / binary_mask_dir
        self.val_raw_dir = self.base / val_raw_dir
        self.val_real_mask_dir = self.base / val_real_mask_dir
        self.train_unet = train_unet
        self.train_star = train_star
        self.erosion_iterations = erosion_iterations
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.axis_norm = axis_norm

    # --------------------------------------------------------- main flow

    def run(self):
        for d in (self.raw_dir, self.real_mask_dir, self.binary_mask_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._sync_masks()

        if self.train_unet:
            self._train_unet()
        if self.train_star:
            self._train_stardist()

    # ----------------------------------------------------- mask sync

    def _sync_masks(self):
        """Make sure both binary_mask/ and real_mask/ exist for every raw."""
        binary_names = {p.name for p in iter_image_files(self.binary_mask_dir)}
        real_names = {p.name for p in iter_image_files(self.real_mask_dir)}

        if binary_names and not real_names:
            print("Deriving real_mask/ from binary_mask/")
            for p in iter_image_files(self.binary_mask_dir):
                img = imread(p).astype(np.uint16)
                if img.max() == 1:
                    img = img * 255
                imwrite(self.real_mask_dir / p.name, binary_to_labels(img))

        if real_names and not binary_names:
            print("Deriving binary_mask/ from real_mask/")
            for p in iter_image_files(self.real_mask_dir):
                img = imread(p).astype(np.uint16)
                if self.erosion_iterations > 0:
                    img = erode_labels(img, self.erosion_iterations).astype(np.uint16)
                imwrite(self.binary_mask_dir / p.name, labels_to_binary(img).astype(np.uint16))

    # ----------------------------------------------------- U-Net stage

    def _train_unet(self):
        print("=== Training U-Net ===")
        raw_files = list(iter_image_files(self.raw_dir))
        mask_files = [self.binary_mask_dir / r.name for r in raw_files]
        val_raw_files = list(iter_image_files(self.val_raw_dir))
        val_mask_files = [self.val_real_mask_dir / r.name for r in val_raw_files]

        train_seq = UNetSequencer(
            raw_files, mask_files,
            axis_norm=self.axis_norm, batch_size=self.batch_size, shape=self.patch_size,
        )
        val_seq = UNetSequencer(
            val_raw_files, val_mask_files,
            axis_norm=self.axis_norm, batch_size=self.batch_size, shape=self.patch_size,
        )
        return self.unet_trainer.fit(
            train_seq, validation_data=val_seq, load_data_sequence=True
        )

    # --------------------------------------------------- StarDist stage

    def _train_stardist(self):
        print("=== Training StarDist ===")
        raw_files = list(iter_image_files(self.raw_dir))
        real_mask_files = list(iter_image_files(self.real_mask_dir))
        val_raw_files = list(iter_image_files(self.val_raw_dir))
        val_real_mask_files = list(iter_image_files(self.val_real_mask_dir))

        X_trn = StarDistSequencer(raw_files, axis_norm=self.axis_norm, normalize_inputs=True)
        Y_trn = StarDistSequencer(real_mask_files, axis_norm=self.axis_norm,
                                  normalize_inputs=False, label_me=True)
        X_val = StarDistSequencer(val_raw_files, axis_norm=self.axis_norm, normalize_inputs=True)
        Y_val = StarDistSequencer(val_real_mask_files, axis_norm=self.axis_norm,
                                  normalize_inputs=False, label_me=True)

        return self.stardist_trainer.fit(X_trn, Y_trn, validation_data=(X_val, Y_val))
