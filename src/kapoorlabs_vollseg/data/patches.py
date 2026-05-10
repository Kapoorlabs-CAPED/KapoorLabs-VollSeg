"""SmartPatches — instance-centered patch extraction + background-paste augmentation.

Single-file (per-patch TIF) port of the original VollSeg ``SmartPatches``,
faithful to its two emit modes:

1. **Foreground patches** — instance-centered, kept only if the
   foreground voxel fraction lies in
   ``[lower_ratio_fore_to_back, upper_ratio_fore_to_back]``.

2. **Background-paste augmentation** (the smart bit) — additively
   blend a cell patch onto a patch centered on a *background* voxel
   (``raw_aug = raw_bg + raw_fg``, ``label_aug = label_fg``). Trains
   the network to be robust to background appearance.

For an H5-emitting variant of the same algorithm, see
:func:`kapoorlabs_vollseg.data.generate_smart_patches_h5`.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Union
from uuid import uuid4

import numpy as np
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from tifffile import imwrite
from tqdm import tqdm

from .io import iter_image_files, read_float, read_int
from .labels import erode_labels


_PatchSize = Union[tuple[int, int], tuple[int, int, int]]


class SmartPatches:
    """Extract instance-centered patches subject to a foreground ratio gate.

    Parameters
    ----------
    raw_dir, label_dir
        Input directories. Files are paired by sorted basename; raw is read
        as float32, labels as uint16.
    raw_out, binary_mask_out, real_mask_out
        Output directories (created if missing).
    patch_size
        Per-axis patch shape. Length 2 for 2D, length 3 for 3D.
    lower_ratio_fore_to_back, upper_ratio_fore_to_back
        Foreground-pixel fraction must lie in this interval for a patch to
        be emitted.
    erosion_iterations
        Erode each instance by this many iterations *before* writing the
        real-mask patch. ``0`` disables.
    max_foreground_patches_per_image
        Per-image cap on instance-centered patches; ``np.inf`` for unlimited.
    paste_augmentation
        If True, also emit background-paste augmented patches (cells
        composited additively onto patches drawn from background-only
        regions of the image).
    max_paste_patches_per_image
        Per-image cap on the paste-augmented patches.
    seed
        Seed for the RNG that shuffles paste pairings (deterministic).
    pattern
        File extension to write (``.tif`` by default).
    """

    def __init__(
        self,
        raw_dir: Union[str, Path],
        label_dir: Union[str, Path],
        raw_out: Union[str, Path],
        binary_mask_out: Union[str, Path],
        real_mask_out: Union[str, Path],
        patch_size: _PatchSize,
        *,
        lower_ratio_fore_to_back: float = 0.5,
        upper_ratio_fore_to_back: float = 0.9,
        erosion_iterations: int = 0,
        max_foreground_patches_per_image: float = np.inf,
        paste_augmentation: bool = False,
        max_paste_patches_per_image: float = 0,
        seed: int = 42,
        pattern: str = ".tif",
    ):
        self.raw_dir = Path(raw_dir)
        self.label_dir = Path(label_dir)
        self.raw_out = Path(raw_out)
        self.binary_mask_out = Path(binary_mask_out)
        self.real_mask_out = Path(real_mask_out)
        for d in (self.raw_out, self.binary_mask_out, self.real_mask_out):
            d.mkdir(parents=True, exist_ok=True)

        self.patch_size = tuple(patch_size)
        self.lower = float(lower_ratio_fore_to_back)
        self.upper = float(upper_ratio_fore_to_back)
        self.erosion_iterations = int(erosion_iterations)
        self.max_fg = max_foreground_patches_per_image
        self.paste_augmentation = bool(paste_augmentation)
        self.max_paste = max_paste_patches_per_image
        self.rng = np.random.default_rng(seed)
        self.pattern = pattern

    # ----------------------------------------------------------- public

    def run(self) -> dict:
        """Walk the input directories and write all patches. Returns counts."""
        n_fg = n_paste = 0
        for raw_path in iter_image_files(self.raw_dir):
            label_path = self.label_dir / raw_path.name
            if not label_path.exists():
                continue
            raw = read_float(raw_path)
            labels = read_int(label_path)
            if raw.ndim != labels.ndim:
                raise ValueError(
                    f"Shape mismatch for {raw_path.name}: raw {raw.shape} vs label {labels.shape}"
                )
            if len(self.patch_size) != raw.ndim:
                raise ValueError(
                    f"patch_size {self.patch_size} doesn't match ndim={raw.ndim}"
                )

            n_fg += self._emit_foreground(raw_path.stem, raw, labels)
            if self.paste_augmentation and self.max_paste > 0:
                n_paste += self._emit_paste_augmentation(raw_path.stem, raw, labels)

        return dict(foreground=n_fg, paste_augmented=n_paste)

    # --------------------------------------------------------- foreground

    def _emit_foreground(self, name: str, raw: np.ndarray, labels: np.ndarray) -> int:
        count = 0
        for prop in tqdm(regionprops(labels), desc=f"fg patches: {name}"):
            if count >= self.max_fg:
                break
            region = self._region_around(tuple(prop.centroid), raw.shape)
            if region is None:
                continue
            patch_labels = labels[region]
            patch_labels = remove_small_objects(
                patch_labels.astype("uint16"), min_size=10
            )

            if patch_labels.shape != self.patch_size:
                continue
            if not self._foreground_ratio_ok(patch_labels):
                continue

            if self.erosion_iterations > 0:
                patch_labels = erode_labels(
                    patch_labels, self.erosion_iterations
                ).astype("uint16")
            self._write_triplet(name, count, raw[region], patch_labels)
            count += 1
        return count

    # ------------------------------------------- background-paste augmentation

    def _emit_paste_augmentation(
        self, name: str, raw: np.ndarray, labels: np.ndarray
    ) -> int:
        """Composite each cell onto a patch centered on a background voxel.

        Port of upstream ``_background_label_maker``: for each
        background voxel, take a patch (which must be pure-zero), then
        additively blend each cell's patch into it. The augmented patch
        keeps the cell's silhouette but lives in a different background
        context.
        """
        zero_coords = np.argwhere(labels == 0)
        if len(zero_coords) == 0:
            return 0
        props = list(regionprops(labels))
        if not props:
            return 0

        self.rng.shuffle(zero_coords)
        self.rng.shuffle(props)

        count = 0
        for bg_idx in zero_coords:
            if count >= self.max_paste:
                break
            bg_region = self._region_around(tuple(bg_idx), raw.shape)
            if bg_region is None:
                continue
            raw_bg = raw[bg_region]
            label_bg = labels[bg_region]
            if label_bg.sum() != 0:  # bg patch must be pure-zero
                continue

            for prop in props:
                if count >= self.max_paste:
                    break
                fg_region = self._region_around(tuple(prop.centroid), raw.shape)
                if fg_region is None:
                    continue
                raw_fg = raw[fg_region]
                label_fg = labels[fg_region]

                raw_aug = raw_bg + raw_fg
                label_aug = label_bg + label_fg  # == label_fg
                if raw_aug.sum() <= 0:
                    continue

                self._write_triplet(
                    f"{name}paste", count, raw_aug, label_aug.astype("uint16")
                )
                count += 1
        return count

    # --------------------------------------------------------- helpers

    def _region_around(self, center, shape):
        slices = []
        for c, p, dim in zip(center, self.patch_size, shape):
            half = p // 2
            lo = int(c - half)
            hi = int(c + half)
            if lo < 0 or hi > dim:
                return None
            slices.append(slice(lo, hi))
        return tuple(slices)

    def _foreground_ratio_ok(self, patch_labels: np.ndarray) -> bool:
        total = patch_labels.size
        if total == 0:
            return False
        fg = int(np.count_nonzero(patch_labels))
        ratio = fg / total
        return self.lower <= ratio <= self.upper

    def _write_triplet(
        self, name: str, count: int, raw_patch: np.ndarray, label_patch: np.ndarray
    ):
        eventid = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
        suffix = f"{name}{eventid}{count}{self.pattern}"
        imwrite(os.fspath(self.raw_out / suffix), raw_patch.astype(np.float32))
        imwrite(
            os.fspath(self.binary_mask_out / suffix),
            (label_patch > 0).astype(np.uint16),
        )
        imwrite(os.fspath(self.real_mask_out / suffix), label_patch.astype(np.uint16))
