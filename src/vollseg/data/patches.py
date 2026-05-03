"""Smart patch extraction with foreground/background ratio vetoing.

Walks instance label images and emits patches centered on each instance
that pass a foreground/background ratio gate. Optionally caps the count
per image. Foreground patches are written to ``raw_out`` /
``binary_mask_out`` / ``real_mask_out``; if ``include_background=True``,
patches drawn from purely-background regions of the image are also
written, with a ``"back"`` suffix in the filename.

This is a clean rewrite of the original ``SmartPatches`` — single-channel
(call twice for nuclei + membrane), no built-in two-channel duplication,
fewer constructor arguments.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple, Union
from uuid import uuid4

import numpy as np
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from tifffile import imwrite
from tqdm import tqdm

from .io import iter_image_files, read_float, read_int
from .labels import erode_labels


_PatchSize = Union[Tuple[int, int], Tuple[int, int, int]]


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
    max_foreground_patches_per_image, max_background_patches_per_image
        Per-image caps; ``np.inf`` for unlimited.
    include_background
        Also emit purely-background patches.
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
        max_background_patches_per_image: float = np.inf,
        include_background: bool = True,
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
        self.max_bg = max_background_patches_per_image
        self.include_background = include_background
        self.pattern = pattern

    # ----------------------------------------------------------- public

    def run(self) -> dict:
        """Walk the input directories and write all patches. Returns counts."""
        n_fg = n_bg = 0
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
            if self.include_background:
                n_bg += self._emit_background(raw_path.stem, raw, labels)

        return dict(foreground=n_fg, background=n_bg)

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
            patch_labels = remove_small_objects(patch_labels.astype("uint16"), min_size=10)

            if patch_labels.shape != self.patch_size:
                continue
            if not self._foreground_ratio_ok(patch_labels):
                continue

            if self.erosion_iterations > 0:
                patch_labels = erode_labels(patch_labels, self.erosion_iterations).astype("uint16")
            self._write_triplet(name, count, raw[region], patch_labels)
            count += 1
        return count

    # --------------------------------------------------------- background

    def _emit_background(self, name: str, raw: np.ndarray, labels: np.ndarray) -> int:
        count = 0
        zero_indices = list(zip(*np.where(labels == 0)))
        for idx in zero_indices:
            if count >= self.max_bg:
                break
            region = self._region_around(idx, raw.shape)
            if region is None:
                continue
            patch_labels_zero = labels[region]
            if np.sum(patch_labels_zero) != 0:
                continue
            if patch_labels_zero.shape != self.patch_size:
                continue
            if np.sum(raw[region]) <= 0:
                continue
            self._write_triplet(f"{name}back", count, raw[region], patch_labels_zero)
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

    def _write_triplet(self, name: str, count: int, raw_patch: np.ndarray, label_patch: np.ndarray):
        eventid = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
        suffix = f"{name}{eventid}{count}{self.pattern}"
        imwrite(os.fspath(self.raw_out / suffix), raw_patch.astype(np.float32))
        imwrite(os.fspath(self.binary_mask_out / suffix), (label_patch > 0).astype(np.uint16))
        imwrite(os.fspath(self.real_mask_out / suffix), label_patch.astype(np.uint16))
