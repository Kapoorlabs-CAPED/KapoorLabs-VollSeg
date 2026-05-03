"""Keras ``Sequence`` loaders for U-Net and StarDist training.

These read image / label files lazily, so the trainer can stream from
disk without materializing the dataset in memory. Used by
:mod:`vollseg.train`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from csbdeep.utils import normalize
from scipy.ndimage import binary_fill_holes

from .io import read_float, read_int


try:
    from tensorflow.keras.utils import Sequence as KerasSequence
except ImportError:  # tf not installed at import time — degrade gracefully
    class KerasSequence:  # type: ignore[no-redef]
        pass


class UNetSequencer(KerasSequence):
    """Yield ``(raw_batch, binary_mask_batch)`` for U-Net semantic training.

    Pairs by *position* in the two file lists, so the caller is responsible
    for matching them up (typically by sorted basename).
    """

    def __init__(
        self,
        raw_files: Sequence[Path],
        mask_files: Sequence[Path],
        *,
        axis_norm: Tuple[int, ...] = (0, 1, 2),
        batch_size: int = 1,
        shape: Tuple[int, ...] = (16, 256, 256),
    ):
        super().__init__()
        if len(raw_files) != len(mask_files):
            raise ValueError(
                f"raw and mask file lists differ: {len(raw_files)} vs {len(mask_files)}"
            )
        self.raw_files = list(raw_files)
        self.mask_files = list(mask_files)
        self.axis_norm = axis_norm
        self.batch_size = batch_size
        self.shape = shape

    def __len__(self) -> int:
        return len(self.raw_files) // self.batch_size

    def __getitem__(self, idx: int):
        i0, i1 = idx * self.batch_size, (idx + 1) * self.batch_size
        raws, masks = [], []
        for r, m in zip(self.raw_files[i0:i1], self.mask_files[i0:i1]):
            raw = read_float(r)
            mask = read_int(m)
            if raw.shape == self.shape and mask.shape == self.shape:
                raws.append(normalize(raw, 1, 99.8, axis=self.axis_norm))
                masks.append(binary_fill_holes(mask > 0))
        return (
            np.asarray(raws, dtype=np.float32),
            np.asarray(masks, dtype=np.float32),
        )


class StarDistSequencer(KerasSequence):
    """Yield individual normalized images or label maps for StarDist training.

    StarDist's trainer expects two separate sequences (X and Y) rather than
    paired batches; instantiate one for raw images (``label_me=False``) and
    one for label images (``label_me=True``).
    """

    def __init__(
        self,
        files: Sequence[Path],
        *,
        axis_norm: Tuple[int, ...] = (0, 1, 2),
        normalize_inputs: bool = True,
        label_me: bool = False,
        binary_me: bool = False,
    ):
        super().__init__()
        self.files = list(files)
        self.axis_norm = axis_norm
        self.normalize_inputs = normalize_inputs
        self.label_me = label_me
        self.binary_me = binary_me

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        if self.label_me:
            x = read_int(self.files[i])
            if self.binary_me:
                x = (x > 0).astype(np.uint16)
            return x.astype(np.uint16)
        x = read_float(self.files[i])
        if self.normalize_inputs:
            x = normalize(x, 1, 99.8, axis=self.axis_norm)
        return x
