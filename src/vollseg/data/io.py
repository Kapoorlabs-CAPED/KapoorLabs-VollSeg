"""Tiny file-IO helpers shared by trainers and patch generators."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Sequence, Union

import numpy as np
from tifffile import imread


_DEFAULT_EXTS = (".tif", ".tiff", ".TIF", ".TIFF", ".png")


def read_float(path: Union[str, Path]) -> np.ndarray:
    """Load an image as float32 (any tifffile-readable format)."""
    return imread(str(path)).astype(np.float32)


def read_int(path: Union[str, Path]) -> np.ndarray:
    """Load a label image as uint16 (any tifffile-readable format)."""
    return imread(str(path)).astype(np.uint16)


def iter_image_files(
    directory: Union[str, Path],
    extensions: Sequence[str] = _DEFAULT_EXTS,
) -> Iterator[Path]:
    """Yield image-file paths in ``directory``, sorted, filtered by extension."""
    d = Path(directory)
    for name in sorted(os.listdir(d)):
        if any(name.endswith(ext) for ext in extensions):
            yield d / name
