"""Generate H5 training files for StarDist from a paired raw/label directory.

Stores ``(raw, label)`` per split — *not* the precomputed prob/dist
targets. Targets are computed on the fly inside the dataset's
``__getitem__`` after geometric augmentation, so any flip / rotation in
any ndim works without ray-channel gymnastics. This matches upstream
stardist's training pattern.

H5 layout::

    /train/raw    (N, *patch_shape)   float32
    /train/label  (N, *patch_shape)   int32
    /val/raw, /val/label              same shape

Foreground filtering is by *fraction of non-zero label voxels* per
patch, not by the prob target's peak — equivalent in spirit and saves
us from materializing the prob map at prep time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
from collections.abc import Iterator, Sequence

import h5py
import numpy as np
from tifffile import imread
from tqdm import tqdm


_PatchShape = Union[tuple[int, int], tuple[int, int, int]]
_Stride = Union[tuple[int, int], tuple[int, int, int]]


def generate_stardist_h5(
    raw_dir: Union[str, Path],
    label_dir: Union[str, Path],
    output_h5: Union[str, Path],
    *,
    patch_shape: _PatchShape,
    stride_train: Optional[_Stride] = None,
    stride_val: Optional[_Stride] = None,
    val_files: int = 1,
    min_foreground_ratio: float = 0.0,
    extensions: Sequence[str] = (".tif", ".tiff", ".TIF", ".TIFF"),
    chunk_rows: int = 16,
    overwrite: bool = False,
) -> dict:
    """Build a StarDist training H5 from paired raw / label directories.

    Parameters
    ----------
    raw_dir, label_dir
        Directories of paired files (matched by sorted basename).
    output_h5
        Destination ``.h5`` path.
    patch_shape
        Per-axis patch size. ``len`` must be 2 or 3.
    stride_train, stride_val
        Sliding-window stride. Defaults: ``patch_shape // 2`` for train
        (≈50% overlap), ``patch_shape`` for val (no overlap).
    val_files
        Number of files at the end of the sorted list reserved for val.
    min_foreground_ratio
        Drop patches whose fraction of non-zero label voxels is below
        this threshold. ``0`` keeps everything.
    extensions, chunk_rows, overwrite
        Standard H5 / file-listing knobs.

    Returns
    -------
    dict
        ``{"train": N_train, "val": N_val}`` patch counts.
    """
    raw_dir, label_dir, output_h5 = Path(raw_dir), Path(label_dir), Path(output_h5)
    ndim = len(patch_shape)
    if ndim not in (2, 3):
        raise ValueError(f"patch_shape must be length 2 or 3, got {patch_shape}")

    stride_train = (
        tuple(stride_train)
        if stride_train is not None
        else tuple(p // 2 for p in patch_shape)
    )
    stride_val = tuple(stride_val) if stride_val is not None else tuple(patch_shape)

    pairs = _list_paired_files(raw_dir, label_dir, extensions)
    if not pairs:
        raise FileNotFoundError(f"No paired files between {raw_dir} and {label_dir}")
    if val_files >= len(pairs):
        raise ValueError(f"val_files={val_files} but only {len(pairs)} files found")

    train_pairs = pairs[:-val_files]
    val_pairs = pairs[-val_files:]

    if output_h5.exists() and overwrite:
        output_h5.unlink()

    counts = {"train": 0, "val": 0}
    with h5py.File(output_h5, "a") as f:
        train_writer = _SplitWriter(f, "train", patch_shape, chunk_rows)
        val_writer = _SplitWriter(f, "val", patch_shape, chunk_rows)

        for raw_path, label_path in tqdm(
            train_pairs, desc=f"train ({len(train_pairs)} files)"
        ):
            counts["train"] += _emit_from_file(
                raw_path,
                label_path,
                train_writer,
                patch_shape,
                stride_train,
                ndim,
                min_foreground_ratio,
            )
        for raw_path, label_path in tqdm(
            val_pairs, desc=f"val ({len(val_pairs)} files)"
        ):
            counts["val"] += _emit_from_file(
                raw_path,
                label_path,
                val_writer,
                patch_shape,
                stride_val,
                ndim,
                min_foreground_ratio,
            )

    print(f"Wrote {counts['train']} train + {counts['val']} val patches → {output_h5}")
    return counts


# --------------------------------------------------------------- internals


def _list_paired_files(raw_dir: Path, label_dir: Path, exts: Sequence[str]):
    raw_files = sorted(p for p in raw_dir.iterdir() if p.suffix in exts)
    pairs = []
    for raw in raw_files:
        label_path = label_dir / raw.name
        if label_path.exists():
            pairs.append((raw, label_path))
    return pairs


def _iter_window_origins(
    volume_shape, patch_shape, stride
) -> Iterator[tuple[int, ...]]:
    starts_per_axis = []
    for v, p, s in zip(volume_shape, patch_shape, stride):
        if v < p:
            return
        positions = list(range(0, v - p + 1, s))
        if positions[-1] != v - p:
            positions.append(v - p)
        starts_per_axis.append(positions)

    if len(starts_per_axis) == 2:
        for y in starts_per_axis[0]:
            for x in starts_per_axis[1]:
                yield (y, x)
    else:
        for z in starts_per_axis[0]:
            for y in starts_per_axis[1]:
                for x in starts_per_axis[2]:
                    yield (z, y, x)


def _slice_for_origin(origin, patch_shape):
    return tuple(slice(o, o + p) for o, p in zip(origin, patch_shape))


def _emit_from_file(
    raw_path,
    label_path,
    writer,
    patch_shape,
    stride,
    ndim,
    min_fg_ratio,
) -> int:
    raw_vol = imread(raw_path).astype(np.float32)
    label_vol = imread(label_path).astype(np.int32)
    if raw_vol.shape != label_vol.shape:
        raise ValueError(
            f"Shape mismatch for {raw_path.name}: {raw_vol.shape} vs {label_vol.shape}"
        )
    if raw_vol.ndim != ndim:
        raise ValueError(f"{raw_path.name}: expected ndim={ndim}, got {raw_vol.ndim}")

    n_emitted = 0
    for origin in _iter_window_origins(raw_vol.shape, patch_shape, stride):
        sl = _slice_for_origin(origin, patch_shape)
        label_patch = label_vol[sl]
        if min_fg_ratio > 0:
            fg = float((label_patch > 0).sum()) / label_patch.size
            if fg < min_fg_ratio:
                continue
        writer.append(raw_vol[sl], label_patch)
        n_emitted += 1
    return n_emitted


class _SplitWriter:
    """Resizable-H5 writer for one split — two datasets, ``raw`` + ``label``."""

    def __init__(self, h5file: h5py.File, split: str, patch_shape, chunk_rows: int):
        self.split = split
        self.h5 = h5file
        if split in h5file:
            del h5file[split]
        grp = h5file.create_group(split)
        self.raw = grp.create_dataset(
            "raw",
            shape=(0, *patch_shape),
            maxshape=(None, *patch_shape),
            dtype="float32",
            chunks=(chunk_rows, *patch_shape),
            compression="lzf",
        )
        self.label = grp.create_dataset(
            "label",
            shape=(0, *patch_shape),
            maxshape=(None, *patch_shape),
            dtype="int32",
            chunks=(chunk_rows, *patch_shape),
            compression="lzf",
        )

    def append(self, raw_patch, label_patch):
        n = self.raw.shape[0]
        self.raw.resize(n + 1, axis=0)
        self.label.resize(n + 1, axis=0)
        self.raw[n] = raw_patch
        self.label[n] = label_patch
