"""SmartPatches H5 emitter — one H5 for U-Net + StarDist training.

Single H5 emitter that produces all the patches needed by both U-Net
and StarDist training; same patch indices are reused for both targets,
so disk usage and prep time are minimized.

H5 layout::

    /train/raw    (N, *spatial)   float32   # always
    /train/label  (N, *spatial)   int32     # always (StarDist target source)
    /train/mask   (N, *spatial)   uint8     # only if binary_mask_dir was set
    /val/raw, /val/label, /val/mask?        # same shape

Both emit modes from the original SmartPatches:

1. **Foreground patches** — instance-centered, kept only if the
   foreground voxel fraction lies in
   ``[lower_ratio_fore_to_back, upper_ratio_fore_to_back]``.

2. **Background-paste augmentation** — additively blend a cell patch
   into a patch centered on a *background* voxel, so the cell silhouette
   appears against a different background context. Faithful port of
   upstream's ``_background_label_maker``.

Targets:
- StarDist reads ``label`` and derives ``(prob, dist)`` on the fly.
- U-Net reads ``mask`` if present (pre-computed binary; respects any
  user erosion / hole-filling); else derives binary from ``label``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
from collections.abc import Sequence

import h5py
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from tifffile import imread
from tqdm import tqdm

from .labels import erode_labels


_PatchShape = Union[tuple[int, int], tuple[int, int, int]]


# ----------------------------------------------------------------- public


def generate_smart_patches_h5(
    raw_dir: Union[str, Path],
    label_dir: Union[str, Path],
    output_h5: Union[str, Path],
    patch_shape: _PatchShape,
    *,
    binary_mask_dir: Union[str, Path, None] = None,
    val_files: int = 1,
    lower_ratio_fore_to_back: float = 0.05,
    upper_ratio_fore_to_back: float = 1.0,
    erosion_iterations: int = 0,
    max_foreground_patches_per_image: float = np.inf,
    paste_augmentation: bool = False,
    max_paste_patches_per_image: float = 0,
    seed: int = 42,
    extensions: Sequence[str] = (".tif", ".tiff", ".TIF", ".TIFF"),
    chunk_rows: int = 16,
    overwrite: bool = False,
    pmin: Optional[float] = 0.1,
    pmax: Optional[float] = 99.9,
) -> dict:
    """Build a unified training H5 containing raw + label (+ optional pre-computed mask).

    Parameters
    ----------
    raw_dir, label_dir
        Required. Integer instance label masks live in ``label_dir``;
        SmartPatches needs them for cell-centroid extraction and the
        background-paste augmentation.
    output_h5
        Destination H5 path.
    patch_shape
        Per-axis patch size; length 2 (2D) or 3 (3D).
    binary_mask_dir
        Optional. If set and a file with the matching basename exists
        there, its binary mask is stored alongside ``label`` as
        ``/train/mask`` (and ``/val/mask``) — the U-Net dataset will
        prefer this to deriving binary from ``label`` on the fly.
        Useful when you've pre-eroded / hole-filled and don't want the
        on-the-fly derivation to clobber that.
    val_files
        Last N files reserved for validation.
    lower_ratio_fore_to_back, upper_ratio_fore_to_back
        SmartPatches veto: a patch is kept only when its foreground
        voxel fraction lies in this interval.
    erosion_iterations
        Per-instance erosion applied to the in-patch label image *before*
        the binary mask is derived (when binary_mask_dir is None).
        Ignored for patches whose mask comes from the pre-computed
        binary_mask_dir.
    max_foreground_patches_per_image
        Per-image cap on instance-centered patches.
    paste_augmentation
        Enable background-paste augmentation (composite cells onto bg
        patches). Train-only; never applied to the val split.
    max_paste_patches_per_image
        Per-image cap on the paste-augmented patches.
    seed
        RNG seed for shuffling paste pairings.
    """
    raw_dir, label_dir, output_h5 = Path(raw_dir), Path(label_dir), Path(output_h5)
    binary_mask_dir = Path(binary_mask_dir) if binary_mask_dir else None
    ndim = len(patch_shape)
    if ndim not in (2, 3):
        raise ValueError(f"patch_shape must be length 2 or 3, got {patch_shape}")

    pairs = _list_paired(raw_dir, label_dir, extensions)
    if not pairs:
        raise FileNotFoundError(f"No paired files between {raw_dir} and {label_dir}")
    if val_files >= len(pairs):
        raise ValueError(f"val_files={val_files} but only {len(pairs)} files found")
    train_pairs, val_pairs = pairs[:-val_files], pairs[-val_files:]

    if output_h5.exists() and overwrite:
        output_h5.unlink()
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    counts = {"train_fg": 0, "train_paste": 0, "val_fg": 0}
    rng = np.random.default_rng(seed)
    has_mask_dir = binary_mask_dir is not None
    n_used_precomputed = 0

    with h5py.File(output_h5, "a") as h5:
        train_w = _Writer(
            h5, "train", patch_shape, has_mask=has_mask_dir, chunk_rows=chunk_rows
        )
        val_w = _Writer(
            h5, "val", patch_shape, has_mask=has_mask_dir, chunk_rows=chunk_rows
        )

        for raw_path, label_path in tqdm(
            train_pairs, desc=f"train ({len(train_pairs)})"
        ):
            binary_path = _binary_for(raw_path, binary_mask_dir)
            n_used_precomputed += int(binary_path is not None)
            fg, paste = _emit_from_file(
                raw_path,
                label_path,
                binary_path,
                train_w,
                patch_shape,
                ndim,
                lower_ratio_fore_to_back,
                upper_ratio_fore_to_back,
                erosion_iterations,
                max_foreground_patches_per_image,
                paste_augmentation,
                max_paste_patches_per_image,
                rng,
                pmin=pmin,
                pmax=pmax,
            )
            counts["train_fg"] += fg
            counts["train_paste"] += paste

        for raw_path, label_path in tqdm(val_pairs, desc=f"val ({len(val_pairs)})"):
            binary_path = _binary_for(raw_path, binary_mask_dir)
            n_used_precomputed += int(binary_path is not None)
            fg, _ = _emit_from_file(
                raw_path,
                label_path,
                binary_path,
                val_w,
                patch_shape,
                ndim,
                lower_ratio_fore_to_back,
                upper_ratio_fore_to_back,
                erosion_iterations,
                max_foreground_patches_per_image,
                paste_augmentation=False,
                max_paste_patches_per_image=0,
                rng=rng,
                pmin=pmin,
                pmax=pmax,
            )
            counts["val_fg"] += fg

    if has_mask_dir:
        print(
            f"Used pre-computed binary masks for {n_used_precomputed}/{len(pairs)} files "
            f"(any others fell back to deriving binary from labels)"
        )
    print(
        f"Train: {counts['train_fg']} fg + {counts['train_paste']} paste-aug | "
        f"Val: {counts['val_fg']} fg → {output_h5}"
    )
    return counts


def _binary_for(raw_path: Path, binary_mask_dir):
    if binary_mask_dir is None:
        return None
    candidate = binary_mask_dir / raw_path.name
    return candidate if candidate.exists() else None


# ----------------------------------------------------------------- helpers


def _list_paired(raw_dir, label_dir, exts):
    raws = sorted(p for p in raw_dir.iterdir() if p.suffix in exts)
    pairs = []
    for raw in raws:
        label_path = label_dir / raw.name
        if label_path.exists():
            pairs.append((raw, label_path))
    return pairs


def _region_around(center, patch_shape, vol_shape):
    slices = []
    for c, p, dim in zip(center, patch_shape, vol_shape):
        half = p // 2
        lo = int(c - half)
        hi = int(c + half)
        if lo < 0 or hi > dim:
            return None
        slices.append(slice(lo, hi))
    return tuple(slices)


def _emit_from_file(
    raw_path,
    label_path,
    binary_path,
    writer,
    patch_shape,
    ndim,
    lo_ratio,
    hi_ratio,
    erosion_iterations,
    max_fg,
    paste_augmentation,
    max_paste_patches_per_image,
    rng,
    *,
    pmin: Optional[float] = None,
    pmax: Optional[float] = None,
) -> tuple[int, int]:
    raw = imread(raw_path).astype(np.float32)
    # Percentile-normalise the WHOLE raw volume BEFORE patch extraction
    # (CARE-style data prep). Patches inherit ``[0, 1]`` values from
    # the already-normalised volume, so train-time / val-time / inference
    # all see the same distribution: a whole-volume percentile of the
    # raw image. Targets (instance labels / binary masks) are NOT
    # normalised — they stay int / uint as written.
    if pmin is not None and pmax is not None:
        lo = float(np.percentile(raw, pmin))
        hi = float(np.percentile(raw, pmax))
        raw = ((raw - lo) / (hi - lo + 1e-8)).clip(0.0, 1.0).astype(np.float32)
    labels = imread(label_path).astype(np.int32)
    if raw.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch in {raw_path.name}: {raw.shape} vs {labels.shape}"
        )
    if raw.ndim != ndim:
        raise ValueError(f"{raw_path.name}: expected ndim={ndim}, got {raw.ndim}")

    binary = None
    if binary_path is not None:
        binary = (imread(binary_path) > 0).astype(np.uint8)
        if binary.shape != labels.shape:
            raise ValueError(
                f"binary_mask {binary_path.name} shape {binary.shape} doesn't match "
                f"labels {labels.shape}"
            )

    n_fg = _emit_foreground(
        raw,
        labels,
        binary,
        writer,
        patch_shape,
        lo_ratio,
        hi_ratio,
        erosion_iterations,
        max_fg,
    )
    n_paste = 0
    if paste_augmentation and max_paste_patches_per_image > 0:
        n_paste = _emit_paste_augmentation(
            raw,
            labels,
            binary,
            writer,
            patch_shape,
            max_paste_patches_per_image,
            rng,
        )
    return n_fg, n_paste


def _emit_foreground(
    raw,
    labels,
    binary,
    writer,
    patch_shape,
    lo_ratio,
    hi_ratio,
    erosion_iterations,
    max_fg,
) -> int:
    n = 0
    for prop in regionprops(labels):
        if n >= max_fg:
            break
        region = _region_around(prop.centroid, patch_shape, raw.shape)
        if region is None:
            continue
        patch_labels = remove_small_objects(
            labels[region].astype("uint16"), min_size=10
        )
        if patch_labels.shape != tuple(patch_shape):
            continue
        ratio = float(np.count_nonzero(patch_labels)) / patch_labels.size
        if not (lo_ratio <= ratio <= hi_ratio):
            continue
        if erosion_iterations > 0:
            patch_labels = erode_labels(patch_labels, erosion_iterations).astype(
                "uint16"
            )

        mask_patch = (
            binary[region].astype(np.uint8)
            if binary is not None
            else (patch_labels > 0).astype(np.uint8)
        )
        writer.append(raw[region], patch_labels.astype(np.int32), mask_patch)
        n += 1
    return n


def _emit_paste_augmentation(
    raw,
    labels,
    binary,
    writer,
    patch_shape,
    max_count,
    rng,
) -> int:
    """Composite cells onto patches centered on background voxels.

    Uses **rejection sampling** rather than enumerating every background
    voxel — for a large 3D volume with ~100M voxels of background, the
    naive ``np.argwhere(labels == 0)`` materializes a ~2 GB index array
    and shuffling it is O(N log N). Rejection sampling is O(max_count).

    For each iteration we pick a uniformly random voxel; if it's
    background, take a patch around it (which must also be all-zero in
    the label image), pair it with a random cell, blend, emit. We
    bail out after ``max_count * 200`` attempts to guarantee termination
    even on volumes that are mostly foreground.
    """
    if not _has_any_background(labels):
        return 0
    props = list(regionprops(labels))
    if not props:
        return 0

    n_props = len(props)
    max_attempts = max(int(max_count) * 200, 1000)
    count = 0
    attempts = 0
    shape = labels.shape

    while count < max_count and attempts < max_attempts:
        attempts += 1
        bg_idx = tuple(int(rng.integers(0, s)) for s in shape)
        if labels[bg_idx] != 0:
            continue
        bg_region = _region_around(bg_idx, patch_shape, raw.shape)
        if bg_region is None:
            continue
        label_bg_patch = labels[bg_region]
        if label_bg_patch.sum() != 0:  # patch must be all-zero
            continue

        prop = props[int(rng.integers(0, n_props))]
        fg_region = _region_around(prop.centroid, patch_shape, raw.shape)
        if fg_region is None:
            continue

        raw_aug = raw[bg_region] + raw[fg_region]
        label_aug = labels[fg_region].astype(np.int32)  # label_bg is 0
        if raw_aug.sum() <= 0:
            continue

        if binary is not None:
            mask_aug = (binary[bg_region] | binary[fg_region]).astype(np.uint8)
        else:
            mask_aug = (label_aug > 0).astype(np.uint8)

        writer.append(raw_aug, label_aug, mask_aug)
        count += 1
    return count


def _has_any_background(labels) -> bool:
    """Cheap check: does the volume contain at least one zero voxel?

    Faster than ``np.argwhere(labels == 0)`` because we short-circuit on
    first hit; still O(N) worst case but with small constants.
    """
    return bool((labels == 0).any())


class _Writer:
    """Resizable-H5 writer for `raw` + `label` (+ optional `mask`).

    All three datasets share the same first axis indexing so a single
    sample index ``i`` yields the matching raw / label / mask triple.
    """

    def __init__(self, h5, split, patch_shape, *, has_mask, chunk_rows):
        if split in h5:
            del h5[split]
        grp = h5.create_group(split)
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
        self.mask = None
        if has_mask:
            self.mask = grp.create_dataset(
                "mask",
                shape=(0, *patch_shape),
                maxshape=(None, *patch_shape),
                dtype="uint8",
                chunks=(chunk_rows, *patch_shape),
                compression="lzf",
            )

    def append(self, raw_patch, label_patch, mask_patch):
        n = self.raw.shape[0]
        self.raw.resize(n + 1, axis=0)
        self.label.resize(n + 1, axis=0)
        self.raw[n] = raw_patch
        self.label[n] = label_patch
        if self.mask is not None:
            self.mask.resize(n + 1, axis=0)
            self.mask[n] = mask_patch
