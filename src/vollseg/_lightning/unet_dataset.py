"""H5-backed paired-patch Dataset for U-Net training.

Reads from the unified SmartPatches H5 produced by
:func:`vollseg.data.generate_smart_patches_h5`. The H5 always contains
``raw`` + ``label``; ``mask`` is optional. The dataset prefers ``mask``
if present (respects user pre-computed binarization) and falls back to
deriving binary from ``label`` on the fly.
"""

from __future__ import annotations

from typing import Callable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5UNetDataset(Dataset):
    """Yield ``(raw, mask)`` patches from one split of a UNet H5.

    Parameters
    ----------
    h5_file
        Path to an H5 produced by
        :func:`vollseg.data.smart_patches_h5.generate_unet_h5`.
    split
        ``"train"`` or ``"val"``.
    transform
        Optional callable ``(raw, mask) -> (raw, mask)``. Apply
        normalization / augmentation here. Both tensors retain their
        per-axis shape; the channel dim is added by the collate / model.
    """

    def __init__(
        self,
        h5_file,
        split: str,
        *,
        transform: Optional[
            Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ):
        self.h5_file = str(h5_file)
        self.split = split
        self.transform = transform

        with h5py.File(self.h5_file, "r", swmr=True) as f:
            grp = f[split]
            self._len = grp["raw"].shape[0]
            self._patch_shape = grp["raw"].shape[1:]
            self._has_mask = "mask" in grp
        self._h5 = None  # lazy-open in worker

    def __len__(self) -> int:
        return self._len

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_file, "r", swmr=True)
        return self._h5

    def __getitem__(self, idx: int):
        grp = self._open()[self.split]
        raw = torch.from_numpy(grp["raw"][idx].astype(np.float32))
        if self._has_mask:
            mask = torch.from_numpy((grp["mask"][idx] > 0).astype(np.float32))
        else:
            # No pre-computed binary in H5 — derive on the fly from labels.
            mask = torch.from_numpy((grp["label"][idx] > 0).astype(np.float32))
        if self.transform is not None:
            raw, mask = self.transform(raw, mask)
        return raw, mask

    @property
    def patch_shape(self):
        return tuple(self._patch_shape)

    def __del__(self):
        if getattr(self, "_h5", None) is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def unet_collate(batch):
    """Stack ``(raw, mask)`` triples and add the channel dim."""
    raws, masks = zip(*batch)
    raw = torch.stack(raws, dim=0).unsqueeze(1)  # (B, 1, *spatial)
    mask = torch.stack(masks, dim=0).unsqueeze(1)  # (B, 1, *spatial)
    return raw, mask
