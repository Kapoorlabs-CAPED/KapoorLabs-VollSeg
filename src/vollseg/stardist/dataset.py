"""H5-backed Dataset for StarDist training.

Reads ``(raw, label)`` pairs from H5, applies any user transform to the
pair, then **computes the ``(prob, dist)`` targets on the fly from the
(possibly augmented) label patch**. This matches upstream stardist's
training pattern and lets any geometric augmentation work in any ndim
— the targets are derived from the augmented labels, not permuted from
precomputed channels.

Workers parallelize the per-sample target compute, so the additional
ray-march cost (≈hundreds of ms per sample with numba) overlaps with
the GPU step. ``num_workers >= 2`` is recommended at training time.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .distance import compute_distance_map, foreground_probability_map


class StarDistH5Dataset(Dataset):
    """Yield ``(raw, prob_target, dist_target)`` from an H5 of ``(raw, label)``.

    Parameters
    ----------
    h5_file
        Path to an H5 produced by :func:`generate_stardist_h5`.
    split
        ``"train"`` or ``"val"``.
    rays
        ``(n_rays, ndim)`` array — same one used at prediction time.
        The dataset uses it to compute the dist target after augmentation.
    transform
        Optional callable ``(raw, label) -> (raw, label)``. Apply here
        any flip / rotation / intensity augmentation; targets are derived
        from the post-transform label, so geometric augmentation in any
        ndim is safe.
    """

    def __init__(
        self,
        h5_file,
        split: str,
        *,
        rays: np.ndarray,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor],
                                      Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        if rays.ndim != 2:
            raise ValueError(f"rays must be 2D (N, ndim), got shape {rays.shape}")
        self.h5_file = str(h5_file)
        self.split = split
        self.rays = np.ascontiguousarray(rays, dtype=np.float32)
        self.transform = transform

        with h5py.File(self.h5_file, "r", swmr=True) as f:
            grp = f[split]
            self._len = grp["raw"].shape[0]
            self._patch_shape = grp["raw"].shape[1:]
        self._h5 = None  # opened on first access in this worker process

    def __len__(self) -> int:
        return self._len

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_file, "r", swmr=True)
        return self._h5

    def __getitem__(self, idx: int):
        grp = self._open()[self.split]
        raw = torch.from_numpy(grp["raw"][idx].astype(np.float32))
        label = torch.from_numpy(grp["label"][idx].astype(np.int32))

        if self.transform is not None:
            raw, label = self.transform(raw, label)

        # Derive targets from the (possibly augmented) label.
        label_np = np.ascontiguousarray(label.numpy(), dtype=np.int32)
        prob = torch.from_numpy(foreground_probability_map(label_np)).unsqueeze(0)
        dist = torch.from_numpy(compute_distance_map(label_np, self.rays))
        return raw, prob, dist

    @property
    def patch_shape(self):
        return tuple(self._patch_shape)

    @property
    def n_rays(self) -> int:
        return int(self.rays.shape[0])

    def __del__(self):
        if getattr(self, "_h5", None) is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def stardist_collate(batch):
    """Stack ``(raw, prob, dist)`` triples and add the channel dim to raw."""
    raws, probs, dists = zip(*batch)
    raw = torch.stack(raws, dim=0).unsqueeze(1)   # (B, 1, *spatial)
    prob = torch.stack(probs, dim=0)              # (B, 1, *spatial)
    dist = torch.stack(dists, dim=0)              # (B, N, *spatial)
    return raw, prob, dist
