"""CellPose trainer.

CellPose's library bundles training and inference into one class, but the
training side has its own coordinate system (per-image lists, optional
2D-slicing of 3D volumes). Keeping it in this trainer leaves the Layer 1
:class:`CellPoseSegmenter` clean.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from tifffile import imread, imwrite

from .._backbones.cellpose import CellPoseBackbone


class CellPoseTrainer:
    """Train (or fine-tune) a CellPose model.

    Parameters
    ----------
    model_dir, model_name
        Output checkpoint location.
    pretrained_path
        Optional starting checkpoint to fine-tune. ``None`` trains from
        the cellpose default initialization.
    diam_mean
        Mean object diameter (pixels) — must match the training data.
    real_train_3D
        If False, 3D training volumes are sliced into 2D Z-slices first
        (matching the original VollSeg ``CellPose`` class behavior).
    """

    def __init__(
        self,
        *,
        model_dir: Union[str, Path],
        model_name: str,
        pretrained_path: Optional[str] = None,
        n_epochs: int = 400,
        diam_mean: float = 30.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        channels: Sequence[int] = (0, 0),
        min_train_masks: int = 1,
        gpu: bool = True,
        real_train_3D: bool = False,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.pretrained_path = pretrained_path
        self.n_epochs = n_epochs
        self.diam_mean = diam_mean
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.channels = list(channels)
        self.min_train_masks = min_train_masks
        self.gpu = gpu
        self.real_train_3D = real_train_3D

    # ------------------------------------------------------------ load

    def load_dataset(
        self,
        raw_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        *,
        save_sliced_dir: Optional[Path] = None,
    ):
        """Read ``(raw_dir, mask_dir)`` and return matched ``(images, labels, names)``.

        For 3D inputs with ``real_train_3D=False``, each Z-slice with at
        least one labeled object becomes its own training sample.
        """
        raw_dir, mask_dir = Path(raw_dir), Path(mask_dir)
        images, labels, names = [], [], []

        for fname in sorted(os.listdir(mask_dir)):
            if not _is_image(fname):
                continue
            name = Path(fname).stem
            label_img = imread(mask_dir / fname).astype(np.uint16)
            raw_img = imread(raw_dir / fname)

            if not self.real_train_3D and label_img.ndim == 3:
                for i in range(label_img.shape[0]):
                    if label_img[i].max() > 0:
                        images.append(raw_img[i])
                        labels.append(label_img[i])
                        names.append(f"{name}_{i}")
            else:
                images.append(raw_img)
                labels.append(label_img)
                names.append(name)

        if save_sliced_dir is not None:
            save_sliced_dir.mkdir(parents=True, exist_ok=True)
            for img, lbl, n in zip(images, labels, names):
                imwrite(save_sliced_dir / f"{n}_raw.tif", img.astype(np.float32))
                imwrite(save_sliced_dir / f"{n}_mask.tif", lbl.astype(np.uint16))

        return images, labels, names

    # ------------------------------------------------------------- fit

    def fit(self, train_images, train_labels, *, test_images=None, test_labels=None):
        """Run cellpose training. Returns the new checkpoint path + backbone."""
        backbone = CellPoseBackbone(
            model_path=self.pretrained_path,
            gpu=self.gpu,
        ) if self.pretrained_path else CellPoseBackbone(
            model_type="cyto3", gpu=self.gpu
        )
        # cellpose >=3 changed the model's diam_mean handling; respect ours.
        backbone.model.diam_mean = self.diam_mean

        new_path = backbone.model.train(
            train_images,
            train_labels,
            test_data=test_images,
            test_labels=test_labels,
            save_path=os.fspath(self.model_dir),
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            channels=self.channels,
            weight_decay=self.weight_decay,
            model_name=self.model_name,
            min_train_masks=self.min_train_masks,
        )
        return new_path, backbone


def _is_image(fname: str) -> bool:
    return any(fname.endswith(ext) for ext in (".tif", ".TIFF", ".TIF", ".png"))
