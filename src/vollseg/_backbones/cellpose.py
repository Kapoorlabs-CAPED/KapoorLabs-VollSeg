"""CellPose backbone — thin adapter around ``cellpose.models.CellposeModel``.

Cellpose's API is a single class that does both training and inference,
so this backbone is mostly a constructor that picks the right of three
supported sources of weights:

- ``model_path`` — a local checkpoint produced by training
- ``model_type`` — a name accepted by ``CellposeModel`` (e.g. ``"cyto3"``)
- ``model_dir`` + ``model_name`` — a checkpoint by directory + filename
"""

from __future__ import annotations

import os
from typing import Optional


class CellPoseBackbone:
    """Wrap a ``cellpose.models.CellposeModel`` instance.

    Parameters
    ----------
    model_path
        Absolute path to a trained cellpose checkpoint. Wins over
        ``model_type`` and ``model_dir`` + ``model_name``.
    model_type
        Built-in cellpose model name (``cyto3``, ``nuclei``, etc.).
    model_dir, model_name
        Alternative way to point at a local checkpoint.
    gpu
        Pass through to cellpose.
    """

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        model_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        gpu: bool = False,
    ):
        from cellpose import models  # local import — heavy dep

        if model_path is None and model_dir is not None and model_name is not None:
            model_path = os.path.join(model_dir, model_name)

        if model_path is None and model_type is None:
            raise ValueError(
                "Provide one of: model_path, model_type, or (model_dir + model_name)."
            )

        if model_path is not None:
            self.model = models.CellposeModel(gpu=gpu, pretrained_model=model_path)
        else:
            self.model = models.CellposeModel(gpu=gpu, model_type=model_type)

        self.model_path = model_path
        self.model_type = model_type
        self.gpu = gpu
