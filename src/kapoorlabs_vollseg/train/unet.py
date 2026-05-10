"""U-Net trainer — first-class PyTorch implementation.

Same shape as :class:`CARETrainer`: build a careamics UNet, wrap in
:class:`CareModule`, hand to ``lightning.Trainer``. The only meaningful
difference is the loss (BCE-with-logits by default, since this is binary
segmentation rather than regression).
"""

from __future__ import annotations

import torch.nn as nn

from .care import CARETrainer


class UNetTrainer(CARETrainer):
    """Train a U-Net for binary semantic segmentation."""

    def __init__(self, **kwargs):
        kwargs.setdefault("loss_func", nn.BCEWithLogitsLoss())
        super().__init__(**kwargs)
