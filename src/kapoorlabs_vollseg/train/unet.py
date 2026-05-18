"""U-Net trainer — first-class PyTorch implementation.

Thin alias over :class:`CARETrainer` so the training script reads
``UNetTrainer`` (intent: semantic segmentation), but the underlying
shape and loss match the kapoorlabs-lightning ``CareInception`` path
that the ROI / nuclei / membrane U-Net checkpoints were trained with
— ``nn.MSELoss`` by default. Override via ``loss_func=`` if you want
BCE-with-logits.
"""

from __future__ import annotations

from .care import CARETrainer


class UNetTrainer(CARETrainer):
    """Train a U-Net for binary semantic segmentation.

    Inherits the MSE default loss from :class:`CARETrainer`, matching
    upstream ``CareInception``. Pass ``loss_func=nn.BCEWithLogitsLoss()``
    explicitly if you want logits-style binary cross-entropy.
    """
