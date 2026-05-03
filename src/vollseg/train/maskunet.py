"""MaskUNet trainer — first-class PyTorch implementation.

Same shape and defaults as :class:`UNetTrainer`. Kept separate for
naming / checkpoint discipline.
"""

from __future__ import annotations

from .unet import UNetTrainer


class MaskUNetTrainer(UNetTrainer):
    pass
