"""PyTorch MaskUNet backbone — same careamics UNet, separate name for separate weights.

Architecturally identical to :class:`UNetBackbone` (same single-output
binary segmentation shape); kept as its own class only so Lightning
checkpoints stay typed by *intent* (mask vs. semantic). The Layer-1
:class:`vollseg.MaskUNetSegmenter` post-processes the same way as the
plain U-Net.
"""

from __future__ import annotations

from .unet import UNetBackbone


class MaskUNetBackbone(UNetBackbone):
    """Alias of :class:`UNetBackbone` — exists for naming discipline."""
    pass
