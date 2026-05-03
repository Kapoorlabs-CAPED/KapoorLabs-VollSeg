"""Layer 1 — singleton inference models.

Each class wraps a single trained backbone and provides one job:

- :class:`CAREDenoiser` — denoise an image
- :class:`UNetSegmenter` — semantic segmentation (binary mask + CC labels)
- :class:`StarDistSegmenter` — instance segmentation via radial distances

All three implement :class:`vollseg.pipelines.Pipeline` so they can be
composed by the Layer 2 wrappers without further adaptation.
"""

from .care import CAREDenoiser
from .maskunet import MaskUNetSegmenter
from .stardist import StarDistSegmenter
from .unet import UNetSegmenter

__all__ = ["CAREDenoiser", "UNetSegmenter", "MaskUNetSegmenter", "StarDistSegmenter"]
