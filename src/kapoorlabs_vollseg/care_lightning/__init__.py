"""Self-contained port of the CARE / ROI training + prediction stack
from ``kapoorlabs_lightning``. Owns the LightningModule under which the
``xenopus_edge_enhancement`` checkpoint was trained, so ``.ckpt`` loading
stays byte-compatible.
"""

from .module import CareModule, stitch_tiles
from .dataset import CarePredictionDataset, H5CareDataset, compute_tile_shape
from .trainer import CareInception

__all__ = [
    "CareModule",
    "stitch_tiles",
    "CarePredictionDataset",
    "H5CareDataset",
    "compute_tile_shape",
    "CareInception",
]
