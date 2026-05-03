"""Training harnesses — separated from inference singletons.

Each trainer's job is to *produce* a backbone (saved on disk under a
chosen ``model_dir/model_name``). The trained backbone can then be
wrapped in the corresponding :mod:`vollseg.models` singleton for
inference, or registered as a pretrained model.
"""

from .care import CARETrainer
from .smartseeds import SmartSeeds
from .stardist import StarDistTrainer
from .unet import UNetTrainer

__all__ = ["CARETrainer", "UNetTrainer", "StarDistTrainer", "SmartSeeds"]
