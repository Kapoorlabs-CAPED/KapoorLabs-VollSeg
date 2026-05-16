"""Napari plugin for the KapoorLabs-VollSeg PyTorch segmentation SDK.

PyTorch-only — the legacy keras stack is intentionally not exposed.
Pretrained weights are pulled from HuggingFace via
:mod:`kapoorlabs_vollseg.hub`; each model dropdown is populated from
the :data:`MODEL_CATALOG` grouping of :data:`hub.XENOPUS_MODELS`.
"""

from ._model_catalog import MODEL_CATALOG
from ._widget import VollSegWidget

__all__ = ["VollSegWidget", "MODEL_CATALOG"]
