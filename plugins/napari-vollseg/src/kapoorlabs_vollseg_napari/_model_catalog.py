"""
Group :data:`kapoorlabs_vollseg.hub.XENOPUS_MODELS` by their role.

Each dropdown in the Models tab is fed from one entry of
:data:`MODEL_CATALOG`. The categorisation is by naming convention on
the HuggingFace repo IDs (``xenopus-care-...``, ``xenopus-unet3d-...``
etc.), which is more robust than depending on the SDK's flat dict.

Roles correspond exactly to the singleton classes:

- ``care``      — :class:`CAREDenoiser`
- ``unet``      — :class:`UNetSegmenter`
- ``maskunet``  — :class:`MaskUNetSegmenter`
- ``stardist``  — :class:`StarDistSegmenter`
- ``cellpose``  — :class:`CellPoseSegmenter`
"""

from __future__ import annotations

from kapoorlabs_vollseg.hub import XENOPUS_MODELS


def _classify(model_name: str, repo_id: str) -> str:
    rid = repo_id.lower()
    name = model_name.lower()
    if "-care-" in rid or name.startswith("care_"):
        return "care"
    if "-maskunet-" in rid or name.startswith("unet_roi_"):
        return "maskunet"
    if "-unet3d-" in rid or name.startswith("unet_"):
        return "unet"
    if "-stardist" in rid:
        return "stardist"
    if "-cellpose-" in rid or "cellpose" in name:
        return "cellpose"
    return "unknown"


def _build_catalog() -> dict[str, list[str]]:
    catalog: dict[str, list[str]] = {
        "care": [],
        "unet": [],
        "maskunet": [],
        "stardist": [],
        "cellpose": [],
    }
    for name, repo in XENOPUS_MODELS.items():
        role = _classify(name, repo)
        if role in catalog:
            catalog[role].append(name)
    for role in catalog:
        catalog[role].sort()
    return catalog


MODEL_CATALOG: dict[str, list[str]] = _build_catalog()


ROLE_LABELS: dict[str, str] = {
    "care": "CARE denoiser",
    "unet": "U-Net (semantic)",
    "maskunet": "ROI Mask-UNet",
    "stardist": "StarDist (instance)",
    "cellpose": "CellPose (membrane)",
}
