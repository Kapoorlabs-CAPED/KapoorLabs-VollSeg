"""Glue layer: turn user selections in the dock widget into a
:class:`kapoorlabs_vollseg.Pipeline` and run ``.predict`` on the image.

Designed for execution inside a :func:`napari.qt.thread_worker` so the
GUI stays responsive while the model downloads / loads / inferences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from kapoorlabs_vollseg import (
    CAREDenoiser,
    CellPoseBackbone,
    CellPoseSegmenter,
    MaskUNetSegmenter,
    StarDistSegmenter,
    UNetSegmenter,
    VollCellSeg,
    VollSeg,
    ensure_cellpose_checkpoint,
    ensure_model,
)


# ============================================================ role choice


@dataclass
class RoleChoice:
    """Per-role tri-state mirroring the original ``RadioButtons``.

    ``mode`` is one of ``"none"`` / ``"pretrained"`` / ``"custom"``.
    Only the relevant field of ``pretrained_name`` / ``custom_path`` is
    consulted, based on ``mode``.
    """

    mode: str = "none"  # "none" | "pretrained" | "custom"
    pretrained_name: str = ""  # HuggingFace registry name
    custom_path: Optional[Path] = None  # local checkpoint folder

    @property
    def active(self) -> bool:
        return self.mode != "none"

    @property
    def is_pretrained(self) -> bool:
        return self.mode == "pretrained"

    @property
    def is_custom(self) -> bool:
        return self.mode == "custom"


@dataclass
class RunSpec:
    """Everything the widget tabs collect, in one bundle."""

    # Input
    image: np.ndarray
    voxel_spacing: tuple[float, ...] = (1.0, 1.0, 1.0)

    # Model picks (one RoleChoice per role)
    care: RoleChoice = field(default_factory=RoleChoice)
    unet: RoleChoice = field(default_factory=RoleChoice)
    maskunet: RoleChoice = field(default_factory=RoleChoice)
    stardist: RoleChoice = field(default_factory=RoleChoice)
    cellpose: RoleChoice = field(default_factory=RoleChoice)

    # Local model cache root (used only when a role is in "pretrained" mode)
    model_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "kapoorlabs-vollseg"
    )

    # Inference knobs
    n_rays: int = 96
    prob_thresh: Optional[float] = None
    nms_thresh: Optional[float] = None
    n_tiles: tuple[int, int, int] = (1, 1, 1)
    seedpool: bool = False
    chunk: Optional[tuple[int, int, int]] = None
    overlap: tuple[int, int, int] = (0, 0, 0)

    # Pipeline shape
    membrane_mode: bool = False  # True → VollCellSeg.from_models


# ============================================================ loaders


def _resolve_folder(choice: RoleChoice, model_dir: Path) -> Path:
    """Return the local folder containing the requested role's weights."""
    if choice.is_pretrained:
        return ensure_model(model_dir, choice.pretrained_name)
    if choice.is_custom:
        if choice.custom_path is None:
            raise ValueError("Custom mode selected but no path provided.")
        return Path(choice.custom_path)
    raise ValueError("RoleChoice.mode is 'none' — caller should not load.")


# The Lightning singletons each expose ``from_folder(folder)``: it
# finds the .ckpt, reads ``training_config.json`` (or the smaller
# ``{experiment_name}.json`` fallback) for arch knobs, and falls back
# to state-dict inference for anything missing — so the plugin only
# has to hand over the folder.


def load_care(choice: RoleChoice, model_dir: Path) -> CAREDenoiser:
    return CAREDenoiser.from_folder(_resolve_folder(choice, model_dir))


def load_unet(choice: RoleChoice, model_dir: Path) -> UNetSegmenter:
    return UNetSegmenter.from_folder(_resolve_folder(choice, model_dir))


def load_maskunet(choice: RoleChoice, model_dir: Path) -> MaskUNetSegmenter:
    return MaskUNetSegmenter.from_folder(_resolve_folder(choice, model_dir))


def load_stardist(
    choice: RoleChoice, model_dir: Path, n_rays: int
) -> StarDistSegmenter:
    return StarDistSegmenter.from_folder(
        _resolve_folder(choice, model_dir),
        n_rays=n_rays,
    )


def load_cellpose(choice: RoleChoice, model_dir: Path) -> CellPoseSegmenter:
    if choice.is_pretrained:
        ckpt = ensure_cellpose_checkpoint(model_dir, choice.pretrained_name)
    else:
        # Custom mode: ``custom_path`` is the checkpoint file (or its folder).
        p = Path(choice.custom_path)
        ckpt = p if p.is_file() else next(p.rglob("*.pth"), None)
        if ckpt is None:
            raise FileNotFoundError(f"No CellPose .pth inside {p}")
    return CellPoseSegmenter(CellPoseBackbone.from_checkpoint(ckpt))


# ====================================================== pipeline build


def build_pipeline(spec: RunSpec):
    """Compose Layer-1 singletons → Layer-2 composites → Layer-3 factory."""
    care = load_care(spec.care, spec.model_dir) if spec.care.active else None
    unet = load_unet(spec.unet, spec.model_dir) if spec.unet.active else None
    maskunet = (
        load_maskunet(spec.maskunet, spec.model_dir) if spec.maskunet.active else None
    )
    stardist = (
        load_stardist(spec.stardist, spec.model_dir, spec.n_rays)
        if spec.stardist.active
        else None
    )
    cellpose = (
        load_cellpose(spec.cellpose, spec.model_dir) if spec.cellpose.active else None
    )

    if spec.membrane_mode:
        if cellpose is None:
            raise ValueError("Membrane mode requires a CellPose model.")
        # Build a nuclei sub-pipeline (StarDist ± U-Net) to seed CellPose.
        any_nuclei = any(m is not None for m in (unet, stardist, maskunet, care))
        nuclei_pipe = (
            VollSeg.from_models(
                care=care,
                unet=unet,
                stardist=stardist,
                roi_unet=maskunet,
                seedpool=spec.seedpool,
                chunk=spec.chunk,
                overlap=spec.overlap,
            )
            if any_nuclei
            else None
        )
        return VollCellSeg.from_models(
            nuclei_pipeline=nuclei_pipe,
            cellpose=cellpose,
            care=None,  # CARE already inside nuclei_pipe if requested
        )

    return VollSeg.from_models(
        care=care,
        unet=unet,
        stardist=stardist,
        roi_unet=maskunet,
        seedpool=spec.seedpool,
        chunk=spec.chunk,
        overlap=spec.overlap,
    )


def run(spec: RunSpec) -> dict[str, Any]:
    """Build the pipeline, run it, return a dict of result layers.

    Keys mirror :class:`kapoorlabs_vollseg.Result` attributes that are
    actually populated, so the caller can iterate and add a napari
    layer per non-``None`` field.
    """
    pipeline = build_pipeline(spec)
    result = pipeline.predict(
        spec.image,
        prob_thresh=spec.prob_thresh,
        nms_thresh=spec.nms_thresh,
        n_tiles=spec.n_tiles,
    )

    layers: dict[str, Any] = {}
    for field_name in ("denoised", "labels", "semantic", "probability"):
        value = getattr(result, field_name, None)
        if value is not None:
            layers[field_name] = value
    return layers
