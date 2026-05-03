"""Plain CellPose membrane segmentation (no nuclei seeding).

Replaces ``CopenhagenWorkflow/01_membrane_segmentation_normal_cellpose.py``.
Uses the prior :mod:`01_enhance_membrane` outputs as input when present;
falls back to the raw membrane channel otherwise.

The pipeline is just :class:`CellPoseSegmenter` — :class:`VollCellSeg.from_models`
collapses to that automatically when no ``nuclei_pipeline`` is provided.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from natsort import natsorted
from tifffile import imread, imwrite
from tqdm import tqdm

from vollseg import CellPoseBackbone, CellPoseSegmenter, VollCellSeg

from scenarios import SegmentScenario


ConfigStore.instance().store(name="SegmentScenario", node=SegmentScenario)


def _build_cellpose(config: SegmentScenario) -> CellPoseSegmenter:
    mp = config.model_paths
    backbone = CellPoseBackbone(
        model_dir=mp.cellpose_model_dir if mp.cellpose_membrane_model_name else None,
        model_name=mp.cellpose_membrane_model_name,
        model_type=mp.cellpose_membrane_model_type,
        gpu=config.parameters.cellpose_gpu,
    )
    p = config.parameters
    return CellPoseSegmenter(
        backbone,
        diameter=p.cellpose_diameter,
        flow_threshold=p.cellpose_flow_threshold,
        cellprob_threshold=p.cellpose_cellprob_threshold,
        stitch_threshold=p.cellpose_stitch_threshold,
        anisotropy=p.cellpose_anisotropy,
        channels=p.cellpose_channels,
        bsize=p.cellpose_bsize,
    )


@hydra.main(version_base="1.3", config_path="conf", config_name="scenario_segment")
def main(config: SegmentScenario) -> None:
    base = Path(config.experiment_data_paths.base_directory)
    membrane_dir = base / config.experiment_data_paths.timelapse_membrane_directory
    enhanced_dir = base / config.experiment_data_paths.membrane_enhanced_directory
    seg_dir = base / config.experiment_data_paths.timelapse_seg_membrane_directory
    seg_dir.mkdir(parents=True, exist_ok=True)

    pipeline = VollCellSeg.from_models(cellpose=_build_cellpose(config))

    axes = config.parameters.axes
    channel_membrane = config.parameters.channel_membrane
    files = natsorted(glob.glob(os.fspath(membrane_dir / config.parameters.file_type)))

    for fname in tqdm(files, desc="cellpose membrane"):
        name = Path(fname).stem
        out = seg_dir / f"{name}.tif"
        if out.exists():
            continue

        # Prefer denoised input if 01_enhance_membrane has run.
        denoised = enhanced_dir / f"{name}.tif"
        if denoised.exists():
            membrane = imread(denoised)
        else:
            image = imread(fname)
            membrane = image[:, channel_membrane, :, :] if image.ndim == 4 else image

        result = pipeline.predict(membrane, axes=axes)
        imwrite(out, np.asarray(result.labels).astype(np.uint16))


if __name__ == "__main__":
    main()
