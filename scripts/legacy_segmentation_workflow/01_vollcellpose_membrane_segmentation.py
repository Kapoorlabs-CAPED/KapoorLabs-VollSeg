"""Full VollCellSeg: nuclei pipeline + CellPose + nuclei-seeded watershed.

Replaces ``CopenhagenWorkflow/01_vollcellpose_membrane_segmentation.py``.
The original chained: pre-saved nuclei seg → pre-saved CellPose mask →
denoised membrane → ``VollCellSeg(...)`` (which is the
``cellpose_watershed_fuse`` watershed).

The new version assembles the same chain through
:func:`kapoorlabs_vollseg.VollCellSeg.from_models`, so the data dependencies are
expressed as Pipeline composition rather than file shuffling.

Inputs that *must* exist on disk first:
- denoised membrane:    ``membrane_enhanced/{name}.tif`` (from 01_enhance_membrane)
- nuclei labels:        ``seg_nuclei_timelapses/{name}_{t}.tif`` (from 01_nuclei_segmentation)

The script feeds these through :meth:`NucleiSeededCellPosePipeline.predict_split`
so it can re-use the cached nuclei labels rather than re-segmenting.
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

from kapoorlabs_vollseg import (
    CellPoseBackbone,
    CellPoseSegmenter,
    cellpose_watershed_fuse,
    ensure_cellpose_checkpoint,
)

from scenarios import SegmentScenario


ConfigStore.instance().store(name="SegmentScenario", node=SegmentScenario)


def _build_cellpose(config: SegmentScenario) -> CellPoseSegmenter:
    mp = config.model_paths
    if mp.cellpose_membrane_model_name:
        ckpt = ensure_cellpose_checkpoint(
            mp.cellpose_model_dir, mp.cellpose_membrane_model_name
        )
        backbone = CellPoseBackbone(
            model_path=str(ckpt),
            gpu=config.parameters.cellpose_gpu,
        )
    else:
        backbone = CellPoseBackbone(
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


@hydra.main(version_base="1.3", config_path="../conf", config_name="scenario_segment")
def main(config: SegmentScenario) -> None:
    base = Path(config.experiment_data_paths.base_directory)
    enhanced_dir = base / config.experiment_data_paths.membrane_enhanced_directory
    nuclei_seg_dir = base / config.experiment_data_paths.timelapse_seg_nuclei_directory
    out_dir = base / config.experiment_data_paths.timelapse_seg_vollcell_directory
    out_dir.mkdir(parents=True, exist_ok=True)

    cellpose = _build_cellpose(config)
    axes = config.parameters.axes

    files = natsorted(glob.glob(os.fspath(enhanced_dir / config.parameters.file_type)))

    for fname in tqdm(files, desc="vollcellpose"):
        name = Path(fname).stem
        out = out_dir / f"{name}.tif"
        if out.exists():
            continue

        membrane = imread(fname)
        nuclei_labels = imread(nuclei_seg_dir / f"{name}.tif")

        # Run CellPose to get the membrane mask, then watershed-fuse with
        # the cached nuclei labels. We bypass NucleiSeededCellPosePipeline's
        # full predict() because the nuclei labels are already on disk —
        # cellpose_watershed_fuse is the kernel of that pipeline.
        cellpose_res = cellpose.predict(membrane, axes=axes)
        cell_labels = cellpose_watershed_fuse(
            membrane,
            nuclei_labels=nuclei_labels.astype(np.uint16),
            cellpose_mask=cellpose_res.labels > 0,
        )
        imwrite(out, cell_labels.astype(np.uint16))


# Reference: equivalent fully-composed call when nuclei are NOT cached on disk.
#
#   from kapoorlabs_vollseg import VollSeg, VollCellSeg, StarDistSegmenter, MaskUNetSegmenter
#   nuclei_pipe = VollSeg.from_models(stardist=star, roi_unet=roi)
#   pipe = VollCellSeg.from_models(
#       nuclei_pipeline=nuclei_pipe,
#       cellpose=cellpose,
#       nuclei_channel=1,
#       membrane_channel=0,
#   )
#   result = pipe.predict(image)   # image is CZYX or TCZYX


if __name__ == "__main__":
    main()
