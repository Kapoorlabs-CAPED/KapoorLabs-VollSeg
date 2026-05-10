"""Segment nuclei in each timelapse using a composed VollSeg pipeline.

Routes each model slot to its PyTorch first-class implementation when a
``.ckpt`` path is configured, otherwise to the keras legacy class:

- StarDist → always :class:`StarDistSegmenterKeras` (no PyTorch port yet)
- U-Net    → :class:`UNetSegmenter` (PyTorch) when ``unet_nuclei_checkpoint``
             is set; else :class:`UNetSegmenterKeras`
- ROI U-Net → :class:`MaskUNetSegmenter` (PyTorch) when ``roi_nuclei_checkpoint``
              is set; else :class:`MaskUNetSegmenterKeras`
- CARE      → :class:`CAREDenoiser` (PyTorch) when ``care_membrane_checkpoint``
              is set; else :class:`CAREDenoiserKeras`

The pipeline shape is then assembled by :func:`kapoorlabs_vollseg.VollSeg.from_models`,
which is backbone-agnostic — any singleton implementing the
:class:`Pipeline` protocol composes.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from natsort import natsorted
from tifffile import imread, imwrite
from tqdm import tqdm

from kapoorlabs_vollseg import (
    CAREBackboneKeras,
    CAREDenoiser,
    CAREDenoiserKeras,
    MaskUNetBackboneKeras,
    MaskUNetSegmenter,
    MaskUNetSegmenterKeras,
    Pipeline,
    StarDist3DBackboneKeras,
    StarDistSegmenterKeras,
    UNetBackboneKeras,
    UNetSegmenter,
    UNetSegmenterKeras,
    VollSeg,
    ensure_model,
)

from scenarios import SegmentScenario


ConfigStore.instance().store(name="SegmentScenario", node=SegmentScenario)


def _pt_kwargs(p) -> dict:
    """Architecture knobs forwarded to *.from_checkpoint for PyTorch backbones."""
    return dict(
        conv_dims=p.pt_conv_dims,
        in_channels=p.pt_in_channels,
        num_classes=p.pt_num_classes,
        depth=p.pt_unet_depth,
        num_channels_init=p.pt_num_channels_init,
        use_batch_norm=p.pt_use_batch_norm,
        n_tiles=list(p.n_tiles),
        tile_overlap=p.pt_tile_overlap,
    )


def _build_unet(config: SegmentScenario) -> Optional[Pipeline]:
    if not config.parameters.use_seedpool:
        return None
    mp, p = config.model_paths, config.parameters
    if mp.unet_nuclei_checkpoint:
        return UNetSegmenter.from_checkpoint(
            mp.unet_nuclei_checkpoint, min_size=p.min_size_mask, **_pt_kwargs(p)
        )
    ensure_model(mp.unet_model_dir, mp.unet_nuclei_model_name)
    return UNetSegmenterKeras(
        UNetBackboneKeras(
            config=None, name=mp.unet_nuclei_model_name, basedir=mp.unet_model_dir
        ),
        min_size=p.min_size_mask,
    )


def _build_roi(config: SegmentScenario) -> Optional[Pipeline]:
    if not config.parameters.use_roi_unet:
        return None
    mp, p = config.model_paths, config.parameters
    if mp.roi_nuclei_checkpoint:
        return MaskUNetSegmenter.from_checkpoint(
            mp.roi_nuclei_checkpoint, min_size=p.min_size_mask, **_pt_kwargs(p)
        )
    ensure_model(mp.roi_model_dir, mp.roi_nuclei_model_name)
    return MaskUNetSegmenterKeras(
        MaskUNetBackboneKeras(
            config=None, name=mp.roi_nuclei_model_name, basedir=mp.roi_model_dir
        ),
        min_size=p.min_size_mask,
    )


def _build_care(config: SegmentScenario) -> Optional[Pipeline]:
    if not config.parameters.use_care_denoise:
        return None
    mp, p = config.model_paths, config.parameters
    if mp.care_membrane_checkpoint:
        return CAREDenoiser.from_checkpoint(
            mp.care_membrane_checkpoint, **_pt_kwargs(p)
        )
    ensure_model(mp.care_model_dir, mp.care_membrane_model_name)
    return CAREDenoiserKeras(
        CAREBackboneKeras(
            config=None, name=mp.care_membrane_model_name, basedir=mp.care_model_dir
        )
    )


def _build_pipeline(config: SegmentScenario) -> Pipeline:
    ensure_model(
        config.model_paths.star_model_dir, config.model_paths.star_nuclei_model_name
    )
    star = StarDistSegmenterKeras(
        StarDist3DBackboneKeras(
            config=None,
            name=config.model_paths.star_nuclei_model_name,
            basedir=config.model_paths.star_model_dir,
        ),
        prob_thresh=config.parameters.prob_thresh,
        nms_thresh=config.parameters.nms_thresh,
    )
    return VollSeg.from_models(
        care=_build_care(config),
        unet=_build_unet(config),
        stardist=star,
        roi_unet=_build_roi(config),
        seedpool=config.parameters.use_seedpool,
    )


def _extract_nuclei_volume(image: np.ndarray, channel_nuclei: int) -> np.ndarray:
    if image.ndim == 5:
        return image[:, :, channel_nuclei, :, :]
    if image.ndim == 4:
        return image
    raise ValueError(f"Expected 4D or 5D image, got ndim={image.ndim}")


@hydra.main(version_base="1.3", config_path="conf", config_name="scenario_segment")
def main(config: SegmentScenario) -> None:
    base = Path(config.experiment_data_paths.base_directory)
    nuclei_dir = base / config.experiment_data_paths.timelapse_nuclei_directory
    seg_dir = base / config.experiment_data_paths.timelapse_seg_nuclei_directory
    seg_dir.mkdir(parents=True, exist_ok=True)
    timelapse_dir = seg_dir / "timelapse"
    timelapse_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _build_pipeline(config)

    n_tiles = tuple(config.parameters.n_tiles)
    axes = config.parameters.axes
    voxel = config.experiment_data_paths.voxel_size_xyz

    files = natsorted(glob.glob(os.fspath(nuclei_dir / config.parameters.file_type)))

    for fname in tqdm(files, desc="nuclei segmentation"):
        name = Path(fname).stem
        image = imread(fname)
        nuclei_tzyx = _extract_nuclei_volume(image, config.parameters.channel_nuclei)

        for t in range(nuclei_tzyx.shape[0]):
            out = seg_dir / f"{name}_{t}.tif"
            if out.exists():
                continue
            result = pipeline.predict(nuclei_tzyx[t], axes=axes, n_tiles=n_tiles)
            imwrite(out, result.labels.astype(np.uint16))

        frames = natsorted(seg_dir.glob(f"{name}_*.tif"))
        stack = np.stack([imread(p) for p in frames], axis=0)
        imwrite(
            timelapse_dir / f"{name}.tif",
            stack,
            imagej=True,
            bigtiff=True,
            photometric="minisblack",
            resolution=(1 / voxel[0], 1 / voxel[1]),
            metadata={"spacing": voxel[2], "unit": "um", "axes": "TZYX"},
        )


if __name__ == "__main__":
    main()
