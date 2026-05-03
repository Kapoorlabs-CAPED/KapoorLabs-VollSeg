"""Denoise membrane channel of each timelapse with a CARE model.

Replaces ``CopenhagenWorkflow/01_enhance_membrane.py``. Routes to the
PyTorch :class:`vollseg.CAREDenoiser` when a Lightning ``.ckpt`` is
configured; otherwise falls back to :class:`vollseg.CAREDenoiserKeras`
loading a csbdeep-trained ``.h5`` from disk. PyTorch is the preferred
path going forward.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from natsort import natsorted
from tifffile import imread, imwrite
from tqdm import tqdm

from vollseg import (
    CAREBackboneKeras,
    CAREDenoiser,
    CAREDenoiserKeras,
)

from scenarios import SegmentScenario


ConfigStore.instance().store(name="SegmentScenario", node=SegmentScenario)


def _build_denoiser(config: SegmentScenario):
    mp = config.model_paths
    p = config.parameters
    if mp.care_membrane_checkpoint:
        return CAREDenoiser.from_checkpoint(
            mp.care_membrane_checkpoint,
            conv_dims=p.pt_conv_dims,
            in_channels=p.pt_in_channels,
            num_classes=p.pt_num_classes,
            depth=p.pt_unet_depth,
            num_channels_init=p.pt_num_channels_init,
            use_batch_norm=p.pt_use_batch_norm,
            n_tiles=list(p.n_tiles),
            tile_overlap=p.pt_tile_overlap,
        )
    # Keras fallback for legacy .h5 weights.
    backbone = CAREBackboneKeras(
        config=None,
        name=mp.care_membrane_model_name,
        basedir=mp.care_model_dir,
    )
    return CAREDenoiserKeras(backbone)


@hydra.main(version_base="1.3", config_path="conf", config_name="scenario_segment")
def main(config: SegmentScenario) -> None:
    base = Path(config.experiment_data_paths.base_directory)
    membrane_dir = base / config.experiment_data_paths.timelapse_membrane_directory
    enhanced_dir = base / config.experiment_data_paths.membrane_enhanced_directory
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    denoiser = _build_denoiser(config)
    is_pytorch = isinstance(denoiser, CAREDenoiser)

    n_tiles = tuple(config.parameters.n_tiles)
    axes = config.parameters.axes
    channel_membrane = config.parameters.channel_membrane

    files = natsorted(glob.glob(os.fspath(membrane_dir / config.parameters.file_type)))

    for fname in tqdm(files, desc="enhance membrane"):
        name = Path(fname).stem
        out = enhanced_dir / f"{name}.tif"
        if out.exists():
            print(f"Skipping {fname} — output already exists at {out}")
            continue

        image = imread(fname)
        if image.ndim == 4:
            membrane = image[:, channel_membrane, :, :]
        elif image.ndim == 3:
            membrane = image
        else:
            raise ValueError(f"Expected 3D or 4D image, got ndim={image.ndim}")

        # PyTorch path takes a 3D volume directly; keras path takes axes too.
        if is_pytorch:
            result = denoiser.predict(membrane, n_tiles=n_tiles)
        else:
            result = denoiser.predict(membrane, axes=axes, n_tiles=n_tiles)
        imwrite(out, result.denoised)


if __name__ == "__main__":
    main()
