"""ROI Mask-UNet prediction — 2D-on-2D *or* 2D-on-3D (MIP + broadcast).

Loads :class:`MaskUNetSegmenter` via ``from_folder(log_path)``. The
``training_config.json`` in the folder carries ``conv_dims=2`` for the
Xenopus ROI checkpoints, so the loader builds a 2D backbone
automatically. The singleton's ``.predict`` then handles the
2D-image-direct vs 3D-image-MIP dispatch internally (matching the
original ``VollSeg_unet`` flow), and broadcasts the resulting 2D mask
back to ``ZYX`` so the output TIFF has the same shape as the input.
"""

from __future__ import annotations

import os
from glob import glob
from pathlib import Path

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tifffile import imread, imwrite

from kapoorlabs_vollseg import MaskUNetSegmenter, ensure_model

from scenarios import RoiPredictScenario


ConfigStore.instance().store(name="RoiPredictScenario", node=RoiPredictScenario)


@hydra.main(
    config_path="../conf", config_name="scenario_predict_roi", version_base="1.3"
)
def main(config: RoiPredictScenario):
    paths = config.experiment_data_paths
    p = config.parameters

    input_dir = os.path.join(paths.base_data_dir, paths.input_dir)
    output_dir = Path(paths.base_data_dir) / paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = paths.log_path
    hf_repo_id = OmegaConf.select(paths, "hf_repo_id", default=None)
    hf_model_dir = OmegaConf.select(paths, "hf_model_dir", default="")
    if hf_repo_id:
        name = hf_repo_id.split("/")[-1]
        log_path = str(
            ensure_model(
                hf_model_dir or paths.log_path,
                name,
                repo_id=hf_repo_id,
            )
        )
    print(f"Loading ROI Mask-UNet from {log_path}")
    roi = MaskUNetSegmenter.from_folder(log_path)
    # Bump the min-size floor; the singleton's default of 10 is too low
    # for the typical mask-UNet output of a tissue ROI.
    roi.min_size = int(p.min_size_mask)
    n_tiles = tuple(p.n_tiles)
    print(f"  model_dim = {roi.model_dim}D")

    files = sorted(glob(os.path.join(input_dir, p.file_type)))
    print(f"Found {len(files)} input file(s) — predicting with n_tiles={n_tiles}")
    for f in files:
        basename = os.path.basename(f)
        print(f"\n{basename}")
        vol = imread(f)
        if vol.ndim == 4:
            vol = vol[0]
        print(f"  input shape: {vol.shape}")
        result = roi.predict(vol, n_tiles=n_tiles)
        out_path = output_dir / basename
        imwrite(out_path, result.labels.astype(np.uint16))
        print(f"  → {out_path}   ({int(result.labels.max())} ROI regions)")

    print("\nDone.")


if __name__ == "__main__":
    main()
