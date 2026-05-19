"""Semantic U-Net prediction (3D nuclei / membrane masks).

Loads :class:`UNetSegmenter` via ``from_folder(log_path)``. The
singleton handles tiled prediction, multi-Otsu thresholding of the
probability map, connected-components labelling, and
``remove_small_objects`` cleanup — so the output TIFF is a uint16
instance label image ready for downstream watershed / measurement.
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
from tqdm import tqdm

from kapoorlabs_vollseg import UNetSegmenter, ensure_model, predict_timelapse

from scenarios import UNetPredictScenario


ConfigStore.instance().store(name="UNetPredictScenario", node=UNetPredictScenario)


@hydra.main(
    config_path="../conf", config_name="scenario_predict_unet", version_base="1.3"
)
def main(config: UNetPredictScenario):
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
    print(f"Loading U-Net from {log_path}")
    unet = UNetSegmenter.from_folder(log_path)
    unet.min_size = int(p.min_size)
    n_tiles = tuple(p.n_tiles)
    print(f"  model_dim = {unet.model_dim}D")

    files = sorted(glob(os.path.join(input_dir, p.file_type)))
    print(f"Found {len(files)} input file(s) — predicting with n_tiles={n_tiles}")
    for f in tqdm(files, desc="files", unit="file"):
        basename = os.path.basename(f)
        vol = imread(f)
        out_path = output_dir / basename

        if vol.ndim == 4:
            out = predict_timelapse(
                unet,
                vol,
                devices=p.devices,
                accelerator=p.accelerator,
                strategy=p.strategy,
                enable_progress_bar=True,
                n_tiles=n_tiles,
            )
            if not out:
                continue
            labels_t = [out["labels"][t] for t in range(out["labels"].shape[0])]
            stacked = np.stack(labels_t, axis=0)
            imwrite(out_path, stacked)
            tqdm.write(
                f"  → {out_path}   shape={stacked.shape}, "
                f"max CC/frame={max(int(x.max()) for x in labels_t)}"
            )
        else:
            result = unet.predict(vol, n_tiles=n_tiles)
            imwrite(out_path, result.labels.astype(np.uint16))
            tqdm.write(f"  → {out_path}   ({int(result.labels.max())} CC labels)")

    print("\nDone.")


if __name__ == "__main__":
    main()
