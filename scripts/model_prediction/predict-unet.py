"""Semantic U-Net prediction (3D nuclei / membrane masks).

Loads :class:`UNetSegmenter` via ``from_folder(log_path)``. The
singleton handles tiled prediction, multi-Otsu thresholding of the
probability map, connected-components labelling, and
``remove_small_objects`` cleanup — so the output TIFF is a uint16
instance label image ready for downstream watershed / measurement.

Three TIFFs are written per input file (same basename, different
suffix), so post-thresholding artefacts can be inspected against the
raw network output:

- ``<basename>.tif``       — uint16 instance labels (final).
- ``<basename>.prob.tif``  — float32 sigmoid probability map, exactly
  what the network produced before any thresholding.
- ``<basename>.mask.tif``  — uint8 binary mask after multi-Otsu, before
  connected-components.
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
    output_dir = Path(paths.base_data_dir) / paths.input_dir / paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = paths.log_path
    hf_repo_id = OmegaConf.select(paths, "hf_repo_id", default=None)
    hf_model_dir = OmegaConf.select(paths, "hf_model_dir", default="")
    if (not log_path or not Path(log_path).is_dir()) and hf_repo_id:
        name = hf_repo_id.split("/")[-1]
        log_path = str(
            ensure_model(
                hf_model_dir or log_path,
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
        stem = Path(basename).stem
        suffix = Path(basename).suffix or ".tif"
        out_path = output_dir / basename
        prob_path = output_dir / f"{stem}.prob{suffix}"
        mask_path = output_dir / f"{stem}.mask{suffix}"

        vol = imread(f)

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
            labels = np.ascontiguousarray(out["labels"], dtype=np.uint16)
            imwrite(out_path, labels)
            if out.get("probability") is not None:
                imwrite(
                    prob_path,
                    np.ascontiguousarray(out["probability"], dtype=np.float32),
                )
            if out.get("semantic") is not None:
                imwrite(
                    mask_path,
                    np.ascontiguousarray((out["semantic"] > 0).astype(np.uint8) * 255),
                )
            tqdm.write(
                f"  → {out_path}   shape={labels.shape}, "
                f"max CC/frame={int(labels.reshape(labels.shape[0], -1).max(axis=1).max())} "
                f"| prob → {prob_path.name}  mask → {mask_path.name}"
            )
        else:
            result = unet.predict(vol, n_tiles=n_tiles)
            imwrite(out_path, np.ascontiguousarray(result.labels, dtype=np.uint16))
            if result.probability is not None:
                imwrite(
                    prob_path,
                    np.ascontiguousarray(result.probability, dtype=np.float32),
                )
            if result.semantic is not None:
                imwrite(
                    mask_path,
                    np.ascontiguousarray((result.semantic > 0).astype(np.uint8) * 255),
                )
            tqdm.write(
                f"  → {out_path}   ({int(result.labels.max())} CC labels) "
                f"| prob → {prob_path.name}  mask → {mask_path.name}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
