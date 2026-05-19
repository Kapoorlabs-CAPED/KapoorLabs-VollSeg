"""StarDist instance-segmentation prediction.

Loads :class:`StarDistSegmenter` via ``from_folder(log_path)``. The
loader picks up:

- the ``.ckpt`` written by ``StarDistTrainer``,
- the ``rays.npy`` sidecar (canonical name) — falls back to a fresh
  golden-spiral set of length ``n_rays`` only if no rays sidecar
  exists in the folder,
- ``training_config.json`` for ``conv_dims`` / ``unet_depth`` etc.

Then runs tiled NMS prediction and writes a uint32 instance-label
TIFF per input.
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

from kapoorlabs_vollseg import StarDistSegmenter, ensure_model, predict_timelapse
from kapoorlabs_vollseg._backbones._config import read_thresholds

from scenarios import StarDistPredictScenario


ConfigStore.instance().store(
    name="StarDistPredictScenario", node=StarDistPredictScenario
)


@hydra.main(
    config_path="../conf", config_name="scenario_predict_stardist", version_base="1.3"
)
def main(config: StarDistPredictScenario):
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
    print(f"Loading StarDist from {log_path}")
    star = StarDistSegmenter.from_folder(log_path, n_rays=p.n_rays)
    n_tiles = tuple(p.n_tiles)

    # JSON thresholds in the model folder override the yaml defaults
    # (yaml = 0.4 / 0.3 baseline; JSON wins when the model ships its own).
    overrides = read_thresholds(log_path)
    prob_thresh = overrides.get("prob_thresh", p.prob_thresh)
    nms_thresh = overrides.get("nms_thresh", p.nms_thresh)
    print(
        f"  rays={star.backbone.rays.shape[0]} "
        f"prob_thresh={prob_thresh}  nms_thresh={nms_thresh}"
        + ("   (from training_config.json)" if overrides else "")
    )

    files = sorted(glob(os.path.join(input_dir, p.file_type)))
    print(f"Found {len(files)} input file(s) — predicting with n_tiles={n_tiles}")
    for f in tqdm(files, desc="files", unit="file"):
        basename = os.path.basename(f)
        vol = imread(f)
        out_path = output_dir / basename

        # 4D = TZYX timelapse → Lightning Trainer.predict shards T across
        # `devices` GPUs (DDP-style) and stacks the per-frame Result fields.
        if vol.ndim == 4:
            out = predict_timelapse(
                star,
                vol,
                devices=p.devices,
                accelerator=p.accelerator,
                strategy=p.strategy,
                enable_progress_bar=True,
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh,
                n_tiles=n_tiles,
            )
            if not out:
                continue  # non-zero DDP rank
            labels_t = [out["labels"][t] for t in range(out["labels"].shape[0])]
            stacked = np.stack(labels_t, axis=0)
            imwrite(out_path, stacked)
            tqdm.write(
                f"  → {out_path}   shape={stacked.shape}, "
                f"max instances/frame={max(int(x.max()) for x in labels_t)}"
            )
        else:
            result = star.predict(
                vol,
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh,
                n_tiles=n_tiles,
            )
            imwrite(out_path, result.labels.astype(np.uint32))
            tqdm.write(f"  → {out_path}   ({int(result.labels.max())} instances)")

    print("\nDone.")


if __name__ == "__main__":
    main()
