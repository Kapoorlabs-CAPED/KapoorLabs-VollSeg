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

from kapoorlabs_vollseg import StarDistSegmenter, ensure_model

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
    print(
        f"  rays={star.backbone.rays.shape[0]} "
        f"prob_thresh={p.prob_thresh}  nms_thresh={p.nms_thresh}"
    )

    files = sorted(glob(os.path.join(input_dir, p.file_type)))
    print(f"Found {len(files)} input file(s) — predicting with n_tiles={n_tiles}")
    for f in files:
        basename = os.path.basename(f)
        print(f"\n{basename}")
        vol = imread(f)
        if vol.ndim == 4:
            vol = vol[0]
        print(f"  input shape: {vol.shape}")
        result = star.predict(
            vol,
            prob_thresh=p.prob_thresh,
            nms_thresh=p.nms_thresh,
            n_tiles=n_tiles,
        )
        out_path = output_dir / basename
        imwrite(out_path, result.labels.astype(np.uint32))
        print(f"  → {out_path}   ({int(result.labels.max())} instances)")

    print("\nDone.")


if __name__ == "__main__":
    main()
