"""ROI Mask-UNet prediction (2D-on-3D MIP, multi-Otsu, CC, min-size).

Thin Hydra wrapper around :class:`MaskUNetSegmenter` — the 2D-on-3D
max-Z projection, percentile normalisation, multi-Otsu threshold,
connected-component labelling and ``remove_small_objects`` all live
inside the singleton (see ``UNetSegmenter.predict``), so this script
only owns config plumbing and disk I/O.

Architecture knobs (``conv_dims`` = 2 for ROI Mask-UNet, ``unet_depth``,
``num_channels_init``, ``use_batch_norm``) come from
``training_config.json`` next to the checkpoint — the canonical record
of how the model was built. No yaml fallback: if the JSON is missing,
fail loud rather than silently mis-construct the network.

Run::

    python predict-roi.py \\
        parameters=predict_roi \\
        experiment_data_paths=xenopus_jeanzay_roi
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

from kapoorlabs_vollseg import MaskUNetSegmenter, ensure_model, predict_timelapse

from scenarios import RoiPredictScenario


ConfigStore.instance().store(name="RoiPredictScenario", node=RoiPredictScenario)


@hydra.main(
    config_path="../conf", config_name="scenario_predict_roi", version_base="1.3"
)
def main(config: RoiPredictScenario):
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
        log_path = str(ensure_model(hf_model_dir or log_path, name, repo_id=hf_repo_id))
    print(f"Loading ROI Mask-UNet from {log_path}")
    roi = MaskUNetSegmenter.from_folder(log_path)
    roi.min_size = int(p.min_size_mask)
    n_tiles = tuple(p.n_tiles)
    print(f"  model_dim = {roi.model_dim}D")

    files = sorted(glob(os.path.join(input_dir, p.file_type)))
    print(f"Found {len(files)} input file(s) — predicting with n_tiles={n_tiles}")
    for f in tqdm(files, desc="files", unit="file"):
        basename = os.path.basename(f)
        out_path = output_dir / basename

        vol = imread(f)
        if vol.ndim == 4:
            out = predict_timelapse(
                roi,
                vol,
                devices=p.devices,
                accelerator=p.accelerator,
                strategy=p.strategy,
                enable_progress_bar=True,
                n_tiles=n_tiles,
            )
            if not out:
                continue
            imwrite(out_path, np.ascontiguousarray(out["labels"], dtype=np.uint16))
        else:
            result = roi.predict(vol, n_tiles=n_tiles)
            imwrite(out_path, np.ascontiguousarray(result.labels, dtype=np.uint16))
        tqdm.write(f"  → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
