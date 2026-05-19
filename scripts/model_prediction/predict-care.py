"""CARE denoising — predict over a folder of TIFFs.

Loads a :class:`CAREDenoiser` via ``from_folder(log_path)`` (picks up the
``.ckpt`` + ``training_config.json`` written by ``CARETrainer``), runs
tiled prediction over every file matching ``file_type``, and writes a
denoised TIFF per input into ``output_dir``.
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

from kapoorlabs_vollseg import CAREDenoiser, ensure_model

from scenarios import CarePredictScenario


ConfigStore.instance().store(name="CarePredictScenario", node=CarePredictScenario)


@hydra.main(
    config_path="../conf", config_name="scenario_predict_care", version_base="1.3"
)
def main(config: CarePredictScenario):
    paths = config.experiment_data_paths
    p = config.parameters

    input_dir = os.path.join(paths.base_data_dir, paths.input_dir)
    output_dir = Path(paths.base_data_dir) / paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = paths.log_path
    # Struct-mode safe — old yamls without HF fields still work.
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
    print(f"Loading CARE model from {log_path}")
    care = CAREDenoiser.from_folder(log_path)
    n_tiles = tuple(p.n_tiles)

    files = sorted(glob(os.path.join(input_dir, p.file_type)))
    print(f"Found {len(files)} input file(s) — denoising with n_tiles={n_tiles}")
    for f in tqdm(files, desc="files", unit="file"):
        basename = os.path.basename(f)
        vol = imread(f)
        out_path = output_dir / basename

        if vol.ndim == 4:
            denoised_t = []
            for t in tqdm(
                range(vol.shape[0]),
                desc=f"  {basename} (T)",
                leave=False,
                unit="frame",
            ):
                r = care.predict(vol[t], n_tiles=n_tiles)
                denoised_t.append(r.denoised.astype(np.float32))
            stacked = np.stack(denoised_t, axis=0)
            imwrite(out_path, stacked)
            tqdm.write(f"  → {out_path}   shape={stacked.shape}")
        else:
            result = care.predict(vol, n_tiles=n_tiles)
            imwrite(out_path, result.denoised.astype(np.float32))
            tqdm.write(f"  → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
