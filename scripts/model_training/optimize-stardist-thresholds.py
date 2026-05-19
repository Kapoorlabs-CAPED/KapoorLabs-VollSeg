"""Tune StarDist ``(prob_thresh, nms_thresh)`` on the val split of the
training H5 and write the result back into the model's
``training_config.json`` so the predict scripts pick it up next run.

Same H5 layout the trainer consumes
(``/{train,val}/raw + /{train,val}/label``); we load the validation
samples, run :class:`OptimizeThreshold` against the freshly-trained
:class:`StarDistSegmenter`, and patch
``training_config.json["parameters"]`` with the optimal pair.

Run::

    python optimize-stardist-thresholds.py \\
        train_data_paths=xenopus_jeanzay
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import hydra
import numpy as np
from hydra.core.config_store import ConfigStore

from kapoorlabs_vollseg import StarDistSegmenter
from kapoorlabs_vollseg.eval import OptimizeThreshold

from scenario_optimize_stardist_thresholds import OptimizeThresholdsScenario


ConfigStore.instance().store(
    name="OptimizeThresholdsScenario",
    node=OptimizeThresholdsScenario,
)


@hydra.main(
    config_path="conf",
    config_name="scenario_optimize_stardist_thresholds",
    version_base="1.3",
)
def main(config: OptimizeThresholdsScenario):
    p = config.parameters
    paths = config.train_data_paths

    h5_path = os.path.join(paths.base_data_dir, paths.h5_file)
    log_path = paths.log_path
    print(f"H5:        {h5_path}")
    print(f"log_path:  {log_path}")

    # Load the (raw, label) pairs from the requested H5 split.
    n_used = int(p.max_samples)
    images, labels = [], []
    with h5py.File(h5_path, "r") as fh:
        grp = fh[p.split]
        n_total = grp["raw"].shape[0]
        n_used = n_total if n_used < 0 else min(n_used, n_total)
        for i in range(n_used):
            images.append(np.asarray(grp["raw"][i]))
            labels.append(np.asarray(grp["label"][i]))
    print(
        f"Loaded {n_used} samples from /{p.split}/ "
        f"(of {n_total}). Patch shape = {images[0].shape}."
    )

    star = StarDistSegmenter.from_folder(log_path, n_rays=p.n_rays)

    opt = OptimizeThreshold(
        star,
        images,
        labels,
        nms_threshs=tuple(p.nms_threshs),
        iou_threshs=tuple(p.iou_threshs),
        measure=p.measure,
        n_tiles=tuple(p.n_tiles),
        savedir=Path(log_path),
        normalize_inputs=p.normalize_inputs,
        norm_axes=tuple(p.norm_axes),
    )
    best = opt.run()
    print(f"\nBest: prob_thresh={best['prob']:.4f}  nms_thresh={best['nms']:.4f}")

    # Patch training_config.json so predict-stardist.py picks these up.
    cfg_path = Path(log_path) / "training_config.json"
    if cfg_path.is_file():
        blob = json.loads(cfg_path.read_text())
    else:
        blob = {"parameters": {}}
    blob.setdefault("parameters", {})
    blob["parameters"]["prob_thresh"] = float(best["prob"])
    blob["parameters"]["nms_thresh"] = float(best["nms"])
    cfg_path.write_text(json.dumps(blob, indent=2))
    print(f"Wrote prob_thresh / nms_thresh into {cfg_path}")


if __name__ == "__main__":
    main()
