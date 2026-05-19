"""Tune StarDist ``(prob_thresh, nms_thresh)`` on the val split of the
training H5 and write the result back into the model's
``training_config.json`` so the predict scripts pick it up next run.

The network is run **once per validation patch** (~46 forwards) and the
``(prob_map, dist_map)`` outputs are cached. The search grid only reruns
the fast NMS + polyhedron-rasterise step
(:func:`kapoorlabs_vollseg.stardist.inference.nms_to_labels`) per
candidate ``(prob_thresh, nms_thresh)``. That's where the ~100× speed-up
over the generic ``OptimizeThreshold`` comes from — the user's earlier
run was stuck at 0/20 because every iteration re-ran the full network.

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
from omegaconf import OmegaConf
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from kapoorlabs_vollseg import StarDistSegmenter
from kapoorlabs_vollseg.eval.matching import matching_dataset
from kapoorlabs_vollseg.stardist.inference import (
    labels_from_precomputed,
    precompute_peaks_and_masks,
)

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

    # Load (raw, label) pairs from the requested H5 split.
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
    rays = star.backbone.rays
    faces = getattr(star.backbone, "faces", None)
    vol_shapes = [img.shape for img in images]
    n_tiles = tuple(p.n_tiles)

    # ───────── Stage 1: run the network once per patch, cache (prob, dist).
    print("\nStage 1: cache network outputs (one forward per patch)…")
    cached_maps = []
    for img in tqdm(images, desc="forward", unit="patch"):
        cached_maps.append(star.predict_maps(img, n_tiles=n_tiles))

    # ───────── Stage 1.5: peak-detect + rasterise once per patch at the
    # lowest prob_thresh the sweep will ever try, so Stage 2 is just a
    # filter + cheap-IoU NMS + paint.
    min_prob = float(OmegaConf.select(p, "min_prob", default=0.01))
    print(
        f"\nStage 1.5: precompute peaks + polyhedron masks "
        f"(min_prob={min_prob:g}, min_distance=2)…"
    )
    precomputed = []
    for (prob_map, dist_map), shape in tqdm(
        list(zip(cached_maps, vol_shapes)), desc="rasterize", unit="patch"
    ):
        precomputed.append(
            precompute_peaks_and_masks(
                prob_map,
                dist_map,
                rays,
                shape,
                min_prob=min_prob,
                min_distance=2,
                faces=faces,
            )
        )

    # ───────── Stage 2: sweep thresholds — only filter+NMS+paint reruns.
    nms_grid = tuple(p.nms_threshs)
    iou_thr = tuple(p.iou_threshs)
    measure = p.measure
    print(f"\nStage 2: sweep NMS={nms_grid}, measure={measure!r} " f"@ IoU={iou_thr}")

    best = {"prob": None, "nms": None, "score": -np.inf}

    def _score(prob_thresh: float, nms_thresh: float) -> float:
        preds = [
            labels_from_precomputed(
                centers,
                scores,
                bboxes,
                masks,
                shape,
                prob_thresh=float(prob_thresh),
                nms_thresh=float(nms_thresh),
            )
            for (centers, scores, bboxes, masks), shape in zip(precomputed, vol_shapes)
        ]
        stats = matching_dataset(
            labels,
            preds,
            thresh=iou_thr,
            show_progress=False,
        )
        return float(np.mean([getattr(s, measure) for s in stats]))

    for nms in nms_grid:
        cache: dict = {}

        with tqdm(total=20, desc=f"NMS = {nms:g}") as bar:

            def fn(prob_thresh: float) -> float:
                if prob_thresh in cache:
                    return -cache[prob_thresh]
                value = _score(prob_thresh, nms)
                cache[prob_thresh] = value
                bar.update()
                bar.set_postfix_str(f"prob={prob_thresh:.3f} -> {measure}={value:.3f}")
                return -value

            # Bracket the search inside [min_prob, 0.95] — prob=1.0 always
            # yields zero instances and just wastes golden-section iters.
            opt = minimize_scalar(
                fn,
                method="bounded",
                bounds=(max(min_prob, 0.05), 0.95),
                options={"xatol": 1e-2, "maxiter": 20},
            )
        score = -opt.fun
        if score > best["score"]:
            best.update(prob=float(opt.x), nms=float(nms), score=score)

    print(
        f"\nBest: prob_thresh={best['prob']:.4f}  nms_thresh={best['nms']:.4f}  "
        f"{measure}={best['score']:.4f}"
    )

    # ───────── Stage 3: patch training_config.json + a sidecar thresholds.json.
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

    (Path(log_path) / "thresholds.json").write_text(
        json.dumps({"prob": float(best["prob"]), "nms": float(best["nms"])}),
    )


if __name__ == "__main__":
    main()
