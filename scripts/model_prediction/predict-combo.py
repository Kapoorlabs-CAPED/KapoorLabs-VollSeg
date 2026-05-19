"""Combo segmentation prediction — StarDist / U-Net / ROI Mask-UNet in any mix.

Resolves the pipeline shape from which model folders the user filled in:

    stardist + maskunet               → ROI(MaskUNet) ∘ StarDist
    stardist + unet (seedpool=False)  → UNetStarDistPipeline (returns both)
    stardist + unet + maskunet (sp=F) → ROI(MaskUNet) ∘ UNetStarDistPipeline
    stardist + unet (seedpool=True)   → marker-controlled watershed fusion
    stardist + unet + maskunet (sp=T) → ROI(MaskUNet) ∘ seedpool-fused watershed

The MaskUNet operates in 2D (and broadcasts the mask across Z internally
— see :class:`MaskUNetSegmenter`); everything else is 3D. A 4D
``(T, Z, Y, X)`` input is processed frame-by-frame and the per-frame
results are stacked along the T axis on output.

Run::

    python scripts/model_prediction/predict-combo.py \\
        experiment_data_paths.base_data_dir=/data/exp \\
        experiment_data_paths.stardist.hf_repo_id=KapoorLabs/xenopus-stardist-pytorch \\
        experiment_data_paths.unet.hf_repo_id=KapoorLabs/xenopus-unet-pytorch \\
        experiment_data_paths.maskunet.hf_repo_id=KapoorLabs/xenopus-maskunet-pytorch \\
        parameters.seedpool=true
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

from kapoorlabs_vollseg import (
    MaskUNetSegmenter,
    StarDistSegmenter,
    UNetSegmenter,
    VollSeg,
    ensure_model,
)

from scenarios import ComboPredictScenario


ConfigStore.instance().store(name="ComboPredictScenario", node=ComboPredictScenario)


def _resolve(ref) -> str:
    """Return the local folder for one role, or ``""`` if role is empty.

    Struct-mode safe — older yamls that pre-date the HF fields still load.
    """
    log_path = OmegaConf.select(ref, "log_path", default="") or ""
    hf_repo_id = OmegaConf.select(ref, "hf_repo_id", default=None)
    hf_model_dir = OmegaConf.select(ref, "hf_model_dir", default="")
    if hf_repo_id:
        name = hf_repo_id.split("/")[-1]
        return str(
            ensure_model(
                hf_model_dir or log_path,
                name,
                repo_id=hf_repo_id,
            )
        )
    return log_path


@hydra.main(
    config_path="../conf", config_name="scenario_predict_combo", version_base="1.3"
)
def main(config: ComboPredictScenario):
    p = config.parameters
    paths = config.experiment_data_paths
    n_tiles = tuple(p.n_tiles)

    input_dir = os.path.join(paths.base_data_dir, paths.input_dir)
    output_dir = Path(paths.base_data_dir) / paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    star_path = _resolve(paths.stardist)
    unet_path = _resolve(paths.unet)
    roi_path = _resolve(paths.maskunet)

    star = (
        StarDistSegmenter.from_folder(star_path, n_rays=p.n_rays) if star_path else None
    )
    unet = UNetSegmenter.from_folder(unet_path) if unet_path else None
    roi = MaskUNetSegmenter.from_folder(roi_path) if roi_path else None
    if unet is not None:
        unet.min_size = int(p.min_size)
    if roi is not None:
        roi.min_size = int(p.min_size_mask)

    pipe = VollSeg.from_models(
        stardist=star,
        unet=unet,
        roi_unet=roi,
        seedpool=p.seedpool,
    )
    mode = (
        f"stardist={'on' if star else 'off'}  "
        f"unet={'on' if unet else 'off'}  "
        f"maskunet={'on' if roi else 'off'}  "
        f"seedpool={p.seedpool}  →  {type(pipe).__name__}"
    )
    print(f"Pipeline: {mode}")

    files = sorted(glob(os.path.join(input_dir, p.file_type)))
    print(f"Found {len(files)} input file(s) — predicting with n_tiles={n_tiles}")

    for f in tqdm(files, desc="files", unit="file"):
        basename = os.path.basename(f)
        vol = imread(f)

        # 4D = TZYX timelapse → iterate over T; everything else passes through.
        if vol.ndim == 4:
            labels_t, semantic_t, roi_t = [], [], []
            for t in tqdm(
                range(vol.shape[0]),
                desc=f"  {basename} (T)",
                leave=False,
                unit="frame",
            ):
                r = pipe.predict(
                    vol[t],
                    prob_thresh=p.prob_thresh,
                    nms_thresh=p.nms_thresh,
                    n_tiles=n_tiles,
                )
                if r.labels is not None:
                    labels_t.append(r.labels.astype(np.uint32))
                if r.semantic is not None:
                    semantic_t.append(r.semantic.astype(np.uint8))
                if r.roi is not None:
                    roi_t.append(r.roi.astype(np.uint8))
            if labels_t:
                imwrite(output_dir / f"labels_{basename}", np.stack(labels_t, axis=0))
            if semantic_t:
                imwrite(
                    output_dir / f"semantic_{basename}", np.stack(semantic_t, axis=0)
                )
            if roi_t:
                imwrite(output_dir / f"roi_{basename}", np.stack(roi_t, axis=0))
        else:
            r = pipe.predict(
                vol,
                prob_thresh=p.prob_thresh,
                nms_thresh=p.nms_thresh,
                n_tiles=n_tiles,
            )
            if r.labels is not None:
                imwrite(output_dir / f"labels_{basename}", r.labels.astype(np.uint32))
            if r.semantic is not None:
                imwrite(
                    output_dir / f"semantic_{basename}", r.semantic.astype(np.uint8)
                )
            if r.roi is not None:
                imwrite(output_dir / f"roi_{basename}", r.roi.astype(np.uint8))
        print(
            "  ✓ wrote labels/" + ("semantic/" if unet else "") + ("roi" if roi else "")
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
