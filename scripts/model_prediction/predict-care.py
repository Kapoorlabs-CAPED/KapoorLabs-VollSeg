"""CARE denoising prediction (+ optional PSNR / SSIM scoring).

Mirrors :mod:`predict-unet` end-to-end — top-level
``from kapoorlabs_vollseg import CAREDenoiser, predict_timelapse,
ensure_model`` (no walking through ``kapoorlabs_vollseg.care_lightning.*``
subfolders) and the same Hydra-driven config shape. Loads
:class:`CAREDenoiser` via :meth:`CAREDenoiser.from_folder` so all tiling
/ normalisation / padding lives in the singleton, not in this script.

When ``parameters.ref_dir`` points at a folder of clean-reference
TIFFs (same basenames as the noisy inputs), each output is scored
against the matching reference via :func:`skimage.metrics.peak_signal_noise_ratio`
and :func:`structural_similarity`. Leave ``ref_dir`` ``null`` to skip
scoring.
"""

from __future__ import annotations

import gc
import os
from glob import glob
from pathlib import Path

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tifffile import TiffWriter, imread, imwrite
from tqdm import tqdm

from kapoorlabs_vollseg import CAREDenoiser, ensure_model, predict_timelapse

from scenarios import CarePredictScenario


ConfigStore.instance().store(name="CarePredictScenario", node=CarePredictScenario)


def _score(denoised: np.ndarray, ref: np.ndarray) -> dict:
    """Return ``{psnr, ssim}`` between ``denoised`` and ``ref``.

    Both arrays are cast to ``float32`` and rescaled to a shared
    ``[0, 1]`` dynamic range using the reference's min/max, so PSNR /
    SSIM aren't sensitive to ``CAREDenoiser`` outputting a different
    intensity scale from the reference (e.g. percentile-normalised vs
    raw uint16).
    """
    ref = ref.astype(np.float32)
    denoised = denoised.astype(np.float32)
    ref_min, ref_max = float(ref.min()), float(ref.max())
    span = max(ref_max - ref_min, 1e-8)
    ref_n = (ref - ref_min) / span
    den_n = np.clip((denoised - ref_min) / span, 0.0, 1.0)
    psnr = float(peak_signal_noise_ratio(ref_n, den_n, data_range=1.0))
    # SSIM needs a 2D / 3D win_size that fits inside every axis; pick
    # the largest odd ``win`` <= 7 that all axes can accommodate.
    win = min(7, min(ref_n.shape))
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)
    ssim = float(
        structural_similarity(
            ref_n, den_n, data_range=1.0, win_size=win, channel_axis=None
        )
    )
    return {"psnr": psnr, "ssim": ssim}


@hydra.main(
    config_path="../conf", config_name="scenario_predict_care", version_base="1.3"
)
def main(config: CarePredictScenario):
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
    print(f"Loading CARE from {log_path}")
    care = CAREDenoiser.from_folder(log_path)
    n_tiles = tuple(p.n_tiles)

    ref_dir = Path(p.ref_dir) if p.ref_dir else None
    if ref_dir is not None and not ref_dir.is_dir():
        print(f"WARNING: ref_dir {ref_dir} does not exist — scoring disabled")
        ref_dir = None

    files = sorted(glob(os.path.join(input_dir, p.file_type)))
    print(f"Found {len(files)} input file(s) — predicting with n_tiles={n_tiles}")
    if ref_dir is not None:
        print(f"Scoring against clean references in {ref_dir}")

    scores: list[dict] = []
    for f in tqdm(files, desc="files", unit="file"):
        basename = os.path.basename(f)
        vol = imread(f)
        out_path = output_dir / basename

        if vol.ndim == 4:
            # Stream each predicted frame to disk inside ``predict_step``
            # via the ``frame_writer`` callback — nothing timelapse-sized
            # accumulates in Python. Each ``TiffWriter.write`` holds at
            # most one float32 frame in memory; the prior path peaked at
            # ~88 GB during the internal ``np.stack`` of the 192 frames.
            denoised_shape = (vol.shape[0],) + tuple(vol.shape[1:])
            tw = TiffWriter(out_path, bigtiff=True)

            def _write_frame(t_idx: int, frame_result: dict, _tw=tw):
                arr = frame_result.get("denoised")
                if arr is None:
                    return
                _tw.write(
                    np.ascontiguousarray(arr, dtype=np.float32),
                    contiguous=False,
                )

            try:
                predict_timelapse(
                    care,
                    vol,
                    devices=p.devices,
                    accelerator=p.accelerator,
                    strategy=p.strategy,
                    enable_progress_bar=True,
                    n_tiles=n_tiles,
                    frame_writer=_write_frame,
                )
            finally:
                tw.close()
            del vol
            gc.collect()
            tqdm.write(f"  → {out_path}   shape={denoised_shape}")
            denoised = None  # streaming sink: nothing in memory to score
        else:
            denoised = care.predict(vol, n_tiles=n_tiles).denoised
            del vol
            gc.collect()
            imwrite(
                out_path,
                np.ascontiguousarray(denoised, dtype=np.float32),
            )
            denoised_shape = denoised.shape

        if ref_dir is not None:
            ref_path = ref_dir / basename
            if not ref_path.is_file():
                tqdm.write(f"  → {out_path}   (no ref {ref_path.name} — score skipped)")
                if denoised is not None:
                    del denoised
                gc.collect()
                continue
            if denoised is None:
                # 4D streaming path discarded the in-memory stack — read
                # the written prediction back frame by frame and score
                # without ever holding both stacks simultaneously.
                from tifffile import TiffFile

                with TiffFile(out_path) as pred_tf, TiffFile(ref_path) as ref_tf:
                    pred_series = pred_tf.series[0]
                    ref_series = ref_tf.series[0]
                    if pred_series.shape != ref_series.shape:
                        tqdm.write(
                            f"  → {out_path}   (ref shape {ref_series.shape} != "
                            f"pred shape {pred_series.shape} — score skipped)"
                        )
                        gc.collect()
                        continue
                    psnrs, ssims = [], []
                    n_frames = (
                        pred_series.shape[0] if len(pred_series.shape) == 4 else 1
                    )
                    for t in range(n_frames):
                        pred_t = (
                            pred_series.asarray(key=t)
                            if n_frames > 1
                            else pred_series.asarray()
                        )
                        ref_t = (
                            ref_series.asarray(key=t)
                            if n_frames > 1
                            else ref_series.asarray()
                        )
                        s_t = _score(pred_t, ref_t)
                        psnrs.append(s_t["psnr"])
                        ssims.append(s_t["ssim"])
                s = {
                    "file": basename,
                    "psnr": float(np.mean(psnrs)),
                    "ssim": float(np.mean(ssims)),
                }
            else:
                ref_vol = imread(ref_path)
                if ref_vol.shape != denoised_shape:
                    tqdm.write(
                        f"  → {out_path}   (ref shape {ref_vol.shape} != "
                        f"pred shape {denoised_shape} — score skipped)"
                    )
                    del denoised, ref_vol
                    gc.collect()
                    continue
                s = _score(denoised, ref_vol)
                s["file"] = basename
                del ref_vol
            scores.append(s)
            tqdm.write(
                f"  → {out_path}   PSNR={s['psnr']:.2f} dB  SSIM={s['ssim']:.4f}"
            )

        # Final release before the next input file pulls a fresh volume.
        if denoised is not None:
            del denoised
        gc.collect()

    if scores:
        mean_psnr = float(np.mean([s["psnr"] for s in scores]))
        mean_ssim = float(np.mean([s["ssim"] for s in scores]))
        print(
            f"\nMean over {len(scores)} files — "
            f"PSNR={mean_psnr:.2f} dB  SSIM={mean_ssim:.4f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
