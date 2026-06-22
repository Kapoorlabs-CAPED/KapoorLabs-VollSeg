# Model training

PyTorch Lightning training entry points, one per task, reading the H5
files emitted by [`../train_data_generation/`](../train_data_generation/).

## Layout

```
model_training/
├── lightning-care.py                  CARE denoiser training
├── lightning-unet.py                  U-Net (binary mask) training
├── lightning-roi.py                   ROI Mask-UNet (2D) training
├── lightning-stardist.py              StarDist 3D training
├── optimize-stardist-thresholds.py    cached prob/nms sweep on the val H5 split
├── scenario_train_{care,unet,roi,stardist}.py     hydra dataclass schemas
├── scenario_optimize_stardist_thresholds.py
├── conf/
│   ├── scenario_train_{care,unet,roi,stardist}.yaml
│   ├── scenario_optimize_stardist_thresholds.yaml
│   ├── parameters/                    per-task arch + training knobs
│   └── train_data_paths/              per-cluster path templates
└── slurm_{train,sweep,*}.sh           jeanzay + gwdg submission scripts
```

## Train

```bash
python lightning-care.py
python lightning-unet.py
python lightning-roi.py
python lightning-stardist.py

# Per-cluster path overrides:
python lightning-stardist.py train_data_paths=xenopus_jeanzay
python lightning-unet.py    train_data_paths=xenopus_gwdg

# Override any field:
python lightning-stardist.py \
    parameters.batch_size=8 \
    parameters.optimizer=adam \
    parameters.learning_rate=1.0e-3 \
    parameters.scheduler=cosine \
    parameters.anisotropy=[2.4286,1,1]
```

## What gets written

Every trainer drops these next to the checkpoint:

```
<log_path>/
├── last.ckpt                              Lightning checkpoint
├── <experiment_name>-epoch=NNN.ckpt       per-epoch checkpoints
├── training_config.json                   {"parameters": {...}}  ← read by from_folder
├── <experiment_name>.json                 flat fallback sidecar
└── metrics.csv                            Lightning CSVLogger
```

StarDist rays are **regenerated deterministically** from
`(conv_dims, n_rays, anisotropy)` in `training_config.json` at load time
— no `rays.npy` sidecar is written or needed.

After threshold optimisation:
```
<log_path>/training_config.json["parameters"]["prob_thresh"|"nms_thresh"]
<log_path>/thresholds.json                  {"prob": ..., "nms": ...}  ← sidecar
```

## Load a trained model

```python
from kapoorlabs_vollseg import StarDistSegmenter, UNetSegmenter, MaskUNetSegmenter, CAREDenoiser

star = StarDistSegmenter.from_folder("<log_path>")
unet = UNetSegmenter.from_folder("<log_path>")
roi  = MaskUNetSegmenter.from_folder("<log_path>")
care = CAREDenoiser.from_folder("<log_path>")
```

`from_folder` reads `training_config.json` (or the flat fallback), reconstructs the architecture, loads the Lightning checkpoint, and — for StarDist — regenerates rays from JSON.

**Thresholds are NOT auto-applied by `from_folder`** — the prediction scripts (`predict-stardist.py`, `compare-stardist-vs-keras.py`, the sweep scorers) read them via `kapoorlabs_vollseg._backbones._config.read_thresholds(log_path)` and pass them explicitly to `predict(..., prob_thresh=..., nms_thresh=...)`.

## StarDist threshold optimisation

```bash
python optimize-stardist-thresholds.py \
    train_data_paths.log_path=/lustre/.../models_stardist_pytorch \
    parameters.nms_threshs=[0.3,0.4,0.5] \
    parameters.measure=accuracy
```

Three-stage flow:

1. Run the network once per val patch (`star.predict_maps`) and cache `(prob_map, dist_map)`.
2. Precompute peaks + per-peak polyhedron masks once at the lowest prob the sweep will visit.
3. For each `nms_thresh` in the grid, golden-section over `prob_thresh ∈ [0.05, 0.95]` with the cached masks. Paint + match only — no model forwards in the inner loop. ~100× faster than re-running the model per candidate.

Winner is written into `training_config.json` and `thresholds.json`. Predict scripts then pick it up automatically.

## SLURM

Single-model training:
```bash
sbatch slurm_train_stardist_jeanzay.sh
sbatch slurm_train_unet_jeanzay.sh
sbatch slurm_train_care_jeanzay.sh
sbatch slurm_train_roi.sh

# Retrain the StarDist sweep winner with anisotropy = (17/7, 1, 1):
sbatch slurm_train_stardist_winner_jeanzay.sh
```

GWDG (Grete) variants for U-Net and StarDist live next to the Jean-Zay ones.

Sweeps (18-task array, 3 optimisers × 3 LRs × 2 schedulers):
```bash
sbatch slurm_sweep_stardist_jeanzay.sh
sbatch slurm_sweep_unet_jeanzay.sh
sbatch slurm_sweep_care_jeanzay.sh
```

Score every sweep run against a keras reference with
`scripts/model_prediction/sweep_predict_{and,unet,roi_stardist}_and_analyze.py` — see [`../model_prediction/`](../model_prediction/) for the prediction side.
