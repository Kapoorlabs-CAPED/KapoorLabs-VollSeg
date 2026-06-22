# scripts/

Hydra-driven CLI for everything not in the SDK: data generation, training, prediction, comparison, analysis, HF model uploads.

## Layout

```
scripts/
├── train_data_generation/        H5 patch generators (SmartPatches + visualizers)
├── model_training/               Lightning trainers per task + sweep + threshold optimisation
├── model_prediction/             Hydra predict scripts + comparison scripts + tile sweeps
├── curvature_physics/            curvature distribution analysis (Hydra)
├── analysis/                     per-model metrics plotting (CSVLogger output)
├── legacy_segmentation_workflow/ original 01_*.py driver scripts (kept for already-trained keras weights — see docs/legacy.md)
├── conf/                         shared Hydra config groups for the legacy scripts
├── upload_pytorch_models_to_hf.py  HF model-repo migration helper
├── clean_checkpoints.py          prune per-epoch ckpts after a sweep
└── linkedin_announcement.md      announcement copy for the StarDist PyTorch port
```

Each sub-folder owns its own conf tree and its own README:

- [`train_data_generation/README.md`](train_data_generation/README.md)
- [`model_training/README.md`](model_training/README.md)

## Install

```bash
pip install -e ".[scripts]"        # adds hydra-core, natsort, huggingface_hub
```

## Pretrained models

PyTorch models live under [`KapoorLabs/`](https://huggingface.co/KapoorLabs) on HuggingFace. Every predict YAML in `conf/experiment_data_paths/` exposes a `log_path` (local) and an `hf_repo_id` (remote) — disk wins when present, HF download is the fallback. The mapping `model_name → repo_id` lives in [`src/kapoorlabs_vollseg/hub.py`](../src/kapoorlabs_vollseg/hub.py).

```
KapoorLabs/xenopus-stardist-pytorch
KapoorLabs/xenopus-unet-pytorch
KapoorLabs/xenopus-maskunet-pytorch
KapoorLabs/xenopus-care-pytorch
```

### Upload / replace HF repos

`upload_pytorch_models_to_hf.py` migrates a local model folder into its HuggingFace repo. Reads `HF_TOKEN` from `scripts/.env`. Append `--dry-run` to see what would happen without touching HF.

```bash
# First-time upload of one model from the standard layout:
python scripts/upload_pytorch_models_to_hf.py \
    --source-root /lustre/.../jean-zay \
    --only models_stardist_pytorch

# Replace existing repo with a non-standard folder (e.g. a sweep winner):
python scripts/upload_pytorch_models_to_hf.py \
    --source-folder /lustre/.../models_stardist_pytorch_sweep/stardist_sweep_adam_lr1p0e-3_noscheduler \
    --only models_stardist_pytorch \
    --replace
```

The mapping `local_folder_name → HF repo_id` is the `MODELS` dict at the top of the script.

## Prediction

`scripts/model_prediction/` holds one Hydra predict script per task:

```
predict-care.py      predict-unet.py      predict-roi.py
predict-stardist.py  predict-combo.py
```

Each script reads `experiment_data_paths.log_path` first; falls back to downloading from `hf_repo_id`. Outputs nest into `<input_dir>/<output_dir>/<file>.tif`.

```bash
# Predict with the local checkpoint (default):
python scripts/model_prediction/predict-stardist.py

# Override the model folder / data on the CLI:
python scripts/model_prediction/predict-stardist.py \
    experiment_data_paths.log_path=/lustre/.../models_stardist_pytorch \
    experiment_data_paths.input_dir=/lustre/.../demo_data
```

The sweep scorers (`sweep_predict_and_analyze.py`, `sweep_predict_unet_and_analyze.py`, `sweep_predict_roi_stardist_and_analyze.py`) score every trained sweep model against a keras reference, write `sweep_predict_summary.csv` per task, and drop subset-T raw + keras companion TIFFs alongside the per-model predictions for side-by-side napari viewing.

`compare-stardist-vs-keras.py` and `compare-roi-stardist-vs-keras.py` compute per-frame regionprops (volume, equivalent radius, marching-cubes surface area) for both the PyTorch and the keras label stacks at the same T-indices and write a long-format CSV. The companion `compare_*_vs_keras.ipynb` notebooks render box plots stratified by developmental stage.

## Analysis

`analysis/plot_metrics{,_unet}.py` walks a sweep folder, reads each model's `metrics.csv` (Lightning CSVLogger), and renders per-model grids, per-metric overlays across all models, and box plots of each model's training distribution.

## Legacy

`legacy_segmentation_workflow/` keeps the original `01_*.py` driver scripts (CARE → ROI → U-Net + StarDist → seedpool → CellPose) plus their shared `conf/` groups. They route each model slot to the PyTorch first-class singleton when a `.ckpt` is configured and fall back to the `*Keras` sibling otherwise. Use only when you need to drive already-trained keras `.h5` weights from the old API; see [`docs/legacy.md`](../docs/legacy.md) for the full index of the legacy surface (HF model registry, scripts, upload helper, deprecation notes).
