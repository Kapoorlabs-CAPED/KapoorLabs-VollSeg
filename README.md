# VollSeg

[![PyPI version](https://img.shields.io/pypi/v/kapoorlabs-vollseg.svg?maxAge=2591000)](https://pypi.org/project/kapoorlabs-vollseg/)
[![Python versions](https://img.shields.io/pypi/pyversions/kapoorlabs-vollseg.svg)](https://pypi.org/project/kapoorlabs-vollseg/)
[![License](https://img.shields.io/pypi/l/kapoorlabs-vollseg.svg?color=green)](https://github.com/Kapoorlabs-CAPED/KapoorLabs-VollSeg/raw/main/LICENSE)

Hierarchical, composable segmentation for biological image data — PyTorch + PyTorch Lightning + [CAREamics](https://github.com/CAREamics/careamics).

```bash
pip install kapoorlabs-vollseg            # SDK
pip install kapoorlabs-vollseg[napari]    # SDK + napari dock plugin
pip install kapoorlabs-vollseg[all]       # everything
```

## Quick start

```python
from tifffile import imread
from kapoorlabs_vollseg import (
    StarDistSegmenter, MaskUNetSegmenter, VollSeg, predict_timelapse,
)

star = StarDistSegmenter.from_folder("models/xenopus_stardist/")
roi  = MaskUNetSegmenter.from_folder("models/xenopus_maskunet/")

# Compose: ROI → StarDist (the production pipeline for embryo timelapses).
pipe = VollSeg.from_models(stardist=star, roi_unet=roi)

# Single volume.
labels = pipe.predict(imread("frame.tif")).labels

# Timelapse sharded across all visible GPUs.
out = predict_timelapse(pipe, imread("timelapse.tif"),
                        devices=-1, strategy="ddp")
labels_tzyx = out["labels"]
```

`from_folder` reads a Lightning `.ckpt` plus a `training_config.json` sidecar that records architecture knobs (`conv_dims`, `unet_depth`, `n_rays`, `anisotropy`, tuned `prob_thresh` / `nms_thresh`). StarDist rays are regenerated deterministically from `(conv_dims, n_rays, anisotropy)`; no `rays.npy` sidecar is needed.

## Architecture

Three orthogonal layers. Composition, not inheritance.

```
Layer 3   VollSeg.from_models / VollCellSeg.from_models           smart factories
Layer 2   ROIPipeline · UNetStarDistPipeline · DenoisedPipeline   composites
          NucleiSeededCellPosePipeline · Chunked
Layer 1   CAREDenoiser · UNetSegmenter · MaskUNetSegmenter        singletons
          StarDistSegmenter · CellPoseSegmenter
```

All singletons + composites implement the same protocol:

```python
class Pipeline(Protocol):
    def predict(self, image: np.ndarray, **kwargs) -> Result: ...
```

| Singleton            | Output (`Result.*`)                  |
| -------------------- | ------------------------------------ |
| `CAREDenoiser`       | `denoised`                           |
| `UNetSegmenter`      | `labels`, `semantic`, `probability`  |
| `MaskUNetSegmenter`  | `labels`, `semantic`, `probability`  |
| `StarDistSegmenter`  | `labels`, `probability`              |
| `CellPoseSegmenter`  | `labels`                             |

2D vs 3D is dispatched on `image.ndim` inside each singleton — no parallel class trees.

| Composite                       | Wraps                  | Adds                                                     |
| ------------------------------- | ---------------------- | -------------------------------------------------------- |
| `ROIPipeline`                   | any downstream         | Mask-UNet ROI bbox → downstream on the crop              |
| `UNetStarDistPipeline`          | unet + stardist        | Runs both; `seedpool=True` fuses via watershed/IoU       |
| `NucleiSeededCellPosePipeline`  | nuclei pipe + cellpose | Nuclei labels seed a CellPose-gated membrane watershed   |
| `DenoisedPipeline`              | any downstream         | CARE denoise → downstream                                |
| `Chunked`                       | any downstream         | Overlapping tiles → predict → label-safe stitch          |

Layer 3 picks the pipeline shape from the supplied models:

```python
pipe = VollSeg.from_models(
    care=care, roi_unet=roi, unet=unet, stardist=star,
    seedpool=True,                          # needs both unet + stardist
    chunk=(64, 256, 256),                   # optional → wraps in Chunked
)
```

Invalid combinations raise at construction, not mid-predict.

## StarDist — PyTorch port

End-to-end PyTorch reimplementation matching upstream `stardist` at the file level (Schmidt & Weigert, MICCAI 2020). See [`src/kapoorlabs_vollseg/stardist/README.md`](src/kapoorlabs_vollseg/stardist/README.md) for the algorithmic notes (rays, CSBDeep tile iterator port, kernel + convex-hull short-circuit polyhedron rasterizer, paint rule, anisotropy handling). The vendored upstream C++ kernel sources live under `src/kapoorlabs_vollseg/stardist/_lib/` for diffing and optional native compilation; **zero runtime dependency on the `stardist` package** in the PyTorch path.

For embryo timelapses with early-stage mostly-empty frames, wrap StarDist in `ROIPipeline(roi_unet=mask_unet, downstream=star)` — the Mask-UNet bbox prevents the whole-volume percentile-normalisation saturation that otherwise inflates polyhedra on near-empty frames. Validated against the legacy keras reference: mean cell volume / radius / surface area within ~2 % across every developmental stage.

## Prediction

Every singleton + the composites accept the same `from_folder(path)` constructor. Predict scripts in `scripts/model_prediction/` are Hydra-driven and accept either a local `log_path` (wins when present) or an `hf_repo_id` (HuggingFace fallback).

```python
from kapoorlabs_vollseg import StarDistSegmenter, ensure_model, predict_timelapse

# Pull from HF if not on disk yet, then load.
folder = ensure_model("./local_models", "xenopus-stardist-pytorch",
                      repo_id="KapoorLabs/xenopus-stardist-pytorch")
star = StarDistSegmenter.from_folder(folder)

# Timelapse on all visible GPUs.
out = predict_timelapse(star, imread("timelapse.tif"),
                        devices=-1, strategy="ddp")
```

`predict_timelapse` wraps any `Pipeline` in a `TimelapsePredictor` `LightningModule`, dispatches via `Trainer.predict` with a `DistributedSampler` over the T axis, gathers per-rank outputs onto rank 0, dedupes against sampler padding, and returns one stacked `(T, …)` array per `Result` field.

Pretrained models live under [`KapoorLabs/`](https://huggingface.co/KapoorLabs) on HuggingFace:

```
KapoorLabs/xenopus-stardist-pytorch
KapoorLabs/xenopus-unet-pytorch
KapoorLabs/xenopus-maskunet-pytorch
KapoorLabs/xenopus-care-pytorch
```

## Curvature & force profiles

After segmentation, `kapoorlabs_vollseg.curvature` computes per-label sliding-window curvature profiles along boundaries (2D) or surfaces (3D), plus optional Young-Laplace pressure and Helfrich bending-energy columns when material constants are supplied.

```python
from kapoorlabs_vollseg.curvature import compute_curvature

profiles = compute_curvature(
    labels,
    spacing=(2.0, 0.6918, 0.6918),  # (dz, dy, dx) μm
    n_window=21, stride=5,
    geodesic=True,                   # mesh-aware neighbours in 3D
    surface_tension=1e-3,            # N/m  — optional, adds Young-Laplace pressure
    bending_modulus=2e-20,           # J    — optional, adds Helfrich f
)
```

Algorithm: 2D → `find_contours` → Kasa algebraic circle fit per window; 3D → `marching_cubes` → geodesic-neighbour Coope sphere fit per vertex. Anisotropic spacing is first-class — pass μm in, get 1/μm out.

## Repository layout

```
KapoorLabs-VollSeg/
├── src/kapoorlabs_vollseg/
│   ├── models/             Layer-1 singletons
│   ├── pipelines/          composites + factories (factory.py, cellseg_factory.py)
│   ├── stardist/           PyTorch StarDist (rays, model, losses, inference, _tiling, _lib/)
│   ├── _backbones/         careamics / stardist / maskunet wrappers + _config.py loader
│   ├── _lightning/         CareModule, dataset, stitch, transforms
│   ├── care_lightning/     vendored CARE Lightning module + signal-handling Trainer
│   ├── training/           TrainingPipeline — Hydra-friendly Lightning fit loop
│   ├── curvature/          per-label curvature + force profiles
│   ├── data/               file IO, label morphology, SmartPatches H5 generator
│   ├── eval/               matching metrics, threshold optimisation primitives
│   ├── fusion.py           watershed_fuse, cellpose_watershed_fuse
│   ├── hub.py              HuggingFace ensure_model + pretrained registry
│   └── seedpool.py         SeedPool / UnetStarMask geometry
├── plugins/
│   ├── napari-vollseg/     segmentation dock plugin
│   └── napari-curvature/   curvature dock plugin
├── scripts/                Hydra CLI for training, prediction, comparison, HF upload
├── docs/                   per-module deep-dives (care, unet, stardist)
└── tests/                  pytest suite — PyTorch path
```

## Documentation

- [`src/kapoorlabs_vollseg/stardist/README.md`](src/kapoorlabs_vollseg/stardist/README.md) — StarDist re-implementation dev notes
- [`docs/care.md`](docs/care.md) — CARE denoising (Backbone, Singleton, Trainer)
- [`docs/unet.md`](docs/unet.md) — U-Net + MaskUNet semantic segmentation
- [`docs/stardist.md`](docs/stardist.md) — StarDist algorithm walkthrough
- [`scripts/README.md`](scripts/README.md) — Hydra pipelines, HF upload, comparison scripts

## Legacy code & pretrained models

Anything pre-VollSeg (the original organic `utils.py` with branching for every denoise / ROI / U-Net / StarDist / seedpool combination, plus the keras / csbdeep / stardist `.h5` Xenopus zoo) lives in **[CopenhagenWorkflow](https://github.com/Kapoorlabs-CAPED/CopenhagenWorkflow)**. Use it only if you have already-trained `.h5` weights you can't retrain. Everything else — new training, new prediction, the napari plugins, the HuggingFace zoo — goes through this repo.

## Development

```bash
git clone https://github.com/Kapoorlabs-CAPED/KapoorLabs-VollSeg
cd KapoorLabs-VollSeg
pip install -e ".[testing]"
pre-commit install
pytest tests/ -v
```

Pre-commit runs `pyupgrade` (py39+), `black`, `flake8`, `autoflake`, plus a local `update_version.py` hook that syncs `src/kapoorlabs_vollseg/_version.py` from the most recent git tag.

## License

BSD-3-Clause — see [`LICENSE`](LICENSE).
