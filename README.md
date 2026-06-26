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
| `DenoisedPipeline`              | any downstream         | CARE denoise → downstream sees the denoised image        |
| `ROIPipeline`                   | any downstream         | Mask-UNet ROI bbox → downstream on the crop, paste back  |
| `UNetStarDistPipeline`          | stardist (+ optional unet) | Side-by-side or seed-pool watershed fusion           |
| `NucleiSeededCellPosePipeline`  | nuclei pipe + cellpose | Nuclei labels seed a CellPose-gated membrane watershed   |
| `Chunked`                       | any downstream         | Overlapping tiles → predict → label-safe stitch          |

### Composition order

`VollSeg.from_models(...)` always nests in the same order — only the
stages whose models you supply appear in the chain:

```
          ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────────┐    ┌────────┐
image ──▶ │ Chunked │──▶ │ Denoised │──▶ │   ROI   │──▶ │ segmentation │──▶ │ Result │
          │ (chunk) │    │  (care)  │    │(roi_unet│    │ core         │    └────────┘
          └─────────┘    └──────────┘    └─────────┘    │ (stardist /  │
                                                       │  unet / fused│
                                                       └──────────────┘
```

Each box is optional and only appears when its model / chunk shape is
supplied. Image always flows left → right; every stage downstream of
CARE sees the **denoised** image, not the raw one.

Predict-time flow:

1. **Chunked** (optional) — split big volumes into overlapping tiles.
2. **DenoisedPipeline** (optional) — CARE denoises the chunk; every
   downstream stage sees the **denoised** image, never the raw one.
3. **ROIPipeline** (optional) — Mask-UNet predicts an ROI mask on the
   denoised image; downstream runs on the bounding-box crop and the
   labels are pasted back into the full-shape array.
4. **Segmentation core** — StarDist alone, U-Net alone, or the
   `UNetStarDistPipeline` composite (side-by-side or fused).

### Segmentation core: how the toggles map to a pipeline

```
                          unet supplied?
                          ┌─────────┴─────────┐
                         yes                  no
                          │                   │
                  ┌───────┴────────┐  ┌───────┴────────┐
                  │ stardist + unet│  │ stardist only  │
                  └───────┬────────┘  └───────┬────────┘
                          │                   │
              seedpool? ──┤                   ├── seedpool?
                          │                   │
             ┌──── T ─────┤                   ├──── T ────┐
             │            │                   │           │
             │     ┌──── F                    F ────┐     │
             │     │                                │     │
             ▼     ▼                                ▼     ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
   │ UNet+StarDist    │  │ UNet+StarDist    │  │ UNetStarDist:        │
   │ + watershed fuse │  │ side-by-side     │  │ Otsu seed pool +     │
   │ (classic VollSeg)│  │ (no fusion)      │  │ watershed fuse       │
   └──────────────────┘  └──────────────────┘  └──────────────────────┘
                                               ┌──────────────────────┐
                                               │ bare StarDist        │
                                               │ (seedpool ignored)   │
                                               └──────────────────────┘
```

### Result fields per scenario

`Result.*` fields are populated **only when the corresponding model
runs** — otherwise they stay `None`. The factory's job is to pick a
pipeline shape that produces the maximal set of fields for the
supplied models.

| `care` | `roi_unet` | `unet` | `stardist` | `seedpool` | Pipeline composition (outer → inner)                                                  | `Result` fields populated                                                              |
| :----: | :--------: | :----: | :--------: | :--------: | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| ✓      | ✓          | ✓      | ✓          |     T      | `Denoised( ROI( UNetStarDist(unet, stardist, seedpool=T) ) )`                          | `labels = vollseg_labels`, `stardist_labels`, `unet_labels`, `semantic`, `denoised`, `roi`, `polys` |
| ✗      | ✗          | ✓      | ✓          |     T      | `UNetStarDist(unet, stardist, seedpool=T)`                                              | `labels = vollseg_labels`, `stardist_labels`, `unet_labels`, `semantic`, `polys`        |
| ✓      | ✗          | ✗      | ✓          |     T      | `Denoised( UNetStarDist(unet=None, stardist, seedpool=T) )` — Otsu threshold seed pool  | `labels = vollseg_labels`, `stardist_labels`, `semantic` (Otsu), `denoised`, `polys`    |
| ✗      | ✗          | ✓      | ✓          |     F      | `UNetStarDist(unet, stardist, seedpool=F)` — side by side                               | `labels = stardist_labels`, `stardist_labels`, `unet_labels`, `semantic`, `polys`       |
| ✓      | ✓          | ✗      | ✓          |     F      | `Denoised( ROI( stardist ) )` — bare StarDist on denoised ROI crop                      | `labels`, `denoised`, `roi`, `polys`                                                    |
| ✓      | ✗          | ✗      | ✓          |     F      | `Denoised( stardist )` — denoise then StarDist                                          | `labels`, `denoised`, `polys`                                                            |
| ✓      | ✗          | ✓      | ✗          | any        | `Denoised( unet )` — denoise then U-Net                                                  | `labels`, `semantic`, `probability`, `denoised`                                          |
| ✗      | ✗          | ✓      | ✗          | any        | bare U-Net singleton (`seedpool` ignored, no `stardist` to fuse)                          | `labels`, `semantic`, `probability`                                                      |
| ✗      | ✗          | ✗      | ✓          | any        | bare StarDist singleton (`seedpool` ignored — no `unet` and no `care` to source a mask)  | `labels`, `probability`, `polys`                                                         |
| ✓      | ✗          | ✗      | ✗          | any        | bare CARE singleton — "denoise as the whole pipeline"                                    | `denoised`                                                                               |
| ✗      | ✓          | ✗      | ✗          | any        | bare Mask-UNet singleton — its output is the ROI mask itself                             | `labels`, `semantic`, `probability`                                                      |
| ✗      | ✗          | ✗      | ✗          | any        | **`ValueError`** — no model to do anything with                                          | —                                                                                        |

Permissive rules — the **only** failure mode is "no model supplied":

- `seedpool=True` is **silently ignored** when its prerequisites aren't
  met (no `stardist` to fuse; or no `unet` AND no `care` to source the
  mask). The factory falls back to the next-best shape.
- Any single-model configuration returns the bare singleton.
- Composition only kicks in when there's actually something to compose.

```python
# Production pipeline for embryo timelapses — denoise, ROI-gate, segment.
pipe = VollSeg.from_models(
    care=care, roi_unet=roi, stardist=star, unet=unet,
    seedpool=True,                  # auto-ignored if prerequisites missing
    chunk=(64, 256, 256),           # optional → wraps in Chunked
)
result = pipe.predict(image)
result.vollseg_labels   # watershed-fused instance labels (canonical = labels)
result.stardist_labels  # raw StarDist instances
result.unet_labels      # CC labels of U-Net mask
result.semantic         # U-Net binary mask
result.denoised         # CARE output
result.roi              # ROI gating mask
```

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

The keras / csbdeep / stardist `.h5` Xenopus zoo, the `*Keras` singleton siblings, and the original `01_*.py` driver scripts (the pre-rewrite VollSeg workflow) all live inside this repo — see [`docs/legacy.md`](docs/legacy.md) for the full index (registry of HF model repos, driver scripts, upload helper, when to fall back to it). Use it only if you have already-trained `.h5` weights you can't retrain. Everything else — new training, new prediction, the napari plugins, the `KapoorLabs/` HuggingFace zoo — goes through the PyTorch path documented above.

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
