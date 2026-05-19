# VollSeg

[![PyPI version](https://img.shields.io/pypi/v/kapoorlabs-vollseg.svg?maxAge=2591000)](https://pypi.org/project/kapoorlabs-vollseg/)
[![Python versions](https://img.shields.io/pypi/pyversions/kapoorlabs-vollseg.svg)](https://pypi.org/project/kapoorlabs-vollseg/)
[![License](https://img.shields.io/pypi/l/kapoorlabs-vollseg.svg?color=green)](https://github.com/Kapoorlabs-CAPED/KapoorLabs-VollSeg/raw/main/LICENSE)
[![Deploy](https://github.com/Kapoorlabs-CAPED/KapoorLabs-VollSeg/actions/workflows/deploy.yml/badge.svg)](https://github.com/Kapoorlabs-CAPED/KapoorLabs-VollSeg/actions/workflows/deploy.yml)
[![Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/entracod)

Hierarchical, composable segmentation for biological image data — a clean rewrite of the original [VollSeg](https://github.com/Kapoorlabs-CAPED/VollSeg).

```bash
pip install kapoorlabs-vollseg                # SDK only (PyTorch first-class)
pip install kapoorlabs-vollseg[napari]        # SDK + the dock-widget plugin
pip install kapoorlabs-vollseg[keras]         # SDK + legacy keras/csbdeep backend
pip install kapoorlabs-vollseg[all]           # everything
```

PyTorch + PyTorch Lightning + [CAREamics](https://github.com/CAREamics/careamics) is the first-class backend. The original keras / csbdeep / stardist stack is kept as a legacy backend with a `Keras` suffix on every class name so already-trained `.h5` weights still work.

### What's new

- **PyTorch StarDist inference is now first-class.** End-to-end pipeline: tile + predict + stitch → peak detection → triangulated polyhedron rasterisation (matches upstream `stardist.polyhedron_to_label`) → NMS → label image. Rays use the same golden-spiral parameterisation as `stardist.Rays_GoldenSpiral` (anisotropy convention included), so weights are transferable.
- **`from_folder(path)` on every singleton.** Pairs the Lightning `.ckpt` with a `training_config.json` sidecar so `conv_dims`, `unet_depth`, `n_rays`, optimised thresholds, etc. are picked up automatically. Drop a folder on disk, call `Singleton.from_folder(folder)`, done.
- **Multi-GPU timelapse prediction.** `predict_timelapse(pipeline, volume, devices=N, strategy="ddp")` shards the T-axis across GPUs via Lightning `Trainer.predict`, gathers the per-rank outputs onto rank 0, and returns a stacked `(T, …)` result. Works for any `Pipeline` — singleton or composite.
- **HuggingFace auto-download with disk-priority.** Predict scripts read `log_path` and `hf_repo_id` from Hydra YAMLs; on-disk path wins when it exists, HF download is the fallback. The new PyTorch model repos live under [`KapoorLabs`](https://huggingface.co/KapoorLabs) (e.g. `KapoorLabs/xenopus-stardist-pytorch`, `KapoorLabs/xenopus-unet-pytorch`, `KapoorLabs/xenopus-maskunet-pytorch`); the legacy keras Xenopus zoo stays under [`KapoorLabs-Copenhagen`](https://huggingface.co/KapoorLabs-Copenhagen).
- **StarDist threshold optimisation, cached.** `scripts/model_training/optimize-stardist-thresholds.py` runs the network **once per validation patch**, precomputes peaks + rasterised polyhedra once at the lowest probability the sweep will visit, and reuses them across every `(prob_thresh, nms_thresh)` candidate. Writes results back into `training_config.json` so prediction picks them up automatically.
- **Predict scripts for every singleton + the composite**: `scripts/model_prediction/predict-{care,roi,unet,stardist,combo}.py`. All Hydra-driven, all support multi-GPU timelapse, all nest their output inside the input directory so files don't sprawl.

**Local checkout** — when developing from a git clone the napari extra cannot resolve from PyPI yet, so install the plugin from the in-repo path instead:

```bash
pip install -e .                              # SDK
pip install -e plugins/napari-vollseg         # add the segmentation napari plugin
pip install -e plugins/napari-curvature       # add the curvature napari plugin
```

---

## Quick start

```python
import numpy as np
from tifffile import imread
from kapoorlabs_vollseg import StarDistSegmenter, MaskUNetSegmenter, VollSeg

# Layer-1 singletons load themselves from Lightning checkpoints (PyTorch)
# or from the Zenodo / HuggingFace pretrained registry (legacy keras).
stardist = StarDistSegmenter.from_checkpoint(
    "models/nuclei.ckpt",
    rays=np.load("models/nuclei.rays.npy"),
)
roi = MaskUNetSegmenter.from_checkpoint("models/roi.ckpt")

# Layer-3 factory composes the right pipeline shape from the supplied models.
pipe = VollSeg.from_models(
    stardist=stardist,
    roi_unet=roi,         # → wraps in ROIPipeline
    seedpool=False,
)

result = pipe.predict(imread("data/sample.tif"))
print(result.labels.max(), "objects")
```

---

## Why a rewrite?

The original VollSeg grew organically into a single `utils.py` with branching `if/else` chains for every combination of *denoise / ROI / U-Net / StarDist / seedpool / 2D / 3D*. Adding a mode meant editing the same mega-functions; testing one path required mocking the rest.

This rewrite replaces that with three orthogonal layers, composed at runtime.

---

## Architecture

```
                    ┌─────────────────────────────────┐
        Layer 3     │  VollSeg.from_models(...)       │   smart factory
                    │  → assembles the right pipeline │
                    │  VollCellSeg.from_models(...)   │   sibling, for membrane
                    └─────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────────┐
        Layer 2     │  Composite Pipelines            │   composition, not inheritance
                    │  • UNetStarDistPipeline         │
                    │  • NucleiSeededCellPosePipeline │
                    │  • DenoisedPipeline             │
                    │  • ROIPipeline                  │
                    │  • Chunked                      │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────┴──────────────────┐
        Layer 1     │  Singleton Models               │   one model, one job
                    │  • CAREDenoiser                 │
                    │  • UNetSegmenter                │
                    │  • MaskUNetSegmenter            │
                    │  • StarDistSegmenter            │
                    │  • CellPoseSegmenter            │
                    └─────────────────────────────────┘
```

### Layer 1 — Singletons

Identical contract:

```python
class Pipeline(Protocol):
    def predict(self, image: np.ndarray, **kwargs) -> Result: ...
```

| Class                | Job                                       | Output (`Result.*`)                |
| -------------------- | ----------------------------------------- | ---------------------------------- |
| `CAREDenoiser`       | Denoise (CAREamics UNet, Lightning)       | `denoised`                         |
| `UNetSegmenter`      | Binary semantic segmentation + CC labels  | `labels`, `semantic`, `probability`|
| `MaskUNetSegmenter`  | Same as `UNetSegmenter`, separate weights | `labels`, `semantic`, `probability`|
| `StarDistSegmenter`  | Instance segmentation via radial dists    | `labels`, `probability`            |
| `CellPoseSegmenter`  | Membrane/cell segmentation (CellPose)     | `labels`                           |

2D vs 3D is dispatched **inside** each singleton on `image.ndim` — no parallel `*2D` / `*3D` class trees.

### Layer 2 — Composites (built by wrapping, not subclassing)

```python
# StarDist + U-Net, fused via SeedPool watershed
pipe = UNetStarDistPipeline(unet, stardist, seedpool=True)

# ...preceded by CARE denoising
pipe = DenoisedPipeline(care, downstream=pipe)

# ...gated by an ROI mask
pipe = ROIPipeline(roi_unet, downstream=pipe)

# ...executed in overlapping chunks for huge volumes
pipe = Chunked(pipe, chunk=(64, 256, 256), overlap=(8, 32, 32))
```

| Composite                       | Wraps                | What it adds                                                |
| ------------------------------- | -------------------- | ----------------------------------------------------------- |
| `UNetStarDistPipeline`          | unet + stardist      | Runs both; if `seedpool=True`, fuses via watershed/IoU      |
| `NucleiSeededCellPosePipeline`  | nuclei pipe + cellpose | Nuclei labels seed a CellPose-gated membrane watershed    |
| `DenoisedPipeline`              | any downstream       | CARE denoise → downstream                                   |
| `ROIPipeline`                   | any downstream       | U-Net mask → downstream restricted to ROI                   |
| `Chunked`                       | any downstream       | Overlapping tiles → predict → label-safe stitch             |

### Layer 3 — Smart factories

```python
pipe = VollSeg.from_models(
    care=care_model,         # optional → wraps in DenoisedPipeline
    roi_unet=roi_model,      # optional → wraps in ROIPipeline
    unet=unet_model,         # optional
    stardist=stardist_model, # optional
    seedpool=True,           # only meaningful with both unet+stardist
    chunk=(64, 256, 256),    # optional → wraps in Chunked
)

# Sibling factory for membrane work — consumes a nuclei pipeline as input.
pipe = VollCellSeg.from_models(
    nuclei_pipeline=nuclei_pipe,
    cellpose=cellpose_model,
    care=membrane_denoiser,  # optional
    nuclei_channel=1, membrane_channel=0,
)
```

Rule: provided models determine the pipeline shape; runtime knobs tune behavior. No silent fallbacks — invalid combinations raise at construction, not at `.predict`.

---

## Two backends — PyTorch first-class, Keras legacy

| Concern         | PyTorch (default)                                     | Keras legacy                                                |
| --------------- | ----------------------------------------------------- | ----------------------------------------------------------- |
| Class names     | `CAREDenoiser`, `UNetSegmenter`, …                    | `CAREDenoiserKeras`, `UNetSegmenterKeras`, …                |
| Model arch      | CAREamics UNet + Lightning                            | csbdeep CARE / stardist                                     |
| Checkpoints     | `.ckpt` (Lightning) — `from_checkpoint(path)`         | csbdeep folder (`config.json` + `weights_*.h5`)             |
| Pretrained zoo  | `kapoorlabs_vollseg.hub.XENOPUS_MODELS` (HuggingFace)            | `kapoorlabs_vollseg.pretrained` (Zenodo)                               |

Both implement the same `Pipeline.predict(image) -> Result` contract, so any composite or factory accepts either or both interchangeably.

The bare-named PyTorch classes are the supported direction. The `*Keras` variants exist to keep already-trained `.h5` weights usable. Both backends now cover training **and** inference for every model — including StarDist 3D, which uses a triangulated star-convex polyhedron rasteriser equivalent to upstream `stardist.polyhedron_to_label`.

---

## Prediction — `from_folder`, multi-GPU, HF auto-download

Every singleton exposes a `from_folder` constructor that pairs a Lightning `.ckpt` with the `training_config.json` sidecar the trainer writes. The loader picks up architecture knobs (`conv_dims`, `unet_depth`, `in_channels`, …), per-model thresholds (StarDist `prob_thresh` / `nms_thresh`), and — for StarDist — the `rays.npy` so inference reuses the same ray geometry the model was trained on.

```python
from kapoorlabs_vollseg import StarDistSegmenter, predict_timelapse

# Single folder: ckpt + training_config.json + rays.npy.
star = StarDistSegmenter.from_folder("models/xenopus_stardist/")

# Single 3D volume.
result = star.predict(imread("frame.tif"))

# 4D (TZYX) timelapse, sharded across 4 GPUs.
out = predict_timelapse(star, imread("timelapse.tif"),
                        devices=4, strategy="ddp")
labels_tzyx = out["labels"]   # full (T, Z, Y, X) stack on rank 0
```

`predict_timelapse` wraps any `Pipeline` (singleton or composite) in a thin `TimelapsePredictor` `LightningModule` and dispatches it via `Trainer.predict` with a `DistributedSampler` over T. Per-rank outputs are gathered onto rank 0 via `torch.distributed.gather_object` (so the 35 GB stack only lives on one rank), deduped against sampler-padding, sorted by T, and stacked.

Each predict script (`scripts/model_prediction/predict-{care,roi,unet,stardist,combo}.py`) supports the same priority on its `log_path` / `hf_repo_id` YAML entries: **disk path wins when it exists**; HF download is the fallback. Outputs land in `<input_dir>/<output_dir>/<file>.tif` so segmentation results are nested inside the raw folder.

---

## Pretrained Xenopus model zoo (HuggingFace)

Two orgs, two backends:

- [**`KapoorLabs/`**](https://huggingface.co/KapoorLabs) — the **new PyTorch** model repos used by the predict scripts (`xenopus-stardist-pytorch`, `xenopus-unet-pytorch`, `xenopus-maskunet-pytorch`, …). Auto-downloaded when a script's `hf_repo_id` is set and the local `log_path` doesn't exist on disk.
- [**`KapoorLabs-Copenhagen/`**](https://huggingface.co/KapoorLabs-Copenhagen) — the **legacy keras / csbdeep / stardist** models published with the original paper, kept around so already-trained `.h5` weights keep working. Resolved via `kapoorlabs_vollseg.ensure_model` from the `XENOPUS_MODELS` registry.

```python
# Legacy keras zoo:
from kapoorlabs_vollseg import ensure_model, XENOPUS_MODELS

ensure_model("./models/StarDist3D", "nuclei_xenopus_mari")
# → downloads from KapoorLabs-Copenhagen/xenopus-stardist3d-nuclei-mari
```

The legacy registry mapping lives in [`src/kapoorlabs_vollseg/hub.py`](src/kapoorlabs_vollseg/hub.py); the new PyTorch repos are addressed by the `hf_repo_id` entry in each predict YAML under `scripts/conf/experiment_data_paths/`. See [`scripts/README.md`](scripts/README.md) for the full table.

---

## Curvature & force profiles

Segmentation is a means, not an end — once you have labels you usually want to *measure* something. The `kapoorlabs_vollseg.curvature` toolkit takes a label image (2D or 3D) and returns, for each region, a sliding-window curvature profile along its boundary or surface, plus optional Young-Laplace pressure and Helfrich bending-energy profiles when material constants are supplied.

```python
from tifffile import imread
from kapoorlabs_vollseg.curvature import compute_curvature

labels = imread("data/segmented_cells.tif")

profiles = compute_curvature(
    labels,
    spacing=(2.0, 0.6918, 0.6918),   # (dz, dy, dx) μm
    n_window=21, stride=5,
    geodesic=True,                    # mesh-aware neighbours in 3D
    surface_tension=1e-3,             # N/m — optional → Young-Laplace ΔP
    bending_modulus=2e-20,            # J   — optional → Helfrich f
)

for label_id, profile in profiles.items():
    print(label_id, profile.summary())
    # profile.centers, .kappa, .normals, .radii   — geometry
    # profile.pressure                            — γκ (2D) or 2γH (3D)
    # profile.bending_density                     — κ_b·(2H-C₀)² + κ_G·K
```

Pipeline:

- **2D** — `skimage.measure.find_contours` per label → ordered sub-pixel contour → sliding window of `n_window` consecutive points → Kasa algebraic circle fit → `κ = ±1/r` (sign from `dot(radius_vec, outward_normal)`).
- **3D** — `skimage.measure.marching_cubes` per label → triangle mesh + per-vertex outward normals → at every `stride`-th vertex, the `n_window` nearest neighbours by **geodesic distance** along the mesh (BFS-hop default, Dijkstra optional, or Euclidean KDTree as opt-in) → Coope linear sphere fit → signed mean curvature.
- **Physics** is bolt-on: pass `surface_tension` to get a Young-Laplace pressure column, pass `bending_modulus` (and optionally `spontaneous_curvature` / `saddle_splay_modulus`) for a Helfrich bending-energy column. Both are skipped when their constants are absent.

Anisotropic voxels are first-class: pass `spacing=(dz, dy, dx)` and the resulting curvatures come out in `1/length` of that unit (so feed μm in, get 1/μm out).

---

## Repository layout

```
KapoorLabs-VollSeg/
├── src/kapoorlabs_vollseg/
│   ├── _backbones/           csbdeep / stardist / careamics / cellpose backbone wrappers
│   ├── _lightning/           inlined Lightning support (CareModule, dataset, stitch, transforms)
│   ├── models/               Layer-1 singletons (PyTorch + Keras siblings)
│   ├── pipelines/            Layer-2 composites + Layer-3 factories
│   ├── stardist/             pure-PyTorch StarDist (rays, distance, model, losses, training, inference)
│   ├── curvature/            per-label curvature + Young-Laplace / Helfrich force profiles
│   ├── train/                Lightning + csbdeep trainers
│   ├── data/                 file IO, label morphology, Sequence loaders, SmartPatches
│   ├── eval/                 matching metrics, NMS, threshold optimization
│   ├── fusion.py             watershed_fuse, cellpose_watershed_fuse
│   ├── hub.py                HuggingFace auto-download for the Xenopus model zoo
│   ├── pretrained.py         legacy Zenodo registry (csbdeep weights)
│   └── seedpool.py           SeedPool / UnetStarMask geometry primitives
├── plugins/
│   └── napari/               kapoorlabs-vollseg-napari — QTabWidget dock plugin (PyTorch-only)
├── scripts/                  Hydra-driven CLI: enhance, segment, score, train_stardist
├── docs/                     Per-module READMEs: care.md, unet.md, stardist.md
├── tests/                    pytest suite (PyTorch path; keras kept legacy)
├── pyproject.toml            packaging + dependencies
├── setup.cfg                 setuptools metadata
└── update_version.py         git-tag → src/kapoorlabs_vollseg/_version.py
```

---

## Documentation

- [`docs/care.md`](docs/care.md) — CARE denoising in PyTorch (Backbone, Singleton, Trainer)
- [`docs/unet.md`](docs/unet.md) — U-Net + MaskUNet semantic segmentation
- [`docs/stardist.md`](docs/stardist.md) — full PyTorch StarDist rewrite (algorithm, training, inference)
- [`scripts/README.md`](scripts/README.md) — Hydra segmentation pipelines + the StarDist demo script + HF model upload

---

## Design rules

1. **Composition over inheritance** for combining behaviors — wrap, don't subclass.
2. **One responsibility per class.** A class either trains, or predicts, or composes — never two.
3. **No 2D/3D class duplication.** Dispatch on `ndim` inside the class.
4. **Runtime concerns are decorators.** Chunking, ROI gating, and denoising all wrap a downstream pipeline; none are baked into the singletons.
5. **Fail at construction, not prediction.** Invalid model combinations raise in `from_models`, not mid-inference.
6. **No silent fallbacks.** If a user asks for `seedpool=True` without both models, raise.
7. **Trainers produce models; they are not models.**

---

## Development

```bash
git clone https://github.com/Kapoorlabs-CAPED/KapoorLabs-VollSeg
cd KapoorLabs-VollSeg
pip install -e ".[testing]"
pre-commit install
pytest tests/ -v
```

The pre-commit hooks run `pyupgrade` (py39+), `black`, `flake8`, `autoflake`, plus a local `update_version.py` hook that syncs `src/kapoorlabs_vollseg/_version.py` from the most recent git tag.

---

## License

BSD-3-Clause — see [`LICENSE`](LICENSE). Same as upstream VollSeg.
