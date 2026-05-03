# VollSeg

Hierarchical, composable segmentation for biological image data — a clean rewrite of the original [VollSeg](https://github.com/Kapoorlabs-CAPED/VollSeg).

```bash
pip install kapoorlabs-vollseg
```

PyTorch + PyTorch Lightning + [CAREamics](https://github.com/CAREamics/careamics) is the first-class backend. The original keras / csbdeep / stardist stack is kept as a legacy backend with a `Keras` suffix on every class name so already-trained `.h5` weights still work.

---

## Quick start

```python
import numpy as np
from tifffile import imread
from vollseg import StarDistSegmenter, MaskUNetSegmenter, VollSeg

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
| Pretrained zoo  | `vollseg.hub.XENOPUS_MODELS` (HuggingFace)            | `vollseg.pretrained` (Zenodo)                               |

Both implement the same `Pipeline.predict(image) -> Result` contract, so any composite or factory accepts either or both interchangeably.

The bare-named PyTorch classes are the supported direction. The `*Keras` variants exist to keep already-trained `.h5` weights usable. `StarDistSegmenterKeras` currently has no PyTorch counterpart in the inference path — the `vollseg.stardist` rewrite ships training, but you can keep using the keras singleton for inference until you retrain.

---

## Pretrained Xenopus model zoo (HuggingFace)

Public model repos live under `KapoorLabs-Copenhagen/` on HuggingFace. The scripts call `vollseg.ensure_model(model_dir, model_name)` for each configured model — if the directory `<model_dir>/<model_name>/` doesn't exist locally, it's downloaded automatically.

```python
from vollseg import ensure_model, XENOPUS_MODELS

ensure_model("./models/StarDist3D", "nuclei_xenopus_mari")
# → downloads from KapoorLabs-Copenhagen/xenopus-stardist3d-nuclei-mari
```

Mapping lives in [`src/vollseg/hub.py`](src/vollseg/hub.py). See [`scripts/README.md`](scripts/README.md) for the full table and one-time upload helper.

---

## Repository layout

```
KapoorLabs-VollSeg/
├── src/vollseg/
│   ├── _backbones/           csbdeep / stardist / careamics / cellpose backbone wrappers
│   ├── _lightning/           inlined Lightning support (CareModule, dataset, stitch, transforms)
│   ├── models/               Layer-1 singletons (PyTorch + Keras siblings)
│   ├── pipelines/            Layer-2 composites + Layer-3 factories
│   ├── stardist/             pure-PyTorch StarDist (rays, distance, model, losses, training, inference)
│   ├── train/                Lightning + csbdeep trainers
│   ├── data/                 file IO, label morphology, Sequence loaders, SmartPatches
│   ├── eval/                 matching metrics, NMS, threshold optimization
│   ├── fusion.py             watershed_fuse, cellpose_watershed_fuse
│   ├── hub.py                HuggingFace auto-download for the Xenopus model zoo
│   ├── pretrained.py         legacy Zenodo registry (csbdeep weights)
│   └── seedpool.py           SeedPool / UnetStarMask geometry primitives
├── scripts/                  Hydra-driven CLI: enhance, segment, score, train_stardist
├── docs/                     Per-module READMEs: care.md, unet.md, stardist.md
├── tests/                    pytest suite (PyTorch path; keras kept legacy)
├── pyproject.toml            packaging + dependencies
├── setup.cfg                 setuptools metadata
└── update_version.py         git-tag → src/vollseg/_version.py
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

The pre-commit hooks run `pyupgrade` (py39+), `black`, `flake8`, `autoflake`, plus a local `update_version.py` hook that syncs `src/vollseg/_version.py` from the most recent git tag.

---

## License

BSD-3-Clause — see [`LICENSE`](LICENSE). Same as upstream VollSeg.
