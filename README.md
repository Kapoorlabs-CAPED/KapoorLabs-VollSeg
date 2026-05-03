# VollSeg

A clean, hierarchical, composable rewrite of [VollSeg](https://github.com/Kapoorlabs-CAPED/VollSeg) for biological image segmentation.

`pip install vollseg` (versioning continues from the original VollSeg — this package supersedes it.)

> **Status:** Early scaffolding. The original VollSeg lives at `Kapoorlabs-CAPED/VollSeg` and remains the source of truth until a `1.0.0` release of this repo. There is **no migration shim** — this is a clean rewrite with a new API.

---

## Why a rewrite?

The original VollSeg grew organically into a single `utils.py` with branching `if/else` chains for every combination of *denoise / ROI / U-Net / StarDist / seedpool / 2D / 3D*. Adding a new mode meant editing the same few mega-functions; testing a single path required mocking the rest.

This rewrite replaces that with three orthogonal layers, composed at runtime.

---

## Architecture

```
                    ┌─────────────────────────────────┐
        Layer 3     │  VollSeg.from_models(...)       │   smart factory
                    │  → assembles the right pipeline │
                    └─────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────────┐
        Layer 2     │  Composite Pipelines            │   composition, not inheritance
                    │  • UNetStarDistPipeline         │
                    │  • DenoisedPipeline(wraps any)  │
                    │  • ROIPipeline(wraps any)       │
                    │  • Chunked(wraps any)           │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────┴──────────────────┐
        Layer 1     │  Singleton Models               │   one model, one job
                    │  • CAREDenoiser                 │
                    │  • UNetSegmenter                │
                    │  • StarDistSegmenter            │
                    └─────────────────────────────────┘
```

### Layer 1 — Singletons

Each does exactly one thing. Identical contract:

```python
class Segmenter(Protocol):
    def predict(self, image: np.ndarray, **runtime_opts) -> Result: ...
```

| Class                | Job                                    | Output                  |
| -------------------- | -------------------------------------- | ----------------------- |
| `CAREDenoiser`       | Denoise (CSBDeep CARE)                 | denoised image          |
| `UNetSegmenter`      | Semantic segmentation + CC labeling    | labels + probability    |
| `StarDistSegmenter`  | Instance segmentation (radial dist.)   | labels + polygons/polys |

2D vs 3D is dispatched **inside** each singleton on `image.ndim` — no parallel `*2D` / `*3D` class trees.

### Layer 2 — Composite Pipelines

Composites are built by **wrapping** other pipelines (composition), not by subclassing. This keeps combinations linear instead of exploding (`denoise × roi × seedpool × 2d/3d` would otherwise be 16 subclasses).

```python
# StarDist + U-Net, fused via SeedPool
pipe = UNetStarDistPipeline(unet, stardist, seedpool=True)

# ...preceded by denoising
pipe = DenoisedPipeline(care, downstream=pipe)

# ...gated by an ROI mask
pipe = ROIPipeline(roi_unet, downstream=pipe)

# ...executed in overlapping chunks for huge volumes
pipe = Chunked(pipe, chunk=(64, 256, 256), overlap=(8, 32, 32))

result = pipe.predict(image)
```

| Composite                | Wraps             | What it adds                                                |
| ------------------------ | ----------------- | ----------------------------------------------------------- |
| `UNetStarDistPipeline`   | unet + stardist   | Runs both; if `seedpool=True`, fuses via watershed/IoU      |
| `DenoisedPipeline`       | any downstream    | CARE denoise → downstream                                   |
| `ROIPipeline`            | any downstream    | U-Net mask → downstream restricted to ROI                   |
| `Chunked`                | any downstream    | Overlapping tiles → predict → stitch (label-safe)           |

### Layer 3 — Smart factory

Replaces the old monolithic `VollSeg()` if/else router. Inspects which models are provided and assembles the right composite chain:

```python
pipe = VollSeg.from_models(
    care=care_model,         # optional → wraps in DenoisedPipeline
    roi_unet=roi_model,      # optional → wraps in ROIPipeline
    unet=unet_model,         # required for semantic / seedpool
    stardist=stardist_model, # required for instance
    seedpool=True,           # only meaningful when both unet+stardist
    chunked=dict(chunk=..., overlap=...),  # optional → wraps in Chunked
)

result = pipe.predict(image)
```

Rule: provided models determine the pipeline shape; runtime options (`seedpool`, `chunked`, ...) tune behavior. No silent fallbacks — missing required models raise at construction time, not at `.predict`.

---

## Training

Training lives in a sibling subpackage and is **completely separated** from inference. A trainer is not a model — it produces one.

```
vollseg/
├── models/        # Layer 1 inference singletons
├── pipelines/     # Layer 2 composites + Layer 3 factory
├── train/
│   ├── care.py            # CAREDenoiser.Trainer
│   ├── unet.py            # UNetSegmenter.Trainer
│   ├── stardist.py        # StarDistSegmenter.Trainer (2D & 3D)
│   └── smartseeds.py      # joint U-Net + StarDist training harness
├── data/
│   ├── patches.py         # SmartPatches: foreground/background veto
│   └── tiles.py           # tiled loaders
└── eval/
    ├── matching.py        # IoU / F1 / precision / recall
    └── threshold.py       # threshold optimization
```

`SmartPatches` keeps the patch-vetoing logic (configurable `lower_ratio_fore_to_back` / `upper_ratio_fore_to_back`, per-image cap), but exposes it as a clean iterable of patches rather than a side-effecting script.

---

## Package layout

```
src/vollseg/
├── __init__.py            # public API surface
├── models/                # Layer 1
│   ├── care.py
│   ├── unet.py
│   └── stardist.py
├── pipelines/             # Layer 2 + 3
│   ├── base.py            # Pipeline protocol, Result dataclass
│   ├── unet_stardist.py
│   ├── denoised.py
│   ├── roi.py
│   ├── chunked.py
│   └── factory.py         # VollSeg.from_models
├── train/                 # training harnesses
├── data/                  # patch + tile generation
├── eval/                  # metrics + threshold tuning
├── pretrained.py          # Zenodo model registry
└── io.py                  # image readers (inrimage, spatial_image)
```

---

## Design rules (binding)

1. **Composition over inheritance** for combining behaviors — wrap, don't subclass.
2. **One responsibility per class.** A class either trains, or predicts, or composes — never two.
3. **No 2D/3D class duplication.** Dispatch on `ndim` inside the class.
4. **Runtime concerns are decorators.** Chunking, ROI gating, and denoising all wrap a downstream pipeline; none are baked into the singletons.
5. **Fail at construction, not prediction.** Invalid model combinations raise in `from_models`, not mid-inference.
6. **No silent fallbacks.** If a user asks for `seedpool=True` without both models, raise.
7. **Trainers produce models; they are not models.**

---

## Roadmap

- [ ] Layer 1 singletons (`CAREDenoiser`, `UNetSegmenter`, `StarDistSegmenter`)
- [ ] `Pipeline` protocol + `Result` dataclass
- [ ] `UNetStarDistPipeline` with optional seedpool fusion
- [ ] `DenoisedPipeline`, `ROIPipeline`, `Chunked` wrappers
- [ ] `VollSeg.from_models` factory
- [ ] Trainers (`train/`) ported from the legacy `SmartSeeds*` / `UNET` / `StarDist*` classes
- [ ] `SmartPatches` ported as a clean iterable
- [ ] Pretrained Zenodo registry parity
- [ ] Test suite covering each composite path independently
- [ ] `1.0.0` release on PyPI (continuing the original `vollseg` version line)

---

## License

Same license as the original VollSeg (see `LICENSE`).
