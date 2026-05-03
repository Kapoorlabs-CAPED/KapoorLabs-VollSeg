# CARE (PyTorch) — denoising

The first-class CARE denoiser in vollseg. Trained as a supervised
regression problem (low SNR → high SNR) using the
[CAREamics](https://github.com/CAREamics/careamics) UNet under PyTorch
Lightning. The same architecture also underlies the U-Net segmenter and
the StarDist model — the only difference is the loss and the head.

The legacy keras (csbdeep) variant lives next to it as
`CAREDenoiserKeras` for already-trained `.h5` checkpoints. New work
should use the bare-named PyTorch classes documented here.

---

## Architecture

```
input volume (Z, Y, X)
        │
        ▼
   careamics.models.unet.UNet
   (conv_dims=3, num_classes=1)
        │
        ▼
   denoised volume (Z, Y, X)
```

Wrapped inside `vollseg._lightning.CareModule` (a `LightningModule`),
which provides `training_step`, `validation_step`, and a tiled
`predict_step`. Saved checkpoints are standard Lightning `.ckpt` files.

---

## Files

| file | role |
|---|---|
| `src/vollseg/_backbones/care.py` | `CAREBackbone` — wraps `CareModule` + the careamics UNet. `from_checkpoint(ckpt, depth=, ...)` rebuilds and loads weights. |
| `src/vollseg/models/care.py` | `CAREDenoiser` — Layer-1 singleton with `predict(image) -> Result`. Tiles via `CarePredictionDataset`, stitches with linear blending. |
| `src/vollseg/train/care.py` | `CARETrainer` — Lightning trainer; takes a `LightningDataModule` or train/val DataLoaders. |
| `src/vollseg/_lightning/care_module.py` | `CareModule` (`BaseModule` subclass). Mirrors kapoorlabs-lightning shape so checkpoints from there load here. |
| `src/vollseg/_lightning/dataset.py` | `CarePredictionDataset` — tiles a volume into overlapping patches. |
| `src/vollseg/_lightning/stitch.py` | `stitch_tiles` — linear-blend overlap reconstruction. |
| `src/vollseg/_lightning/transforms.py` | `PercentileNormalize`, `ToFloat32` — input normalization. |

---

## Public API quick reference

```python
from vollseg import CAREBackbone, CAREDenoiser, CARETrainer
```

### Inference

```python
denoiser = CAREDenoiser.from_checkpoint(
    "models/membrane.ckpt",
    # Architecture must match training-time values:
    depth=3,
    num_channels_init=64,
    use_batch_norm=True,
    # Inference knobs:
    n_tiles=[1, 4, 4],
    tile_overlap=0.125,
    pmin=0.1, pmax=99.9,
)

result = denoiser.predict(volume)        # volume: (Z, Y, X) numpy array
denoised = result.denoised               # (Z, Y, X) float32
```

`CAREDenoiser` implements the `Pipeline` protocol, so it composes inside
the Layer-2 wrappers and the `VollSeg.from_models` factory:

```python
from vollseg import VollSeg

pipe = VollSeg.from_models(
    care=denoiser,
    stardist=star,
    seedpool=False,
)
result = pipe.predict(volume)            # CARE → StarDist
```

### Training

```python
import torch.nn as nn
from torch.utils.data import DataLoader
from vollseg import CARETrainer

trainer = CARETrainer(
    model_name="membrane_v1",
    model_dir="out/models",
    epochs=100,
    batch_size=16,
    learning_rate=4e-4,
    unet_depth=3,
    num_channels_init=64,
)

# Bring your own DataLoader yielding (low_batch, high_batch) pairs.
trainer.fit(train_dataloader=train_loader, val_dataloader=val_loader)
```

The trainer writes a sidecar `{model_name}.json` with the architecture
knobs alongside the Lightning checkpoint, so the loader knows what shape
of UNet to rebuild.

---

## Comparison with the keras backend

| | PyTorch (`CAREDenoiser`) | Keras (`CAREDenoiserKeras`) |
|---|---|---|
| Backbone | careamics UNet | csbdeep CARE |
| Training | PyTorch Lightning | csbdeep `model.train()` |
| Checkpoint | `.ckpt` (Lightning) | `.h5` + `config.json` (csbdeep) |
| Loaded via | `CAREDenoiser.from_checkpoint(ckpt, depth=, ...)` | `CAREDenoiserKeras(CAREBackboneKeras(config=None, name=, basedir=))` |
| Pretrained weights | (none yet — trained models registered in `vollseg.hub.XENOPUS_MODELS`) | Zenodo registry in `vollseg.pretrained` |

Both implement the same `Pipeline.predict(image) -> Result(denoised=)`
contract, so they're interchangeable in any composite.
