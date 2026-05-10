# U-Net (PyTorch) — semantic segmentation

The first-class U-Net segmenter in vollseg. Same careamics UNet as
`CAREDenoiser`, but trained with `BCEWithLogitsLoss` for binary
foreground/background segmentation. The output probability map is
multi-Otsu thresholded into a binary mask, then connected components
yields per-instance integer labels.

`MaskUNetSegmenter` is a thin subclass that swaps in `MaskUNetBackbone`
— same architecture, separate name so MaskUNet checkpoints can be
referenced by intent.

The legacy `UNetSegmenterKeras` / `MaskUNetSegmenterKeras` remain for
already-trained csbdeep `.h5` weights.

---

## Architecture

```
input volume (Z, Y, X)
        │
        ▼
   careamics.models.unet.UNet
   (conv_dims=3, num_classes=1)
        │
        ▼  raw logits  → torch.sigmoid →  prob map
        ▼
   multi-Otsu threshold  →  binary semantic mask
        │
        ▼
   skimage.measure.label   →  uint16 instance labels
        │
        ▼
   remove_small_objects(min_size=)
```

Wrapped inside the same `CareModule` Lightning class as the denoiser —
the difference is the loss (BCE-with-logits vs MSE) and the
post-processing on the model output.

---

## Files

| file | role |
|---|---|
| `src/kapoorlabs_vollseg/_backbones/unet.py` | `UNetBackbone` — wraps a `CareModule` whose output is interpreted as binary logits. |
| `src/kapoorlabs_vollseg/_backbones/maskunet.py` | `MaskUNetBackbone` — alias of `UNetBackbone` for naming discipline. |
| `src/kapoorlabs_vollseg/models/unet.py` | `UNetSegmenter` — Layer-1 singleton with `predict(image) -> Result(labels=, semantic=, probability=)`. |
| `src/kapoorlabs_vollseg/models/maskunet.py` | `MaskUNetSegmenter` — subclass of `UNetSegmenter`, only `from_checkpoint` differs. |
| `src/kapoorlabs_vollseg/train/unet.py` | `UNetTrainer` — subclass of `CARETrainer` with the loss swapped to BCE. |
| `src/kapoorlabs_vollseg/train/maskunet.py` | `MaskUNetTrainer` — subclass of `UNetTrainer` (currently identical defaults). |
| `src/kapoorlabs_vollseg/_lightning/care_module.py` | `CareModule` — shared with CARE. |

---

## Public API quick reference

```python
from kapoorlabs_vollseg import (
    UNetBackbone, UNetSegmenter, UNetTrainer,
    MaskUNetBackbone, MaskUNetSegmenter, MaskUNetTrainer,
)
```

### Inference

```python
seg = UNetSegmenter.from_checkpoint(
    "models/unet_xenopus.ckpt",
    depth=3,
    num_channels_init=64,
    # Layer-1 knobs:
    min_size=10,
    morph_iterations=0,
    n_tiles=[1, 4, 4],
)

result = seg.predict(volume)
labels = result.labels             # uint16 (Z, Y, X) — connected-components instances
binary = result.semantic           # bool (Z, Y, X) — thresholded mask
prob   = result.probability        # float32 (Z, Y, X) — sigmoid output
```

`UNetSegmenter` (and `MaskUNetSegmenter`) implements `Pipeline`, so it
plays in `VollSeg.from_models`:

```python
from kapoorlabs_vollseg import VollSeg

# StarDist gated by a U-Net ROI mask
pipe = VollSeg.from_models(
    stardist=star,
    roi_unet=MaskUNetSegmenter.from_checkpoint("models/roi.ckpt", depth=3),
)
```

### Training

```python
from kapoorlabs_vollseg import UNetTrainer

trainer = UNetTrainer(
    model_name="xenopus_seg_v1",
    model_dir="out/models",
    epochs=100,
    batch_size=16,
    learning_rate=4e-4,
    unet_depth=3,
    num_channels_init=64,
)

trainer.fit(train_dataloader=train_loader, val_dataloader=val_loader)
```

The trainer's `LightningModule` is the same `CareModule` used by CARE —
it expects `(input_batch, target_batch)` pairs from the loader, where
the target is a binary mask (will be cast to float and compared against
the logits via BCE-with-logits).

---

## Comparison with the keras backend

| | PyTorch (`UNetSegmenter`) | Keras (`UNetSegmenterKeras`) |
|---|---|---|
| Backbone | careamics UNet | csbdeep CARE |
| Loss | `BCEWithLogitsLoss` | csbdeep `mae` (default) |
| Output | logits → sigmoid → multi-Otsu | regression → multi-Otsu |
| Training | PyTorch Lightning | csbdeep `model.train()` |
| Checkpoint | `.ckpt` | `.h5` + `config.json` |
| Pretrained zoo | `kapoorlabs_vollseg.hub.XENOPUS_MODELS` (HF) | `kapoorlabs_vollseg.pretrained` (Zenodo) |

`UNetSegmenter.predict` returns the same `Result(labels=, semantic=, probability=)`
shape as the keras version, so Layer-2 composites and the factory don't
care which one they got.
