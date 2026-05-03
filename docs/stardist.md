# StarDist (PyTorch) — instance segmentation via star-convex shapes

A clean PyTorch rewrite of [StarDist](https://github.com/stardist/stardist)
on top of the same careamics UNet trunk used by `CAREDenoiser` and
`UNetSegmenter`. Training is via PyTorch Lightning; data prep stores
`(raw, label)` patches in H5 and the per-batch dist / prob targets are
computed on the fly so any geometric augmentation works in any ndim.

The legacy keras `StarDistSegmenterKeras` remains until the PyTorch port
is battle-tested.

---

## Concept (one paragraph)

StarDist predicts, at every pixel, (1) the probability that the pixel
belongs to *some* foreground object and (2) the distance from that
pixel to the object boundary along each of `n_rays` ray directions.
At inference, peaks of the probability map become candidate object
centers; their predicted distances trace out a star-convex polygon
(2D) or polyhedron (3D); a non-max-suppression pass keeps only
non-overlapping survivors and paints them into a label image.

---

## Architecture

```
input volume (Z, Y, X)                       (or 2D — same code path)
        │
        ▼
   careamics.models.unet.UNet  (trunk, num_classes ≡ feature width)
        │
        ├──► 1×1 conv  → 1 ch     prob_logits (sigmoid in inference)
        └──► 1×1 conv  → n_rays   distances along each ray
```

Wrapped in `StarDistModule` (a `BaseModule` subclass) for training and
tiled prediction.

---

## Files

```
src/vollseg/stardist/
├── rays.py                  rays_2d (angles), rays_3d_golden_spiral (anisotropy-aware)
├── distance.py              foreground_probability_map, compute_distance_map (numba-jit if available)
├── model.py                 StarDistUNet — trunk + (prob_head, dist_head)
├── losses.py                prob_loss (BCEWithLogits), dist_loss (masked L1), stardist_loss
├── h5_data.py               generate_stardist_h5(raw, label, …) → H5 of (raw, label) patches
├── dataset.py               StarDistH5Dataset — yields (raw, prob, dist); targets derived per __getitem__
├── transforms.py            Compose, RandomFlip, RandomRot90, InputGaussianNoise, InputPercentileNormalize
├── lightning_module.py      StarDistModule — training_step / validation_step / predict_step
└── inference.py             predict_volume — tile + stitch + peaks + rasterize + NMS + paint

src/vollseg/_backbones/stardist.py     StarDistBackbone (Lightning module + rays array)
src/vollseg/models/stardist.py         StarDistSegmenter (Layer-1 singleton)
src/vollseg/train/stardist.py          StarDistTrainer (Lightning trainer)
```

---

## Public API quick reference

```python
from vollseg import (
    StarDistBackbone,
    StarDistSegmenter,
    StarDistTrainer,
)
from vollseg.stardist import (
    rays_2d, rays_3d_golden_spiral,
    generate_stardist_h5, StarDistH5Dataset, stardist_collate,
    Compose, RandomFlip, RandomRot90,
    InputPercentileNormalize, InputGaussianNoise,
    predict_volume, StarDistResult,
    foreground_probability_map, compute_distance_map,
)
```

---

## Worked example: train + predict

The repo includes [`scripts/02_train_stardist_pytorch.py`](../scripts/02_train_stardist_pytorch.py)
which exercises the whole pipeline via three argparse subcommands.
Annotated walkthrough:

### 1. Data prep — store `(raw, label)` patches

```bash
python scripts/02_train_stardist_pytorch.py prep \
    --raw  data/raw \
    --label data/labels \
    --out  out/stardist_train.h5 \
    --patch  16 256 256 \
    --val-files 1 \
    --min-fg 0.005
```

Walks paired files (matched by basename), slides a window across each
volume, and streams `(raw_patch, label_patch)` into a resizable H5
(lzf-compressed). The last `--val-files` files are reserved for
validation. **Rays are not specified at this stage** — they're a
property of the trained model, so they're decided at training time.

### 2. Train

```bash
python scripts/02_train_stardist_pytorch.py train \
    --h5         out/stardist_train.h5 \
    --model-dir  out/models \
    --model-name xenopus_v1 \
    --n-rays 96 --anisotropy 2.0 1.0 1.0 \
    --epochs 100 --batch-size 4 --lr 4e-4 \
    --augment
```

`StarDistTrainer` constructs the `StarDistUNet`, wraps it in
`StarDistModule`, and hands them to `lightning.Trainer.fit`. With
`--augment`, the dataset transform pipeline is:

```python
Compose([
    InputPercentileNormalize(pmin=0.1, pmax=99.9),
    RandomFlip(p=0.5),                  # per-axis, ndim-agnostic
    RandomRot90(p=0.5),                 # YX-plane only
    InputGaussianNoise(std=0.01, p=0.3),
])
```

The dataset re-derives `(prob, dist)` targets from the augmented label
on every `__getitem__` — so the same flips/rotations work in 2D and 3D
without channel-permutation gymnastics.

Outputs:
- `out/models/xenopus_v1/last.ckpt` (Lightning checkpoint)
- `out/models/xenopus_v1.rays.npy` (sidecar — same array used at predict)
- `out/models/xenopus_v1.json` (architecture knobs)

### 3. Predict

```bash
python scripts/02_train_stardist_pytorch.py predict \
    --ckpt out/models/xenopus_v1/last.ckpt \
    --rays out/models/xenopus_v1.rays.npy \
    --image data/test/sample.tif \
    --out out/predictions/sample_seg.tif \
    --prob-thresh 0.5 --nms-thresh 0.4
```

Equivalent Python:

```python
from vollseg import StarDistSegmenter
import numpy as np
from tifffile import imread

seg = StarDistSegmenter.from_checkpoint(
    "out/models/xenopus_v1/last.ckpt",
    rays=np.load("out/models/xenopus_v1.rays.npy"),
    prob_thresh=0.5,
    nms_thresh=0.4,
)
result = seg.predict(imread("data/test/sample.tif"))
print(result.labels.max(), "objects")
```

`StarDistSegmenter` implements `Pipeline`, so it composes inside
`VollSeg.from_models(stardist=seg, roi_unet=…, care=…, seedpool=…)` like
any other Layer-1 singleton.

---

## Inference algorithm (in `inference.py:predict_volume`)

1. **Tile** the input via `CarePredictionDataset` — overlapping windows
   sized to `n_tiles` per axis.
2. **Predict** `prob` (sigmoid) + `dist` per tile, **stitch** both with
   linear-blend overlap weighting.
3. **Peak detection** — `skimage.feature.peak_local_max` with
   `threshold_abs=prob_thresh`, `min_distance` for spatial debouncing.
4. **Rasterize each star polyhedron** in its own bounding box. Per
   voxel: find the *nearest ray* (largest dot product with the unit
   vector from center to voxel) and accept the voxel iff its distance
   from the center is ≤ that ray's predicted length. Good
   approximation for ≥64 rays; fewer rays will look faceted.
5. **NMS** — sort peaks by descending probability; greedily drop any
   peak whose mask has IoU ≥ `nms_thresh` with an already-kept peak.
   IoU is computed over the intersection of the two bounding boxes —
   no full-volume mask materialized.
6. **Paint** survivors into a `uint16` label image.

---

## Comparison with upstream stardist

| | This (PyTorch) | Upstream (keras / Cython) |
|---|---|---|
| Backbone | careamics UNet (shared with CARE/UNet) | Custom 2D/3D UNet |
| Training | PyTorch Lightning | Keras (`tf.keras`) |
| Targets | Computed on-the-fly from augmented labels in the dataset | Computed on-the-fly from augmented labels |
| Augmentation | `RandomFlip`/`RandomRot90` work in any ndim | Same idea (user's `augmenter` callable) |
| H5 layout | `(raw, label)` per split | (no H5; in-memory) |
| Distance ray-march | NumPy + optional numba | Cython kernel |
| NMS | bounding-box-IoU on rasterized polyhedra (NumPy) | Cython kernel using polyhedron volume |
| Inference | tiled predict + linear-blend stitch + peaks + NMS | tiled predict + peaks + NMS |
| Speed (training) | similar with numba; ~50× slower without | reference |
| Speed (inference) | comparable | reference |

---

## Performance notes

- **Numba is a soft dependency** that turns the per-sample distance ray-
  march into a fast inner loop. Without it, training will work but
  per-batch CPU time on a 16×256×256 patch with 96 rays is on the order
  of 10s. With numba, ~200 ms.
- `--num-workers 2` (or higher) on the training DataLoader lets the
  CPU-side target compute overlap with GPU forward/backward — strongly
  recommended.
- The inference rasterizer uses **nearest-ray** approximation. For
  `n_rays ≥ 64` this is visually indistinguishable from true
  star-polyhedron rendering. For very few rays (≤32), barycentric
  interpolation among the K nearest rays would be better — not
  implemented yet because no real model uses that few rays.
- 3D `RandomRot90` rotates only in the YX plane. Rotation in any plane
  involving Z would be possible but introduces `ZY` aspect-ratio issues
  with anisotropic data; skipped.
