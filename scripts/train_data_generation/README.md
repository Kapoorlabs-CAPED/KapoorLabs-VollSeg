# Training-data generation

One Hydra-driven script that emits **a single H5** consumed by both
U-Net and StarDist training. Same patches, same paste-augmentation,
two trainer-specific target keys.

## H5 layout

```
/train/raw    (N, *patch_shape)   float32      # always
/train/label  (N, *patch_shape)   int32        # always — StarDist target source
/train/mask   (N, *patch_shape)   uint8        # only if binary_mask_dir was set
/val/raw, /val/label, /val/mask?              same shape, fewer rows
```

- **StarDist** training reads `raw + label` and derives `(prob, dist)`
  targets on the fly from the (possibly augmented) labels.
- **U-Net** training reads `raw + mask` if `mask` is in the H5;
  otherwise falls back to deriving binary on the fly from `label`.

## Layout

```
train_data_generation/
├── generate-training-data.py     hydra entry point
├── scenario_generate.py          hydra dataclass schema
├── conf/
│   ├── scenario_generate.yaml    defaults composition
│   ├── parameters/default.yaml   patch shape, foreground veto, paste-aug
│   └── train_data_paths/
│       ├── xenopus_default.yaml  edit-locally template
│       ├── xenopus_jeanzay.yaml  jean-zay paths
│       └── xenopus_gwdg.yaml     gwdg/grete paths
├── slurm_generate_jeanzay.sh
├── slurm_generate_gwdg.sh
└── README.md
```

## Patch extraction

Uses [`kapoorlabs_vollseg.data.generate_smart_patches_h5`](../../src/kapoorlabs_vollseg/data/smart_patches_h5.py),
which reproduces the original VollSeg ``SmartPatches`` algorithm:

1. **Foreground patches** — instance-centered, kept only if the
   foreground voxel fraction lies in
   `[lower_ratio_fore_to_back, upper_ratio_fore_to_back]`. Optionally
   erodes each instance before binarizing.
2. **Background-paste augmentation** — for each background voxel, take
   a patch (which must be pure-zero) and additively blend cell patches
   into it (`raw_aug = raw_bg + raw_fg`, `label_aug = label_fg`). Keeps
   the cell silhouette but presents it on a different background
   context. Train-only; val never sees synthetic patches.

## binary_mask_dir fallback chain

The U-Net target is the binary mask. The generator picks the source per
file:

1. If `binary_mask_dir` is set in `train_data_paths` AND a file with the
   matching basename exists there → write that into the H5's `mask`
   dataset (preserves any user erosion / hole-filling).
2. Otherwise → no `mask` dataset is written; the U-Net dataset derives
   binary on the fly from `label` (which is always present).

The integer label image (`label_dir`) is *always* required —
SmartPatches needs cell centroids and bg voxel locations.

## Usage

```bash
# Local
python generate-training-data.py

# Per-cluster path overrides
python generate-training-data.py train_data_paths=xenopus_jeanzay
python generate-training-data.py train_data_paths=xenopus_gwdg

# Override individual knobs
python generate-training-data.py \
    parameters.patch_shape=[8,128,128] \
    parameters.lower_ratio_fore_to_back=0.1 \
    parameters.paste_augmentation=true \
    parameters.max_paste_patches_per_image=500
```

## SLURM

```bash
sbatch slurm_generate_jeanzay.sh
sbatch slurm_generate_gwdg.sh
```

CPU-only (no GPU needed for patch extraction).
