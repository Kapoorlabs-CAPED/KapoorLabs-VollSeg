# Training-data generation

Hydra-driven H5 generators, one per task. All scripts emit
`generate_smart_patches_h5` SmartPatches-style instance-centered
patches with whole-volume percentile normalisation applied **before**
patch extraction (CARE-style) — so train, val and inference all see
the same `[0, 1]` per-volume distribution.

## Layout

```
train_data_generation/
├── generate-training-data.py        StarDist + U-Net (shared H5)
├── generate-care-training-data.py   CARE denoiser (low/high pair stack)
├── generate-roi-training-data.py    ROI Mask-UNet
├── merge_h5_datasets.py             concat multiple H5s along /train and /val
├── h5_inspect.py                    quick schema dump for any H5
├── *_h5_visualizer.ipynb            per-task napari-free notebooks
├── scenario_generate*.py            Hydra dataclass schemas
├── conf/                            scenario yamls + parameters/ + train_data_paths/
├── slurm_generate_{jeanzay,gwdg}.sh         StarDist/U-Net SmartPatches
├── slurm_generate_care.sh                   CARE H5
├── slurm_generate_roi.sh                    ROI Mask-UNet H5
└── slurm_merge_h5.sh                        merge job
```

CPU-only — no GPU needed for any of these.

## StarDist + U-Net (`generate-training-data.py`)

H5 layout (consumed by `lightning-stardist.py` AND `lightning-unet.py`):

```
/train/raw    (N, *patch)   float32   percentile-normalised whole-volume → patch
/train/label  (N, *patch)   int32     instance labels — StarDist target source
/train/mask   (N, *patch)   uint8     optional, written when binary_mask_dir is set
/val/raw, /val/label, /val/mask?      same shape
```

- StarDist reads `raw + label`, derives `(prob, dist)` targets on the fly.
- U-Net reads `raw + mask` if present; otherwise derives binary from `label > 0`.

### Patch extraction

`kapoorlabs_vollseg.data.generate_smart_patches_h5`:

1. **Whole-volume normalisation** — `(raw - p_pmin) / (p_pmax - p_pmin)` clipped to `[0, 1]` per source volume, before any patch is cut. Default `pmin=0.1, pmax=99.9`.
2. **Foreground patches** — instance-centered, kept when the foreground voxel fraction is in `[lower_ratio_fore_to_back, upper_ratio_fore_to_back]`. Optional per-instance erosion before binarising.
3. **Background-paste augmentation** — for each background voxel, additively blend cell patches into a pure-zero patch (`raw_aug = raw_bg + raw_fg`, `label_aug = label_fg`). Train-only; val never sees synthetic patches.

### `mask` source fallback

1. `binary_mask_dir/<basename>` exists → that file is written into `/.../mask`.
2. Otherwise → no `mask` dataset; U-Net dataset derives binary from `label > 0` at training time.

The integer label image (`label_dir`) is always required.

## CARE (`generate-care-training-data.py`)

Reads paired low-SNR / high-SNR TIFFs (`low_dir` + `high_dir`, same basenames) and emits one H5 with `/train/{low,high}` and `/val/{low,high}` — stride is `~2/3 × patch` (≈ 33 % overlap) on train and full-patch on val.

```
/train/low, /train/high   (N, patch_z, patch_y, patch_x)   float32   percentile-normalised
/val/low,   /val/high     same                                       non-overlapping stride
```

## ROI Mask-UNet (`generate-roi-training-data.py`)

Same shape as the StarDist/U-Net generator but emits 2D MIP patches consumed by the ROI Mask-UNet trainer.

## Running

```bash
# Default config = xenopus_jeanzay paths.
python generate-training-data.py
python generate-care-training-data.py
python generate-roi-training-data.py

# Override paths or knobs on the CLI:
python generate-training-data.py train_data_paths=xenopus_gwdg
python generate-training-data.py \
    parameters.patch_shape=[8,128,128] \
    parameters.paste_augmentation=true \
    parameters.max_paste_patches_per_image=500 \
    parameters.pmin=0.1 parameters.pmax=99.9
```

## SLURM

```bash
sbatch slurm_generate_jeanzay.sh        # StarDist + U-Net on jean-zay
sbatch slurm_generate_gwdg.sh           # StarDist + U-Net on grete
sbatch slurm_generate_care.sh           # CARE H5
sbatch slurm_generate_roi.sh            # ROI Mask-UNet H5
sbatch slurm_merge_h5.sh                # merge multiple H5s
```

## Inspection

```bash
python h5_inspect.py                     # dump shape / dtype for every dataset
```

Open any of the `*_h5_visualizer.ipynb` notebooks to scroll through patches frame by frame — `unet_h5_visualizer.ipynb` also includes a polarity check (the dataset-wide foreground fraction) so an inverted mask is caught before training.
