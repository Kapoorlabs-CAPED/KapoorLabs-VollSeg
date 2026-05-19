# StarDist — Lightning (PyTorch) vs Keras (legacy)

A side-by-side walkthrough for training, predicting, and comparing the
two StarDist 3D implementations that ship with `kapoorlabs_vollseg`:

| | Lightning (new) | Keras (legacy) |
| --- | --- | --- |
| Class | `kapoorlabs_vollseg.StarDistSegmenter` | `kapoorlabs_vollseg.StarDistSegmenterKeras` |
| Backend | PyTorch + Lightning + CAREamics UNet trunk | csbdeep + tensorflow + upstream `stardist` |
| Checkpoint | Lightning `.ckpt` + `rays.npy` + `training_config.json` | csbdeep folder (`config.json` + `weights_*.h5`) |
| Rasterizer | Triangulated polyhedron (`ConvexHull` faces, pure numpy) | C/CUDA `polyhedron_to_label` |
| Rays | `Rays_GoldenSpiral` parameterisation (z = linspace, divide-by-anisotropy) — **matches upstream exactly** | `stardist.Rays_GoldenSpiral` |
| Multi-GPU | `predict_timelapse(..., devices=N, strategy="ddp")` — T-axis sharded via Lightning DDP | not supported |
| Hub repo | [`KapoorLabs/`](https://huggingface.co/KapoorLabs) (PyTorch) | [`KapoorLabs-Copenhagen/`](https://huggingface.co/KapoorLabs-Copenhagen) (keras) |

The Lightning implementation is the supported direction; the Keras
classes are kept so already-trained `.h5` weights still work, and so
you can run apples-to-apples comparisons on the same input volumes.

---

## Prereqs

```bash
# Lightning path:
pip install -e .                               # SDK
# torch + careamics + lightning are core deps

# Keras path (legacy):
pip install -e ".[keras]"                      # adds tensorflow + csbdeep + stardist
```

If you already have keras `.h5` weights and predicted TIFFs from a
previous run, you don't need to retrain — skip straight to
[**Compare**](#compare-two-segmentations).

---

## Train

### Lightning

```bash
cd scripts/model_training/

# Single-machine
python lightning-stardist.py

# Cluster (per-cluster path overrides live in train_data_paths/)
python lightning-stardist.py train_data_paths=xenopus_jeanzay
python lightning-stardist.py train_data_paths=xenopus_gwdg

# Tweak any parameter from the command line
python lightning-stardist.py \
    parameters.n_rays=96 \
    parameters.batch_size=4 \
    parameters.augment=true
```

Outputs land under `<log_path>/<experiment_name>/`:
- `last.ckpt` (Lightning checkpoint)
- `rays.npy` (ray geometry sidecar — **required** at predict time)
- `training_config.json` (full Hydra config dump — used by `from_folder`)

The H5 file consumed by training is built by the scripts under
[`scripts/train_data_generation/`](../scripts/train_data_generation/) —
they convert `(raw.tif, label.tif)` pairs into a streaming H5 with
random patches, ready for `StarDistH5Dataset`.

### Keras (legacy)

There is no in-repo trainer for the keras stardist — the previously
published Xenopus weights live on the HuggingFace Hub at
[`KapoorLabs-Copenhagen/xenopus-stardist3d-nuclei-mari`](https://huggingface.co/KapoorLabs-Copenhagen/xenopus-stardist3d-nuclei-mari)
(and `.../xenopus-stardist3d-membrane-mari`). Auto-download via:

```python
from kapoorlabs_vollseg import ensure_model
ensure_model("./models/StarDist3D", "nuclei_xenopus_mari")
# → KapoorLabs-Copenhagen/xenopus-stardist3d-nuclei-mari
```

If you genuinely need to *retrain* under the keras stack, that's an
upstream `stardist` workflow — see [stardist's official training
notebook](https://github.com/stardist/stardist/blob/main/examples/3D/2_training.ipynb).
Most users won't need this.

---

## Tune `(prob_thresh, nms_thresh)` (Lightning only)

`optimize-stardist-thresholds.py` runs the network **once per validation
patch** and reuses the cached `(prob_map, dist_map)` + rasterised
polyhedra across the threshold sweep. Results are written back into
`training_config.json` so the predict script picks them up next run.

```bash
cd scripts/model_training/
python optimize-stardist-thresholds.py train_data_paths=xenopus_jeanzay
```

Typical optimum for Xenopus nuclei after a healthy training run sits
around `prob_thresh=0.6`, `nms_thresh=0.3`. Sanity-check: at
`nms_thresh=0.0` accuracy must collapse to ~0 (every overlap suppresses
everything) — if it doesn't, the rasterizer is buggy.

---

## Predict

### Lightning

```bash
cd scripts/model_prediction/

# Single GPU
python predict-stardist.py experiment_data_paths=predict_jeanzay_stardist

# Multi-GPU timelapse (T-axis sharded via Lightning DDP)
python predict-stardist.py \
    experiment_data_paths=predict_jeanzay_stardist \
    parameters.devices=4 \
    parameters.strategy=ddp
```

Output goes to `<input_dir>/<output_dir>/<file>.tif` (nested under the
raw folder, not somewhere else). Each output TIFF is a `uint32` instance
label image with the same shape as the input — for 4D inputs `(T, Z, Y,
X)` the script gathers all per-rank shards onto rank 0 before writing,
so the file on disk has the full T.

YAML knobs that matter (`scripts/conf/experiment_data_paths/predict_*.yaml`):
- `log_path` — local folder containing the trained model (wins if it exists)
- `hf_repo_id` — fallback HuggingFace repo, e.g. `KapoorLabs/xenopus-stardist-pytorch`
- `input_dir` / `output_dir` / `file_type`
- `devices` / `accelerator` / `strategy` (`ddp` for multi-GPU)

The `from_folder` constructor reads `training_config.json` and the
`rays.npy` sidecar automatically — no manual ray construction needed:

```python
from kapoorlabs_vollseg import StarDistSegmenter

star = StarDistSegmenter.from_folder("models/xenopus_stardist/")
labels = star.predict(volume_zyx).labels
```

### Keras (legacy)

```python
import numpy as np
from tifffile import imread, imwrite
from kapoorlabs_vollseg import StarDistSegmenterKeras, ensure_model

model_dir = ensure_model("./models/StarDist3D", "nuclei_xenopus_mari")

seg = StarDistSegmenterKeras(
    model_dir=model_dir.parent,
    model_name=model_dir.name,
    prob_thresh=0.5,
    nms_thresh=0.3,
)

vol = imread("frame.tif")
result = seg.predict(vol)
imwrite("frame_keras_stardist.tif", result.labels.astype(np.uint16))
```

For timelapse data with the keras backend, loop over T explicitly:

```python
labels_t = np.stack(
    [seg.predict(vol[t]).labels.astype(np.int32) for t in range(vol.shape[0])],
    axis=0,
)
```

(There's no multi-GPU path through the keras stack — that's one of the
reasons for the Lightning rewrite.)

---

## Compare two segmentations

The script `scripts/model_prediction/compare_segmentations.py` reports
per-frame and dataset-aggregate metrics (`precision`, `recall`, `F1`,
`accuracy`, `panoptic_quality`, `mean_matched_score`, `mean_true_score`)
at multiple IoU thresholds. It treats the *first* TIFF as ground truth
and the *second* as prediction.

**Memory note**: the script streams one timepoint per stack at a time
via lazy `TiffFile` access (`_LazyFrames`), so a `(192, 19, 1560, 1560)`
int32 timelapse peaks at ~360 MB resident, not 70 GB.

### Workflow if you already have keras-stardist outputs

You ran the legacy keras pipeline once and saved its TIFF outputs to
`<somewhere>/keras_prediction/*.tif`. You just trained or downloaded
the Lightning model and want to score the new outputs against the old
ones (or against ground-truth keras outputs you trust). The flow:

1. **Predict with Lightning** — produces parallel TIFFs:
   ```bash
   cd scripts/model_prediction/
   python predict-stardist.py experiment_data_paths=predict_jeanzay_stardist
   # → <input_dir>/stardist/<basename>.tif      (PyTorch)
   ```

2. **Point the compare script at both paths**. Edit the two `Path(...)`
   lines at the top of `compare_segmentations.py`:
   ```python
   keras_path = Path(".../keras_prediction/timelapse_fifth_dataset.tif")
   pytorch_path = Path(".../stardist/timelapse_fifth_dataset.tif")
   iou_threshs = (0.3, 0.5, 0.7)
   ```

3. **Run it** (as a script or notebook — it has `# %%` cell markers
   so VS Code / Jupytext treats it as a notebook):
   ```bash
   python scripts/model_prediction/compare_segmentations.py
   ```

The script:
- Reads only the TIFF headers first — fails loud if `shape` or
  `len(T)` mismatch (this is also how the recent DDP-gather-bug got
  caught: one rank's 48-frame output was sitting in place of the full
  192-frame stack).
- Streams one `(yt, yp)` pair at a time through
  `kapoorlabs_vollseg.eval.matching`.
- Reports per-frame metrics, then a streaming dataset aggregate.

### What "good" looks like

For Xenopus nuclei on the public model, the Lightning output should
land within a few percent of the keras baseline at `IoU=0.5`. Two
caveats:

- **Rays must be consistent**: a Lightning model trained against
  `kapoorlabs_vollseg`'s pre-fix rays will predict distances along a
  different golden-spiral covering of the sphere than a keras model
  trained against `Rays_GoldenSpiral`. The rays in this repo now match
  upstream exactly, but if your checkpoint was trained earlier, its
  `rays.npy` sidecar is still loaded at inference for internal
  consistency.
- **Rasterizer**: the Lightning inference now uses a triangulated
  polyhedron (same surface keras's `polyhedron_to_label` reconstructs).
  Before that fix, the old cone-union rasterizer over-painted by ~5%
  and was the dominant source of `accuracy` gap vs keras outputs.

---

## Troubleshooting

| symptom | fix |
| --- | --- |
| Compare reports `shape mismatch: keras=(192,…) pytorch=(48,…)` | DDP gather bug — your predict-stardist run is on an old build. Pull the latest `predict_timelapse` (gathers all ranks onto rank 0 before writing). |
| Stage 1.5 of the optimizer eats 10s of GB | Lower `parameters.min_prob` is the wrong knob — *raise* it (e.g. `0.05` or `0.1`) so fewer peaks get rasterised in the precompute pass. |
| `RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same` during `predict_maps` | The fix is in `StarDistSegmenter.predict_maps` — it now does `model.to(device); model.eval()` before stitching. If you see this error, your install is stale. |
| `mkdir -p failed for path /gpfsscratch/...` (matplotlib) | Environmental, not a bug. `export MPLCONFIGDIR=$JOBSCRATCH/.mpl` in your sbatch script. |
| HF repo gets downloaded even though `log_path` points at a valid folder | The disk-wins-over-HF priority requires the directory to *exist on disk*. Check `ls $log_path`. |

---

## File map

```
src/kapoorlabs_vollseg/
├── stardist/                            ← PyTorch StarDist (the whole rewrite)
│   ├── rays.py                          rays_3d_golden_spiral, compute_faces (ConvexHull)
│   ├── distance.py                      foreground_probability_map, compute_distance_map
│   ├── model.py                         StarDistUNet (CAREamics trunk + prob/dist heads)
│   ├── losses.py                        stardist_loss (BCE + masked L1)
│   ├── dataset.py                       StarDistH5Dataset
│   ├── lightning_module.py              StarDistModule
│   └── inference.py                     predict_volume, nms_to_labels,
│                                        precompute_peaks_and_masks (sweep cache),
│                                        _inside_polyhedron (triangulated rasterizer)
├── _backbones/stardist.py               StarDistBackbone (module + rays + faces)
├── models/stardist.py                   StarDistSegmenter (PyTorch singleton)
├── models/stardist_keras.py             StarDistSegmenterKeras (legacy)
└── pipelines/timelapse_predict.py       predict_timelapse, TimelapsePredictor (DDP)

scripts/
├── model_training/
│   ├── lightning-stardist.py            train PyTorch StarDist (Hydra)
│   └── optimize-stardist-thresholds.py  cached (prob_thresh, nms_thresh) sweep
├── model_prediction/
│   ├── predict-stardist.py              tiled prediction + multi-GPU timelapse
│   └── compare_segmentations.py         streaming keras-vs-pytorch metrics
└── legacy_segmentation_workflow/
    └── 01_nuclei_segmentation.py        keras-era VollSeg pipeline (composed)
```
