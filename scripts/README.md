# Segmentation scripts

Hydra-driven CLI scripts that mirror the segmentation pipeline used in
[CopenhagenWorkflow](https://github.com/Kapoorlabs-CAPED/CopenhagenWorkflow),
re-implemented against the new `vollseg` API (Layer 1 singletons + Layer 2/3
pipelines).

## Layout

```
scripts/
├── scenarios.py                          # @dataclass schemas for Hydra
├── conf/
│   ├── scenario_segment.yaml             # default composition
│   ├── parameters/default.yaml           # runtime knobs (toggles, tiles, sizes)
│   ├── model_paths/jeanzay.yaml          # model directories on the JeanZay HPC
│   └── experiment_data_paths/testdataset.yaml
├── 01_enhance_membrane.py                # CARE denoise → membrane_enhanced/
├── 01_nuclei_segmentation.py             # VollSeg.from_models → seg_nuclei/
├── 01_membrane_segmentation_cellpose.py  # VollCellSeg.from_models(cellpose=...) only
├── 01_vollcellpose_membrane_segmentation.py  # cellpose + nuclei-seeded watershed
└── 01_segmentation_metrics.py            # vollseg.eval.matching_dataset
```

## Running

```bash
# from repo root
pip install -e ".[scripts]"      # installs hydra-core, natsort

python scripts/01_enhance_membrane.py
python scripts/01_nuclei_segmentation.py

# Override any field on the CLI:
python scripts/01_nuclei_segmentation.py \
    parameters.use_seedpool=true \
    parameters.use_care_denoise=true \
    parameters.n_tiles=[1,2,2]

# Swap config groups (e.g. point at a local model dir):
python scripts/01_nuclei_segmentation.py model_paths=local
# → loads conf/model_paths/local.yaml instead of jeanzay.yaml
```

## Pipeline composition

`01_nuclei_segmentation.py` builds the segmenter from these toggles in
`parameters/default.yaml`:

| toggle               | effect                                        |
| -------------------- | --------------------------------------------- |
| `use_roi_unet`       | wraps in `ROIPipeline(roi_unet, ...)`         |
| `use_seedpool`       | wraps in `UNetStarDistPipeline(seedpool=True)` |
| `use_care_denoise`   | wraps in `DenoisedPipeline(care, ...)`        |

The factory composes them in the order: `chunked(roi(denoised(unet+stardist)))`
— matching the diagram in the top-level README.

## Pretrained models — auto-download from HuggingFace

The Xenopus model zoo lives as public model repos under
`huggingface.co/KapoorLabs-Copenhagen/`. The scripts call
`vollseg.ensure_model(model_dir, model_name)` for every configured model
before constructing backbones — if the directory `<model_dir>/<model_name>/`
doesn't exist locally, it's downloaded automatically.

The mapping `model_name → HF repo id` lives in
[`src/vollseg/hub.py`](../src/vollseg/hub.py):

| `model_name` (from YAML)       | HF repo                                                            |
| ------------------------------ | ------------------------------------------------------------------ |
| `membrane_edge_enhancement`    | `KapoorLabs-Copenhagen/xenopus-care-membrane-edge-enhancement`     |
| `unet_nuclei_xenopus_mari`     | `KapoorLabs-Copenhagen/xenopus-unet3d-nuclei-mari`                 |
| `unet_membrane_xenopus_mari`   | `KapoorLabs-Copenhagen/xenopus-unet3d-membrane-mari`               |
| `unet_roi_nuclei_xenopus`      | `KapoorLabs-Copenhagen/xenopus-maskunet-roi-nuclei`                |
| `nuclei_xenopus_mari`          | `KapoorLabs-Copenhagen/xenopus-stardist3d-nuclei-mari`             |
| `membrane_xenopus_mari`        | `KapoorLabs-Copenhagen/xenopus-stardist3d-membrane-mari`           |
| `mem_mneongreen`               | `KapoorLabs-Copenhagen/xenopus-cellpose-mem-mneongreen`            |

### One-time upload (you run this once)

The model weights live today as private folders inside the dataset
`KapoorLabs-Copenhagen/Xenopus_Models`. To migrate them into the public
model repos referenced above:

```bash
# 1. Pull the dataset locally (or point --source at where it already is)
huggingface-cli login    # one-time
git lfs clone https://huggingface.co/datasets/KapoorLabs-Copenhagen/Xenopus_Models

# 2. Dry-run to confirm what will be uploaded
python scripts/_upload_models_to_hf.py \
    --source Xenopus_Models/Mari_Models \
    --dry-run

# 3. Actually create + upload each model repo
python scripts/_upload_models_to_hf.py \
    --source Xenopus_Models/Mari_Models
```

The helper creates each repo as **public**, writes a minimal model card
if the source folder has none, and uploads the contents. Source layout
expected (only these are migrated; cellpose3D and the other CellPose
checkpoints are deliberately ignored):

```
Mari_Models/
├── CARE/membrane_edge_enhancement/
├── Unet3D/unet_nuclei_xenopus_mari/
├── Unet3D/unet_membrane_xenopus_mari/
├── MASKUNET/unet_roi_nuclei_xenopus/
├── StarDist3D/nuclei_xenopus_mari/
├── StarDist3D/membrane_xenopus_mari/
└── CellPose/mem_mneongreen
```

To add a new model later: upload it under `KapoorLabs-Copenhagen/...`,
add an entry to `vollseg.hub.XENOPUS_MODELS`, and an entry to
`scripts/_upload_models_to_hf.py:SOURCE_LAYOUT` if you want the helper
to handle it.

## CellPose / membrane workflow

The CellPose hierarchy is a sibling of `VollSeg`, with its own factory:

```python
from vollseg import VollSeg, VollCellSeg, ...

nuclei_pipe = VollSeg.from_models(stardist=star, roi_unet=roi)

# Plain CellPose (no nuclei seeding):
pipe = VollCellSeg.from_models(cellpose=cellpose)

# Nuclei-seeded watershed (the "VollCellPose" of the original repo):
pipe = VollCellSeg.from_models(
    nuclei_pipeline=nuclei_pipe,
    cellpose=cellpose,
    nuclei_channel=1,
    membrane_channel=0,
)
```

Two scripts demo each path:

| script                                            | shape                                               |
| ------------------------------------------------- | --------------------------------------------------- |
| `01_membrane_segmentation_cellpose.py`            | `VollCellSeg.from_models(cellpose=...)`             |
| `01_vollcellpose_membrane_segmentation.py`        | `cellpose_watershed_fuse(membrane, nuclei, mask)` from cached inputs |

The latter uses the watershed-fuse kernel directly because nuclei labels
are usually already cached on disk by the time you run cell segmentation
— but the file's footer shows the equivalent fully-composed
`VollCellSeg.from_models(nuclei_pipeline=..., cellpose=...)` call for
when you want a one-shot run.
