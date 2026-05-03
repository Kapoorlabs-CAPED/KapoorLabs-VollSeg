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
