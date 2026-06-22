# Legacy code & pretrained models

Everything keras / csbdeep / stardist `.h5` from the original VollSeg lives
in this repo under two locations, kept around so already-trained weights
keep working without a retrain. New work goes through the PyTorch path
documented in the top-level [`README`](../README.md); this page is the
single index of the legacy surface.

## What "legacy" means here

The original VollSeg grew organically into a single ``utils.py`` with
branching ``if/else`` chains for every combination of
``denoise / ROI / U-Net / StarDist / seedpool / 2D / 3D``. Adding a mode
meant editing the same mega-functions; testing one path required mocking
the rest. The current repo replaces that with the three-layer
composition stack documented in the top-level README — but the keras
backbones, the pretrained Xenopus zoo published with the original paper,
and the original ``01_*.py`` driver scripts are preserved here so users
already running an end-to-end pipeline don't get broken.

## Surface

| Component                       | Where                                                              |
| ------------------------------- | ------------------------------------------------------------------ |
| Legacy keras backbones          | ``src/kapoorlabs_vollseg/models/{care,unet,maskunet,stardist}_keras.py`` |
| Legacy pretrained Xenopus zoo   | HuggingFace org ``KapoorLabs-Copenhagen/`` (see registry below)    |
| Legacy registry                 | ``src/kapoorlabs_vollseg/hub.py::XENOPUS_MODELS``                  |
| Legacy driver scripts           | ``scripts/legacy_segmentation_workflow/01_*.py``                   |
| Legacy HF migration helper      | ``scripts/legacy_segmentation_workflow/_upload_models_to_hf.py``   |
| Legacy keras trainers           | ``src/kapoorlabs_vollseg/train/{care,unet,stardist,cellpose}_keras.py`` |

## Keras singletons

Same ``Pipeline.predict(image) -> Result`` protocol as the PyTorch
singletons. Importable from the top-level package, gated behind the
``[keras]`` extra:

```bash
pip install kapoorlabs-vollseg[keras]
```

```python
from kapoorlabs_vollseg import (
    CAREDenoiserKeras,
    UNetSegmenterKeras,
    MaskUNetSegmenterKeras,
    StarDistSegmenterKeras,
)
```

Each one accepts a csbdeep-style model folder (``config.json`` +
``weights_*.h5``) at construction:

```python
seg = StarDistSegmenterKeras.from_folder("models/nuclei_xenopus_mari")
labels = seg.predict(volume).labels
```

Because they implement the same protocol, every composite in
``pipelines/`` (``ROIPipeline``, ``UNetStarDistPipeline``,
``DenoisedPipeline``, ``Chunked``) and every factory
(``VollSeg.from_models``, ``VollCellSeg.from_models``) accepts them
interchangeably with the PyTorch ones.

## Pretrained Xenopus model zoo (HuggingFace)

Hosted as public model repos under
[`KapoorLabs-Copenhagen/`](https://huggingface.co/KapoorLabs-Copenhagen).
Resolved through ``kapoorlabs_vollseg.hub.ensure_model``:

```python
from kapoorlabs_vollseg import ensure_model

# Downloads on first call, then becomes a local-disk lookup.
folder = ensure_model(
    "./models/StarDist3D",            # parent directory
    "nuclei_xenopus_mari",            # registered model name
)
seg = StarDistSegmenterKeras.from_folder(folder)
```

| ``model_name`` (registry key)  | HuggingFace repo                                                    |
| ------------------------------ | ------------------------------------------------------------------- |
| ``membrane_edge_enhancement``  | ``KapoorLabs-Copenhagen/xenopus-care-membrane-edge-enhancement``    |
| ``unet_nuclei_xenopus_mari``   | ``KapoorLabs-Copenhagen/xenopus-unet3d-nuclei-mari``                |
| ``unet_membrane_xenopus_mari`` | ``KapoorLabs-Copenhagen/xenopus-unet3d-membrane-mari``              |
| ``unet_roi_nuclei_xenopus``    | ``KapoorLabs-Copenhagen/xenopus-maskunet-roi-nuclei``               |
| ``nuclei_xenopus_mari``        | ``KapoorLabs-Copenhagen/xenopus-stardist3d-nuclei-mari``            |
| ``membrane_xenopus_mari``      | ``KapoorLabs-Copenhagen/xenopus-stardist3d-membrane-mari``          |
| ``mem_mneongreen``             | ``KapoorLabs-Copenhagen/xenopus-cellpose-mem-mneongreen``           |

Single source of truth: ``src/kapoorlabs_vollseg/hub.py::XENOPUS_MODELS``.
To add a new legacy model, append a row to that dict, upload the weights
under the same ``KapoorLabs-Copenhagen/...`` repo, and (optionally) add
an entry to the ``SOURCE_LAYOUT`` dict in the migration helper below.

### One-time upload helper

``scripts/legacy_segmentation_workflow/_upload_models_to_hf.py`` migrates
a local folder into the HF repo named in ``XENOPUS_MODELS``. Reads
``HF_TOKEN`` from the environment.

```bash
huggingface-cli login                      # one-time

python scripts/legacy_segmentation_workflow/_upload_models_to_hf.py \
    --source /path/to/Mari_Models \
    --dry-run                              # preview before push

python scripts/legacy_segmentation_workflow/_upload_models_to_hf.py \
    --source /path/to/Mari_Models \
    --only nuclei_xenopus_mari             # restrict to one model
```

Expected source layout (only these are migrated; ``cellpose3D`` and
unrelated CellPose checkpoints are ignored):

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

## Driver scripts (`scripts/legacy_segmentation_workflow/`)

Hydra-driven, one script per step of the original pipeline:

| Script                                       | Pipeline                                              |
| -------------------------------------------- | ----------------------------------------------------- |
| ``01_enhance_membrane.py``                   | CARE denoise → ``membrane_enhanced/``                 |
| ``01_nuclei_segmentation.py``                | ``VollSeg.from_models(stardist=…, roi_unet=…, …)``    |
| ``01_membrane_segmentation_cellpose.py``     | ``VollCellSeg.from_models(cellpose=…)`` only          |
| ``01_vollcellpose_membrane_segmentation.py`` | CellPose + nuclei-seeded watershed                    |
| ``01_segmentation_metrics.py``               | ``kapoorlabs_vollseg.eval.matching_dataset``          |

All five share the conf groups under ``scripts/conf/`` (``parameters/``,
``model_paths/``, ``experiment_data_paths/``). Each script routes a model
slot to the PyTorch first-class singleton when a ``.ckpt`` path is set
in the YAML, otherwise falls back to the keras sibling — so the legacy
scripts can be incrementally migrated checkpoint by checkpoint.

```bash
# Run the original nuclei pipeline against the keras zoo.
python scripts/legacy_segmentation_workflow/01_nuclei_segmentation.py

# Override toggles on the CLI:
python scripts/legacy_segmentation_workflow/01_nuclei_segmentation.py \
    parameters.use_seedpool=true \
    parameters.use_care_denoise=true \
    parameters.n_tiles=[1,2,2]

# Swap config groups (e.g. point at a local model dir):
python scripts/legacy_segmentation_workflow/01_nuclei_segmentation.py model_paths=local
```

### Composition toggles

``01_nuclei_segmentation.py`` reads these from
``conf/parameters/default.yaml``:

| toggle             | effect                                          |
| ------------------ | ----------------------------------------------- |
| ``use_roi_unet``   | wraps in ``ROIPipeline(roi_unet, …)``           |
| ``use_seedpool``   | wraps in ``UNetStarDistPipeline(seedpool=True)``|
| ``use_care_denoise`` | wraps in ``DenoisedPipeline(care, …)``        |

Effective shape: ``chunked(roi(denoised(unet + stardist)))``.

## When NOT to use the legacy path

If you can retrain the model you should — the PyTorch path (top-level
README) has:

- Anisotropy-aware StarDist rays + the kernel + convex-hull short-circuit
  polyhedron rasterizer.
- ROI Mask-UNet gating for the saturation-on-empty-frames problem.
- Tuned ``(prob_thresh, nms_thresh)`` baked into ``training_config.json``.
- The validated production model
  ``KapoorLabs/xenopus-stardist-pytorch`` on HuggingFace (mean
  volume / radius / surface area within ~2 % of the keras reference at
  every developmental stage).

The legacy surface only exists so already-trained ``.h5`` weights stay
usable. There is no plan to add new models or new scripts to it.
