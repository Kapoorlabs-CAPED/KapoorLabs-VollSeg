# kapoorlabs-vollseg-napari

Napari dock widget for the [KapoorLabs-VollSeg](../../README.md) PyTorch segmentation SDK.

PyTorch-only. The legacy keras stack is intentionally not exposed — this plugin is part of the KapoorLabs in-house tooling, not a general-purpose napari component.

```bash
pip install -e plugins/napari
napari
# Plugins → KapoorLabs VollSeg
```

## Layout

A `QTabWidget` with five tabs:

| Tab         | Controls                                                                   |
| ----------- | -------------------------------------------------------------------------- |
| Input       | Image layer picker, voxel spacing `(dz, dy, dx)`, leading-T axis flag      |
| Models      | Per-role HuggingFace dropdown (CARE, U-Net, Mask-UNet, StarDist, CellPose) + local cache folder + Membrane-mode toggle |
| Inference   | StarDist `n_rays`, `prob_thresh` / `nms_thresh` overrides, `n_tiles` per axis, Seedpool fusion |
| Postproc    | Chunked prediction with per-axis chunk + overlap                            |
| Output      | Layer prefix, optional TIFF dump to disk                                    |

`Run ▶` collects every tab's state into a `RunSpec`, builds the right pipeline via `VollSeg.from_models(...)` / `VollCellSeg.from_models(...)`, runs it inside a `napari.qt.thread_worker`, and writes each non-empty result field (`labels`, `denoised`, `probability`, `semantic`) back as a napari layer.

## Adding a new pretrained model to the dropdowns

The Models tab is populated from `kapoorlabs_vollseg.hub.XENOPUS_MODELS`, classified by HuggingFace repo-id convention (`xenopus-care-…`, `xenopus-unet3d-…`, `xenopus-stardist…`, `xenopus-maskunet-…`, `xenopus-cellpose-…`). Add an entry to that dict and it appears here automatically.
