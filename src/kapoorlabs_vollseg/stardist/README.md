# StarDist re-implementation — developer notes

PyTorch port of the StarDist 3D / 2D segmentation algorithm. Algorithmic
parity with upstream stardist (Schmidt & Weigert) at the level of the
`predict_instances` → label-image flow; no runtime dependency on the
`stardist` Python package. C++ source of the upstream kernels is
vendored under `_lib/` for reference and optional native compilation.

## Layout

```
stardist/
├── README.md              this file
├── __init__.py            re-exports rays + inference entry points
├── rays.py                rays_2d, rays_3d_golden_spiral, compute_faces
├── model.py               StarDistUNet (nn.Module: prob_head + dist_head)
├── lightning_module.py    StarDistModule (Lightning wrapper, predict_step)
├── losses.py              stardist_loss (prob BCE + dist L1, λ-mixed)
├── inference.py           predict_volume + helpers (peaks, NMS, paint)
├── _tiling.py             csbdeep Tile / Tiling / tile_iterator (verbatim port)
└── _lib/                  vendored upstream C++ sources (build-optional)
```

## Pipeline (`predict_volume` → label image)

1. **Normalise** input volume — whole-volume percentile (`pmin..pmax`).
2. **Pad** each spatial axis to a multiple of `block_size = 2**depth`.
3. **Tile** via `tile_iterator` (csbdeep port). One forward per tile;
   each tile's `crop_slice` is hard-assigned to its `write_slice` in
   the destination — no blending, boundary tiles own the volume edge.
4. **Crop** padded `(prob, dist)` buffers back to the input shape.
5. **Peaks** — `skimage.feature.peak_local_max(prob, threshold_abs=
   prob_thresh, min_distance, exclude_border=False)`.
6. **NMS** — `_bbox_nms_kdtree`: greedy bbox-IoU with cKDTree spatial
   pre-filter, processed in score-descending order.
7. **Paint** — `_polyhedra_to_label`: per-cell rasterise via
   tetrahedron decomposition, paint with the "earlier wins" rule.

## Rays + faces (`rays.py`)

- `rays_3d_golden_spiral(n_rays, anisotropy=(z,y,x))` — direct port of
  `stardist.Rays_GoldenSpiral`. Axis order `(z, y, x)`. Anisotropy is
  applied as `rays = rays / anisotropy` then re-normalised (upstream
  divide convention). Anisotropy must match keras at train time AND
  inference time — mismatch produces systematically smaller / larger
  polyhedra in the affected axis.
- `compute_faces(rays)` — `scipy.spatial.ConvexHull(rays).simplices`.
  Same triangulation upstream's `Rays_GoldenSpiral.faces` carries.
- Persistence: rays are **not** cached to `rays.npy`. They are
  regenerated deterministically from `(conv_dims, n_rays, anisotropy)`
  in `training_config.json` at every load. See
  `_backbones/_config.py::read_rays_params`.

## Tile iterator (`_tiling.py`)

Verbatim port of `csbdeep.internals.predict.{Tile, Tiling, tile_iterator,
total_n_tiles}`. The `Tile` slot logic matches upstream byte-for-byte:

- `read_slice` — what to feed the network (with overlap).
- `write_slice` — non-overlapping target region in the destination.
- `crop_slice = write - read` — sub-region of the tile's prediction.

`Tiling.for_n_tiles(n_blocks, n_tiles, n_block_overlap)` picks the
smallest `tile_size` for which `len(Tiling) ≤ n_tiles`. `guarantee='size'`
only (matches stardist's `predict_instances`); `'n_tiles'` not ported.

Boundary tiles get `write_slice` extended to the volume edge — the
fix for the weight-0 boundary holes the previous blending stitcher
produced.

## Block size + overlap

- `block_size = 2 ** depth` (depth-3 careamics U-Net → 8). Read off
  `model.network.depth`, defaults to 8.
- `n_block_overlap = 6` blocks (= 48 vox) by default — covers the
  empirical receptive field of a depth-3 U-Net with slack. Exposed as a
  kwarg on `predict_volume` and `StarDistSegmenter.predict`. Larger
  cells (~ > 50 vox radius) → bump up.
- Legacy `tile_overlap` accepts:
  - `None` → uses default block count;
  - `>= 1` → interpreted as voxels;
  - `< 1` → fraction of smallest axis (back-compat).

## Polyhedron rasterizer (`_polyhedra_to_label`)

Exact port of `stardist3d_impl.cpp::_COMMON_polyhedron_to_label`
(render_mode 0 = `"full"`). The C++ inside-test:

```
inside = in_kernel  OR  (in_convex_hull  AND  inside_polyhedron(...))
```

Implementation:

- `_kernel_halfspaces(polyverts, faces)` — port of
  `halfspaces_kernel + build_halfspace`. One plane per face;
  cross-product normal with the C++ sign convention; offset
  `d = -(A·N)`. Returns `(F, 4)` `(Nz, Ny, Nx, d)`.
- `_convex_hull_halfspaces(polyverts)` —
  `scipy.spatial.ConvexHull(polyverts).equations`. Qhull under the
  hood, same library upstream links.
- `_inside_polyhedron(coords, rays, dists, faces)` — fallback tetrahedron
  test. Per face, builds `M = [d_a·v_a, d_b·v_b, d_c·v_c]` columns,
  inverts once, computes barycentric coords for each voxel via
  `einsum('fij,mj->fmi', M⁻¹, coords)`; voxel is inside iff all four
  barycentric coords (three + `1 - sum`) are non-negative.

Short-circuit: kernel and hull tests are single batched matmuls
(voxels × planes). The expensive tetrahedron test runs only on the
"shell" (in hull, not in kernel) — typically 10–30% of bbox voxels
on cell-shaped polyhedra.

Painting rule (verbatim port of the C++ kernel):

```
region[mask & (region == 0)] = label_value
```

i.e. earlier polyhedra claim a voxel; later ones leave it alone.
`kept_idx` is already score-descending out of NMS.

## NMS (`_bbox_nms_kdtree`)

Greedy axis-aligned bbox-IoU NMS, with a cKDTree spatial pre-filter
that returns candidates within `r_i + r_max` of each centre
(``r = max(dist[i])``). The bbox-IoU test is vectorised against the
returned neighbour set. Score-descending iteration; lower-score
peaks with IoU ≥ `nms_thresh` against any kept peak are suppressed.

Trade-off vs upstream: upstream uses *true* polyhedron IoU via the
C++ kernel; we use *bbox* IoU. The differences are typically a few
percent at `nms_thresh=0.3–0.4`. If the rasterizer-IoU-based path
ever matters more than the speed, see `_rasterize_to_bbox` (already
uses the tetrahedron rasterizer for per-peak masks) — wiring it
into the NMS loop is straightforward but ~10× slower.

## Threshold optimisation flow

`precompute_peaks_and_masks` + `labels_from_precomputed` cache the
expensive forwards + rasterisation once per validation patch. The
threshold sweep (`scripts/model_training/optimize-stardist-thresholds.py`)
then iterates `(prob_thresh, nms_thresh)` over the cached per-peak
bboxes + masks — paint + match only. ~100× faster than rerunning the
network per candidate.

Tuned thresholds land in two files inside the model folder:

- `training_config.json["parameters"]["prob_thresh"/"nms_thresh"]` —
  read by `_backbones/_config.py::read_thresholds`.
- `thresholds.json` — sidecar with `{prob, nms}` for tooling that
  prefers a flat file.

**Consumers must read the tuned thresholds explicitly** —
`StarDistSegmenter.from_folder(...)` only reads architecture knobs,
NOT thresholds. The prediction scripts (`predict-stardist.py`,
`compare-stardist-vs-keras.py`, the sweep scorers) call
`read_thresholds(log_path)` and pass `(prob_thresh, nms_thresh)`
explicitly into `predict_timelapse(...)`. Skipping that step falls
back to the `StarDistSegmenter.__init__` defaults (`0.5` / `0.4`).

## Anisotropy

Anisotropy is a per-axis ray scaling that compensates for non-cubic
voxel spacing. Used as `rays = rays / anisotropy` then re-normalised
(upstream "divide" convention) inside `rays_3d_golden_spiral`. The
**train-time and inference-time values must match exactly** —
mismatch produces systematically smaller / larger polyhedra along
the affected axis.

Empirical default (`scripts/model_training/conf/parameters/stardist_default.yaml`):
`anisotropy = [2.4285714285714284, 1.0, 1.0]` — matches the Xenopus
keras models (Z = 17/7 × XY voxel ratio). Override per dataset.

Persisted in `training_config.json["parameters"]["anisotropy"]` and
read back at load time by `_backbones/_config.py::read_rays_params`
→ rays are regenerated deterministically, never cached to a
`rays.npy` sidecar.

## ROI Mask-UNet gating (production pipeline)

Raw `StarDistSegmenter.predict(frame)` saturates badly on
mostly-empty volumes. Whole-volume `pmin=0.1 / pmax=99.9`
normalisation on a frame that's 95–98% background pushes the few
foreground voxels far outside the training distribution; the
distance head emits inflated values; polyhedra come out too big.

Symptom in our Xenopus benchmark (compare-stardist-vs-keras.py,
early-stage frames, ~370 cells in 1560×1560×19):

- Mean nucleus volume **+38% vs keras** at early-stage frames.
- Mean equivalent radius **+9%**, mean surface area **+21%**.
- Total tissue volume **+50%** at early-stage frames.
- Per-frame variance 3-8× wider than keras.

Fix: wrap StarDist in `kapoorlabs_vollseg.pipelines.ROIPipeline`
with a Mask-UNet ROI model that gates each frame to its
foreground bbox **before** percentile normalisation. The
normalisation then operates on the cropped, mostly-foreground
patch, restoring the training distribution.

```python
from kapoorlabs_vollseg import (
    MaskUNetSegmenter, ROIPipeline, StarDistSegmenter,
)

star = StarDistSegmenter.from_folder(stardist_log_path)
mask_unet = MaskUNetSegmenter.from_folder(maskunet_log_path)
pipeline = ROIPipeline(roi_unet=mask_unet, downstream=star)

labels = pipeline.predict(frame).labels
```

After this change, the same benchmark reports:

| Metric          | Early Δ vs Keras | Mid Δ vs Keras | Late Δ vs Keras |
|-----------------|------------------|----------------|-----------------|
| Mean volume     | +1.5%            | -1.8%          | -4.5%           |
| Mean radius     | ~0%              | -0.3%          | -2.1%           |
| Mean SA         | -1%              | -4%            | +1%             |
| Total volume    | +0.2%            | -4.4%          | -5.0%           |
| Nuclei count    | -3.3%            | -1.9%          | +2.8%           |

Treat `ROIPipeline(roi_unet=mask_unet, downstream=star)` as the
production model. Bare `StarDistSegmenter.predict` is only
appropriate for already-cropped patches.

## Validation against keras

End-to-end correctness is verified by
`scripts/model_prediction/compare-stardist-vs-keras.py` (and the
ROI variant `compare-roi-stardist-vs-keras.py`). Both run the
PyTorch port on the same first/mid/last subset of timepoints the
sweep already scored, compute per-instance regionprops (volume,
equivalent-sphere radius, marching-cubes surface area), and write
a CSV the companion notebooks (`compare_stardist_vs_keras.ipynb` /
`compare_roi_stardist_vs_keras.ipynb`) box-plot by developmental
stage.

This is the canonical regression test — any change to the
algorithmic path (rays, rasterizer, NMS, paint, tile stitcher)
that doesn't keep `mean volume`, `mean radius`, `mean SA`,
`nuclei count` and their totals within a few percent of the keras
reference at every stage is a regression and needs an explanation.

## Lightning module (`lightning_module.py`)

- `forward(x)` → `(prob_logits, dists)` from `StarDistUNet`.
- `predict_step` pads each tile to a multiple of `2**depth`, runs the
  network, crops back, applies `sigmoid` to prob, returns
  `(prob, dists, coords)`. The `predict_volume` loop wraps this.

## Loss (`losses.py`)

`stardist_loss(prob_logits, dists, prob_target, dist_target, lam)`:

- `prob` head: `BCEWithLogitsLoss(prob_logits, prob_target)`.
- `dist` head: foreground-masked `L1(dists, dist_target)` — masked by
  `prob_target > 0`. Background dist regression is intentionally not
  penalised (matches upstream).
- Total = `prob_loss + lam * dist_loss` (`lam = 0.2` per the default
  yaml). The returned tuple is `(total, prob_loss, dist_loss)` for
  per-term logging.

## Algorithmic invariants

- Rays at predict time **must** be byte-identical to rays at train
  time. The deterministic `rays_3d_golden_spiral(n, anisotropy)`
  pipeline + JSON-driven regeneration guarantees this.
- Polyhedron paint order is score-descending; the paint rule must
  preserve earlier-wins semantics for clipping to match upstream.
- Tile iteration is hard-assignment — never blend.
- The model's spatial divisor `2**depth` must be honoured at both
  the tile and whole-volume level.

## Diagnostics

- Set `KAPOORLABS_VOLLSEG_PROGRESS=1` to surface inner-loop tqdm
  bars (tiles / NMS peaks / paint cells). Off by default (SLURM-log
  friendly).
- `predict-stardist.py` writes the raw prob map / labels TIFFs that
  let you inspect prob distribution + per-cell rasterisation
  independently.

## Vendored C++ sources (`_lib/`)

Verbatim copies from upstream stardist (BSD-3, see
`STARDIST_LICENSE.txt`). Build with the upstream Makefile after
vendoring qhull + nanoflann (instructions in `_lib/README.md`). The
Python port in `inference.py` matches the algorithms in those C++
files; diff against them when verifying upstream parity.
