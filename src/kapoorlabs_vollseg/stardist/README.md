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
