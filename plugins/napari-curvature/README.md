# kapoorlabs-curvature-napari

Napari dock widget for measuring **curvature timelapses along user-drawn line profiles** — pairs a raw image with the per-voxel curvature map (produced by [`kapoorlabs_vollseg.curvature`](../../README.md#curvature--force-profiles)) and exports per-line kymographs of κ + intensity vs. time.

```bash
pip install -e plugins/napari-curvature
napari                                                        # then Plugins → KapoorLabs Curvature
napari -w kapoorlabs-curvature-napari                         # pin the dock at startup
```

## Layout

Same shell as `kapoorlabs-vollseg-napari` (QTabWidget, magicgui paths, logo header):

| Tab     | What it does                                                                                  |
| ------- | --------------------------------------------------------------------------------------------- |
| Input   | magicgui `FileEdit`s for the raw TIFF / folder and the curvature TIFF / folder → Load button → both become napari Image layers. |
| Lines   | Pick a Shapes layer (or create one), draw polylines with napari's tools, table of `(id, vertices, length)`. |
| Plot    | Embedded matplotlib canvas — curvature on the primary axis + intensity on a twin axis, along the selected line, updating live when napari's T slider moves. |
| Record  | Output folder, samples-per-line, line width → `Record ▶` walks the T axis and writes one TIFF per (channel × line) kymograph plus a per-line CSV. |

## Output layout

`out_dir/<stem>_<channel>_line<id>.tif` is a `float32` `(T, L)` kymograph — each row is one timepoint, columns are evenly-spaced samples along the polyline. `out_dir/<stem>_line<id>.csv` has columns `frame, sample_idx, curvature, intensity` for ingestion into pandas / Excel.
