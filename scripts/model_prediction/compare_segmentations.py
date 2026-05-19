# %%
"""Compare two label-image stacks (keras vs pytorch StarDist outputs).

The TIFFs aren't loaded into memory up-front — instead, each timepoint
is pulled in only when needed, and the previous one is dropped before
the next iteration. Peak memory is therefore ~one frame per stack
(~360 MB for a typical (Z=19, Y=1560, X=1560) int32 frame), not the
whole timelapse (~70 GB for a 192-frame stack).

The headers are read first to verify the two stacks have the same
shape; mismatch fails loud before any heavy I/O happens.

Run as a notebook (cell-by-cell in VS Code / Jupytext) or as a plain
script — there's no argparse on purpose.
"""

from pathlib import Path

import numpy as np
from tifffile import TiffFile

from kapoorlabs_vollseg.eval import matching, matching_dataset


# %%
keras_path = Path(
    "/lustre/fsn1/projects/rech/jsy/uzj81mi/demo_data/keras_prediction/timelapse_fifth_dataset.tif"
)
pytorch_path = Path(
    "/lustre/fsn1/projects/rech/jsy/uzj81mi/demo_data/stardist/timelapse_fifth_dataset.tif"
)
iou_threshs = (0.3, 0.5, 0.7)


# %%
class _LazyFrames:
    """Sized indexable view over a TZYX-layout TIFF.

    ``__getitem__(t)`` pulls only the t-th sub-volume into memory via
    ``tifffile.TiffPage.asarray``. ``len`` and ``shape`` are read from
    the TIFF header so we can validate the two stacks' geometry
    without touching pixel data.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._tf = TiffFile(self.path)
        series = self._tf.series[0]
        self.shape = tuple(series.shape)
        self.dtype = series.dtype
        if len(self.shape) < 3:
            raise ValueError(
                f"{self.path.name}: expected ≥3D (ZYX or TZYX), got shape {self.shape}"
            )
        self._series = series

    def __len__(self) -> int:
        # 4D → T frames; 3D → treat the whole volume as one "frame".
        return int(self.shape[0]) if len(self.shape) == 4 else 1

    def __getitem__(self, t: int) -> np.ndarray:
        if len(self.shape) == 4:
            return self._series.asarray(key=t).astype(np.int32, copy=False)
        if t != 0:
            raise IndexError(t)
        return self._series.asarray().astype(np.int32, copy=False)

    def close(self):
        self._tf.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# %%
def _print_row(thr, m):
    print(
        f"  IoU≥{thr:.2f}  "
        f"P={m.precision:.3f}  R={m.recall:.3f}  F1={m.f1:.3f}  "
        f"acc={m.accuracy:.3f}  "
        f"PQ={m.panoptic_quality:.3f}  "
        f"meanIoU_matched={m.mean_matched_score:.3f}  meanIoU_true={m.mean_true_score:.3f}  "
        f"TP={m.tp}  FP={m.fp}  FN={m.fn}   "
        f"(n_true={m.n_true}, n_pred={m.n_pred})"
    )


# %%
with _LazyFrames(keras_path) as keras, _LazyFrames(pytorch_path) as pytorch:
    # ────────── header-only sanity checks (no pixel data read yet).
    print(f"keras   : shape={keras.shape}  dtype={keras.dtype}  T={len(keras)}")
    print(f"pytorch : shape={pytorch.shape}  dtype={pytorch.dtype}  T={len(pytorch)}")
    if keras.shape != pytorch.shape:
        raise ValueError(
            f"shape mismatch: keras={keras.shape}  pytorch={pytorch.shape}"
        )
    if len(keras) != len(pytorch):
        raise ValueError(
            f"T-axis mismatch: keras has {len(keras)} frames, "
            f"pytorch has {len(pytorch)} — segmentation outputs were "
            f"truncated or padded. Re-run prediction with the same T."
        )

    n_t = len(keras)
    is_timelapse = len(keras.shape) == 4

    if is_timelapse:
        # ────────── Per-frame metrics: only the current T sits in RAM.
        print(f"\nTimelapse of {n_t} frame(s) — per-frame metrics (streaming):")
        for t in range(n_t):
            yt = keras[t]
            yp = pytorch[t]
            print(
                f"\nframe {t}:  (y_true max={int(yt.max())}, y_pred max={int(yp.max())})"
            )
            for thr in iou_threshs:
                _print_row(thr, matching(yt, yp, thresh=thr))
            # yt / yp drop out of scope here; the next iteration overwrites.

        # ────────── Dataset aggregate — matching_dataset iterates over
        # the lazy view, so still only one (yt, yp) pair lives at a time.
        print("\nDataset aggregate (streaming, accumulated across frames):")
        agg = matching_dataset(
            keras,
            pytorch,
            thresh=iou_threshs,
            show_progress=True,
        )
        for thr, m in zip(iou_threshs, agg):
            _print_row(thr, m)
    else:
        print("\nVolume metrics:")
        yt = keras[0]
        yp = pytorch[0]
        for thr in iou_threshs:
            _print_row(thr, matching(yt, yp, thresh=thr))


# %%
