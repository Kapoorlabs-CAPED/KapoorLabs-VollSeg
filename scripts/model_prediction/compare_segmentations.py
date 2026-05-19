# %%
from pathlib import Path

import numpy as np
from tifffile import imread

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
keras = imread(keras_path).astype(np.int32)
pytorch = imread(pytorch_path).astype(np.int32)
print(f"keras   : {keras.shape}  dtype={keras.dtype}  max={keras.max()}")
print(f"pytorch : {pytorch.shape}  dtype={pytorch.dtype}  max={pytorch.max()}")
assert (
    keras.shape == pytorch.shape
), f"shape mismatch: keras={keras.shape}  pytorch={pytorch.shape}"


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


if keras.ndim == 4:
    # Per-frame metrics (treating keras as ground truth, pytorch as prediction).
    print(f"\nTimelapse of {keras.shape[0]} frame(s) — per-frame metrics:")
    for t in range(keras.shape[0]):
        print(f"\nframe {t}:")
        for thr in iou_threshs:
            _print_row(thr, matching(keras[t], pytorch[t], thresh=thr))

    # Dataset-level aggregate (averages each metric across frames).
    print("\nDataset aggregate (mean across frames):")
    agg = matching_dataset(
        list(keras),
        list(pytorch),
        thresh=iou_threshs,
        show_progress=False,
    )
    for thr, m in zip(iou_threshs, agg):
        _print_row(thr, m)
elif keras.ndim == 3:
    print("\nVolume metrics:")
    for thr in iou_threshs:
        _print_row(thr, matching(keras, pytorch, thresh=thr))
else:
    raise ValueError(
        f"Expected 3D (ZYX) or 4D (TZYX) label volumes, got ndim={keras.ndim}"
    )


# %%
