"""Score segmentation predictions against ground-truth label volumes.

Replaces ``CopenhagenWorkflow/01_segmentation_metrics.py``, which
delegated to the external ``caped_ai_metrics.SegmentationScore`` package.
This version uses :func:`kapoorlabs_vollseg.eval.matching_dataset` directly so the
metrics live in the same repo as the segmenter.

Pairs prediction and ground-truth files by sorted basename. Reports
F1 / precision / recall / panoptic-quality at three IoU thresholds.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from tifffile import imread

from kapoorlabs_vollseg.data.io import iter_image_files
from kapoorlabs_vollseg.eval import matching_dataset

from scenarios import SegmentScenario


ConfigStore.instance().store(name="SegmentScenario", node=SegmentScenario)


@hydra.main(version_base="1.3", config_path="conf", config_name="scenario_segment")
def main(config: SegmentScenario) -> None:
    gt_dir = config.experiment_data_paths.metrics_ground_truth_directory
    if gt_dir is None:
        raise SystemExit(
            "metrics_ground_truth_directory is null in experiment_data_paths "
            "— set it before running metrics."
        )

    base = Path(config.experiment_data_paths.base_directory)
    pred_dir = base / config.experiment_data_paths.timelapse_seg_nuclei_directory
    gt_path = Path(gt_dir)

    pred_files = sorted(iter_image_files(pred_dir))
    gt_files = [gt_path / p.name for p in pred_files if (gt_path / p.name).exists()]
    pred_files = [p for p in pred_files if (gt_path / p.name).exists()]

    if not pred_files:
        raise SystemExit(f"No paired pred/gt files between {pred_dir} and {gt_path}.")

    print(f"Scoring {len(pred_files)} pred/gt pairs from {pred_dir}")
    y_true = [imread(p) for p in gt_files]
    y_pred = [imread(p) for p in pred_files]

    stats = matching_dataset(y_true, y_pred, thresh=(0.3, 0.5, 0.7))
    for s in stats:
        print(
            f"IoU>={s.thresh}: f1={s.f1:.3f}  precision={s.precision:.3f}  "
            f"recall={s.recall:.3f}  PQ={s.panoptic_quality:.3f}  "
            f"(tp={s.tp}, fp={s.fp}, fn={s.fn})"
        )


if __name__ == "__main__":
    main()
