"""Evaluation: instance-matching metrics, NMS, and threshold tuning."""

from .matching import (
    matching,
    matching_dataset,
    intersection_over_union,
    intersection_over_pred,
    intersection_over_true,
    precision,
    recall,
    accuracy,
    f1,
)
from .nms import NMSLabel
from .threshold import OptimizeThreshold

__all__ = [
    "matching",
    "matching_dataset",
    "intersection_over_union",
    "intersection_over_pred",
    "intersection_over_true",
    "precision",
    "recall",
    "accuracy",
    "f1",
    "NMSLabel",
    "OptimizeThreshold",
]
