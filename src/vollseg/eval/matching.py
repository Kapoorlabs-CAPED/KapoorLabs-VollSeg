"""Instance-segmentation matching metrics — IoU / precision / recall / F1.

Algorithm: for each (y_true, y_pred) pair we build an overlap matrix and
solve the Hungarian assignment that maximizes total IoU above ``thresh``.
This is the same logic as ``stardist.matching`` (and the original VollSeg
``matching.py``), but with the numba dependency dropped — pure NumPy.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from csbdeep.utils import _raise
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import relabel_sequential
from tqdm import tqdm


# --- low-level helpers ----------------------------------------------------

def _check_label_array(y, name: str = "labels", check_sequential: bool = False):
    err = ValueError(
        f"{name} must be an array of "
        f"{'sequential ' if check_sequential else ''}non-negative integers."
    )
    if not (isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.integer)):
        raise err
    if y.size == 0:
        return True
    if check_sequential:
        labels = np.unique(y)
        if (set(labels.tolist()) - {0}) != set(range(1, 1 + int(labels.max()))):
            raise err
    elif y.min() < 0:
        raise err
    return True


def _safe_divide(x, y, eps: float = 1e-10):
    if np.isscalar(x) and np.isscalar(y):
        return x / y if abs(y) > eps else 0.0
    out = np.zeros(np.broadcast(x, y).shape, np.float32)
    np.divide(x, y, out=out, where=np.abs(y) > eps)
    return out


def _label_overlap(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + int(x.max()), 1 + int(y.max())), dtype=np.uint64)
    np.add.at(overlap, (x, y), 1)
    return overlap


def label_overlap(x, y, check: bool = True):
    if check:
        _check_label_array(x, "x", True)
        _check_label_array(y, "y", True)
        x.shape == y.shape or _raise(ValueError("x and y must have the same shape"))
    return _label_overlap(x, y)


# --- matching criteria ----------------------------------------------------

def intersection_over_union(overlap: np.ndarray) -> np.ndarray:
    _check_label_array(overlap, "overlap")
    if np.sum(overlap) == 0:
        return overlap
    n_pred = np.sum(overlap, axis=0, keepdims=True)
    n_true = np.sum(overlap, axis=1, keepdims=True)
    return _safe_divide(overlap, n_pred + n_true - overlap)


def intersection_over_true(overlap: np.ndarray) -> np.ndarray:
    _check_label_array(overlap, "overlap")
    if np.sum(overlap) == 0:
        return overlap
    return _safe_divide(overlap, np.sum(overlap, axis=1, keepdims=True))


def intersection_over_pred(overlap: np.ndarray) -> np.ndarray:
    _check_label_array(overlap, "overlap")
    if np.sum(overlap) == 0:
        return overlap
    return _safe_divide(overlap, np.sum(overlap, axis=0, keepdims=True))


_MATCHING_CRITERIA = {
    "iou": intersection_over_union,
    "iot": intersection_over_true,
    "iop": intersection_over_pred,
}


# --- aggregate metrics ----------------------------------------------------

def precision(tp, fp, fn): return tp / (tp + fp) if tp > 0 else 0
def recall(tp, fp, fn): return tp / (tp + fn) if tp > 0 else 0
def accuracy(tp, fp, fn): return tp / (tp + fp + fn) if tp > 0 else 0
def f1(tp, fp, fn): return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0


# --- public matching API --------------------------------------------------

def matching(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresh: float = 0.5,
    criterion: str = "iou",
    report_matches: bool = False,
):
    """Compute matching stats between two label images at one or more thresholds."""
    _check_label_array(y_true, "y_true")
    _check_label_array(y_pred, "y_pred")
    y_true.shape == y_pred.shape or _raise(
        ValueError(f"y_true {y_true.shape} and y_pred {y_pred.shape} have different shapes")
    )
    criterion in _MATCHING_CRITERIA or _raise(
        ValueError(f"Unknown matching criterion '{criterion}'")
    )
    if thresh is None:
        thresh = 0
    thresh = float(thresh) if np.isscalar(thresh) else list(map(float, thresh))

    y_true, _, map_rev_true = relabel_sequential(y_true)
    y_pred, _, map_rev_pred = relabel_sequential(y_pred)

    overlap = label_overlap(y_true, y_pred, check=False)
    scores = _MATCHING_CRITERIA[criterion](overlap)
    assert 0 <= np.min(scores) <= np.max(scores) <= 1

    scores = scores[1:, 1:]  # drop background row/col
    n_true, n_pred = scores.shape
    n_matched = min(n_true, n_pred)

    def _single(thr: float):
        not_trivial = n_matched > 0 and np.any(scores >= thr)
        true_ind, pred_ind, match_ok = (), (), np.array([], dtype=bool)
        if not_trivial:
            costs = -(scores >= thr).astype(float) - scores / (2 * n_matched)
            true_ind, pred_ind = linear_sum_assignment(costs)
            match_ok = scores[true_ind, pred_ind] >= thr
        tp = int(np.count_nonzero(match_ok))
        fp = n_pred - tp
        fn = n_true - tp

        sum_score = float(np.sum(scores[true_ind, pred_ind][match_ok])) if not_trivial else 0.0
        mean_matched = _safe_divide(sum_score, tp)
        mean_true = _safe_divide(sum_score, n_true)
        panoptic = _safe_divide(sum_score, tp + fp / 2 + fn / 2)

        d = dict(
            criterion=criterion, thresh=thr,
            fp=fp, tp=tp, fn=fn,
            precision=precision(tp, fp, fn),
            recall=recall(tp, fp, fn),
            accuracy=accuracy(tp, fp, fn),
            f1=f1(tp, fp, fn),
            n_true=n_true, n_pred=n_pred,
            mean_true_score=mean_true,
            mean_matched_score=mean_matched,
            panoptic_quality=panoptic,
        )
        if report_matches:
            if not_trivial:
                d.update(
                    matched_pairs=tuple(
                        (int(map_rev_true[i]), int(map_rev_pred[j]))
                        for i, j in zip(1 + true_ind, 1 + pred_ind)
                    ),
                    matched_scores=tuple(scores[true_ind, pred_ind]),
                    matched_tps=tuple(map(int, np.flatnonzero(match_ok))),
                )
            else:
                d.update(matched_pairs=(), matched_scores=(), matched_tps=())
        return namedtuple("Matching", d.keys())(*d.values())

    return _single(thresh) if np.isscalar(thresh) else tuple(map(_single, thresh))


def matching_dataset(
    y_true,
    y_pred,
    thresh=0.5,
    criterion: str = "iou",
    by_image: bool = False,
    show_progress: bool = True,
):
    """Average matching stats over a dataset of label image pairs."""
    len(y_true) == len(y_pred) or _raise(
        ValueError("y_true and y_pred must have the same length.")
    )

    single_thresh = np.isscalar(thresh)
    threshs = (thresh,) if single_thresh else tuple(thresh)

    pairs = list(zip(y_true, y_pred))
    stats_all = tuple(
        matching(yt, yp, thresh=threshs, criterion=criterion, report_matches=False)
        for yt, yp in tqdm(pairs, disable=not show_progress)
    )

    keys = {
        "fp", "tp", "fn", "precision", "recall", "accuracy", "f1",
        "criterion", "thresh", "n_true", "n_pred",
        "mean_true_score", "mean_matched_score", "panoptic_quality",
    }
    n_images = len(stats_all)
    accums = [{} for _ in threshs]
    for stats in stats_all:
        for i, s in enumerate(stats):
            acc = accums[i]
            for k, v in s._asdict().items():
                if k == "mean_true_score" and not by_image:
                    acc[k] = acc.setdefault(k, 0) + v * s.n_true
                else:
                    try:
                        acc[k] = acc.setdefault(k, 0) + v
                    except TypeError:
                        pass

    out = []
    for thr, acc in zip(threshs, accums):
        set(acc.keys()) == keys or _raise(ValueError("unexpected keys"))
        acc["criterion"] = criterion
        acc["thresh"] = thr
        acc["by_image"] = by_image
        if by_image:
            for k in ("precision", "recall", "accuracy", "f1",
                      "mean_true_score", "mean_matched_score", "panoptic_quality"):
                acc[k] /= n_images
        else:
            tp, fp, fn, n_true = acc["tp"], acc["fp"], acc["fn"], acc["n_true"]
            sum_score = acc["mean_true_score"]
            acc.update(
                precision=precision(tp, fp, fn),
                recall=recall(tp, fp, fn),
                accuracy=accuracy(tp, fp, fn),
                f1=f1(tp, fp, fn),
                mean_true_score=_safe_divide(sum_score, n_true),
                mean_matched_score=_safe_divide(sum_score, tp),
                panoptic_quality=_safe_divide(sum_score, tp + fp / 2 + fn / 2),
            )
        out.append(namedtuple("DatasetMatching", acc.keys())(*acc.values()))

    return out[0] if single_thresh else tuple(out)
