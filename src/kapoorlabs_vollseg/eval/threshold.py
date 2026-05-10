"""Tune StarDist ``prob_thresh`` / ``nms_thresh`` against a labelled set.

Sweeps a grid of ``nms_thresh`` values; for each, golden-section search on
``prob_thresh`` to maximize the chosen matching metric (default: accuracy)
averaged over a list of IoU thresholds.

Unlike the original ``OptimizeThreshold``, this works against any
:class:`kapoorlabs_vollseg.pipelines.Pipeline` — it doesn't hard-code the legacy
``VollSeg()`` function.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from collections.abc import Sequence

import numpy as np
from csbdeep.utils import _raise, normalize
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from .matching import matching_dataset


class OptimizeThreshold:
    """Search ``(prob_thresh, nms_thresh)`` that maximize a matching metric.

    Parameters
    ----------
    pipeline
        Any pipeline; ``predict(image, prob_thresh=..., nms_thresh=...)`` is
        invoked per candidate setting. Must return ``Result.labels``.
    images, labels
        Validation images and ground-truth label images.
    nms_threshs, iou_threshs
        The search grid for NMS, and the IoU thresholds at which the metric
        is averaged.
    measure
        Matching metric to maximize: one of ``accuracy``, ``precision``,
        ``recall``, ``f1``.
    """

    def __init__(
        self,
        pipeline,
        images: Sequence[np.ndarray],
        labels: Sequence[np.ndarray],
        *,
        nms_threshs: Sequence[float] = (0.0, 0.3, 0.4, 0.5),
        iou_threshs: Sequence[float] = (0.3, 0.5, 0.7),
        measure: str = "accuracy",
        n_tiles: Optional[tuple] = None,
        savedir: Optional[Path] = None,
        normalize_inputs: bool = True,
        norm_axes: tuple = (0, 1),
    ):
        self.pipeline = pipeline
        self.labels = list(labels)
        if normalize_inputs:
            self.images = [normalize(x, 1, 99.8, axis=norm_axes) for x in images]
        else:
            self.images = list(images)
        self.nms_threshs = tuple(nms_threshs)
        self.iou_threshs = tuple(iou_threshs)
        self.measure = measure
        self.n_tiles = n_tiles
        self.savedir = Path(savedir) if savedir is not None else None

        self.thresholds: dict = {}

    def run(
        self, *, tol: float = 1e-2, maxiter: int = 20, verbose: bool = True
    ) -> dict:
        best_prob, best_score, best_nms = None, -np.inf, None
        for nms in self.nms_threshs:
            prob, score = self._optimize_one(
                nms, tol=tol, maxiter=maxiter, verbose=verbose
            )
            if score > best_score:
                best_prob, best_score, best_nms = prob, score, nms

        self.thresholds = dict(prob=float(best_prob), nms=float(best_nms))
        if self.savedir is not None:
            self.savedir.mkdir(parents=True, exist_ok=True)
            (self.savedir / "thresholds.json").write_text(json.dumps(self.thresholds))
        return self.thresholds

    def _optimize_one(
        self, nms_thresh: float, *, tol: float, maxiter: int, verbose: bool
    ):
        np.isscalar(nms_thresh) or _raise(ValueError("nms_thresh must be a scalar"))
        cache: dict = {}

        with tqdm(
            total=maxiter, disable=not verbose, desc=f"NMS = {nms_thresh:g}"
        ) as bar:

            def fn(prob_thresh: float) -> float:
                if prob_thresh in cache:
                    return -cache[prob_thresh]

                preds = [
                    self.pipeline.predict(
                        x,
                        prob_thresh=float(prob_thresh),
                        nms_thresh=float(nms_thresh),
                        n_tiles=self.n_tiles,
                    ).labels
                    for x in self.images
                ]
                stats = matching_dataset(
                    self.labels, preds, thresh=self.iou_threshs, show_progress=False
                )
                value = float(np.mean([getattr(s, self.measure) for s in stats]))
                cache[prob_thresh] = value
                bar.update()
                bar.set_postfix_str(
                    f"prob={prob_thresh:.3f} -> {self.measure}={value:.3f}"
                )
                return -value

            opt = minimize_scalar(
                fn, method="golden", tol=tol, options={"maxiter": maxiter}
            )

        return opt.x, -opt.fun
