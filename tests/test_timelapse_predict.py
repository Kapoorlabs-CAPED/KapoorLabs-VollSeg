"""Tests for ``kapoorlabs_vollseg.predict_timelapse`` and its
``TimelapsePredictor`` Lightning shell.

Multi-GPU DDP behaviour can't be exercised in a unit test, but the
single-process path covers:

- per-frame dispatch ordering (each frame seen exactly once)
- T-axis stacking from the gathered results
- Result-field filtering (only fields the pipeline populates make it
  into the output dict, no ``None``-stuffed entries)
- the dedupe-by-T pass that hides DistributedSampler padding
"""

from __future__ import annotations

import numpy as np
import pytest

from kapoorlabs_vollseg.pipelines.base import Pipeline, Result


# Lightning isn't installed in every dev env; gate the whole module.
lightning = pytest.importorskip("lightning")


from kapoorlabs_vollseg import predict_timelapse  # noqa: E402
from kapoorlabs_vollseg.pipelines.timelapse_predict import (  # noqa: E402
    TimelapsePredictor,
)


class _LabelsEchoPipeline(Pipeline):
    """Fake pipeline whose ``labels`` output is just the input frame cast
    to uint16 — lets us verify that each frame reaches the pipeline and
    lands at the right T index after gather/sort/stack."""

    def predict(self, image, **kwargs):
        return Result(labels=image.astype(np.uint16))


class _DenoiserPipeline(Pipeline):
    """Pipeline that populates the ``denoised`` field only — verifies
    the result-dict filter drops unpopulated fields."""

    def predict(self, image, **kwargs):
        return Result(denoised=image.astype(np.float32) * 2.0)


class _KwargCapturePipeline(Pipeline):
    """Captures the kwargs each call receives, so we can verify
    ``predict_kwargs`` is forwarded."""

    def __init__(self):
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(dict(kwargs))
        return Result(labels=np.zeros_like(image, dtype=np.uint16))


class TestPredictTimelapseSingleProcess:
    def test_stacks_in_t_order(self):
        # 4 frames of distinct content; the stacked labels stack must
        # come back in original T-order.
        volume = np.stack(
            [np.full((4, 4), i, dtype=np.uint16) for i in range(4)], axis=0
        )
        out = predict_timelapse(
            _LabelsEchoPipeline(),
            volume,
            devices=1,
            accelerator="cpu",
            enable_progress_bar=False,
        )
        assert "labels" in out
        assert out["labels"].shape == volume.shape
        # Each frame's content equals its T-index.
        for t in range(volume.shape[0]):
            assert int(out["labels"][t].max()) == t

    def test_unpopulated_fields_absent(self):
        volume = np.zeros((3, 4, 4), dtype=np.float32)
        out = predict_timelapse(
            _DenoiserPipeline(),
            volume,
            devices=1,
            accelerator="cpu",
            enable_progress_bar=False,
        )
        # Only "denoised" was populated; nothing else should appear.
        assert set(out.keys()) == {"denoised"}
        assert out["denoised"].shape == volume.shape

    def test_kwargs_forwarded_per_frame(self):
        cap = _KwargCapturePipeline()
        volume = np.zeros((3, 4, 4), dtype=np.uint16)
        predict_timelapse(
            cap,
            volume,
            devices=1,
            accelerator="cpu",
            enable_progress_bar=False,
            prob_thresh=0.42,
            n_tiles=(1, 2, 2),
        )
        # Each frame got the same kwargs.
        assert len(cap.calls) == 3
        for k in cap.calls:
            assert k["prob_thresh"] == 0.42
            assert k["n_tiles"] == (1, 2, 2)

    def test_3d_timelapse_zyx_per_frame(self):
        # T=3, Z=2, Y=4, X=4 — confirms the helper doesn't insist on 2D
        # frames; each frame is a 3D volume passed through unchanged.
        volume = np.stack(
            [np.full((2, 4, 4), i, dtype=np.uint16) for i in range(3)], axis=0
        )
        out = predict_timelapse(
            _LabelsEchoPipeline(),
            volume,
            devices=1,
            accelerator="cpu",
            enable_progress_bar=False,
        )
        assert out["labels"].shape == volume.shape
        for t in range(3):
            assert int(out["labels"][t].max()) == t


class TestTimelapsePredictorPredictStep:
    """Direct unit test of TimelapsePredictor.predict_step — it just
    delegates to ``pipeline.predict`` and wraps the result into a dict
    with the T-index attached for gather/sort downstream."""

    def test_returns_t_and_result_fields(self):
        predictor = TimelapsePredictor(_LabelsEchoPipeline())
        frame = np.full((4, 4), 7, dtype=np.uint16)
        out = predictor.predict_step((frame, 3), batch_idx=0)
        assert out["t"] == 3
        assert out["labels"].dtype == np.uint16
        assert out["labels"].max() == 7
        # Other fields are None — caller's gather/stack filter skips them.
        for k in ("semantic", "roi", "denoised", "probability"):
            assert out[k] is None

    def test_predict_kwargs_propagate(self):
        cap = _KwargCapturePipeline()
        predictor = TimelapsePredictor(cap, predict_kwargs={"prob_thresh": 0.5})
        predictor.predict_step((np.zeros((2, 2), dtype=np.uint16), 0), batch_idx=0)
        assert cap.calls == [{"prob_thresh": 0.5}]
