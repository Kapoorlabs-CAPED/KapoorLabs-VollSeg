"""Tests for VollSeg.from_models — input validation rules."""

from __future__ import annotations

import numpy as np
import pytest

from kapoorlabs_vollseg import Result, VollCellSeg, VollSeg


class _FakePipeline:
    """Minimal Pipeline implementation for composition tests."""

    def __init__(self, label_value: int = 1):
        self.label_value = label_value

    def predict(self, image, **kwargs) -> Result:
        return Result(
            labels=np.full(image.shape, self.label_value, dtype=np.uint16),
            semantic=np.ones(image.shape, dtype=bool),
            denoised=image,
        )


class TestVollSegFactory:
    def test_requires_at_least_one_model(self):
        with pytest.raises(ValueError):
            VollSeg.from_models()

    def test_seedpool_silently_ignored_when_prereqs_missing(self):
        # Permissive composition: ``seedpool=True`` is dropped (not raised)
        # when its prerequisites aren't met. With only stardist, the
        # mask-source half (unet OR care) is missing → fall back to bare
        # StarDist. With only unet, there's nothing to fuse → fall back
        # to bare U-Net.
        s = _FakePipeline()
        assert VollSeg.from_models(stardist=s, seedpool=True) is s
        u = _FakePipeline()
        assert VollSeg.from_models(unet=u, seedpool=True) is u

    def test_stardist_only_returns_singleton(self):
        s = _FakePipeline()
        pipe = VollSeg.from_models(stardist=s)
        assert pipe is s

    def test_unet_only_returns_singleton(self):
        u = _FakePipeline()
        pipe = VollSeg.from_models(unet=u)
        assert pipe is u

    def test_care_only_returns_singleton(self):
        # care-only is a degenerate "denoise as the whole pipeline".
        c = _FakePipeline()
        pipe = VollSeg.from_models(care=c)
        assert pipe is c

    def test_care_wraps_inner(self):
        # care + stardist should compose into DenoisedPipeline(care, stardist)
        # rather than returning either bare. We probe by checking the type.
        from kapoorlabs_vollseg.pipelines.denoised import DenoisedPipeline

        c = _FakePipeline()
        s = _FakePipeline()
        pipe = VollSeg.from_models(care=c, stardist=s)
        assert isinstance(pipe, DenoisedPipeline)

    def test_roi_wraps_outermost(self):
        from kapoorlabs_vollseg.pipelines.roi import ROIPipeline

        s = _FakePipeline()
        roi = _FakePipeline()
        pipe = VollSeg.from_models(stardist=s, roi_unet=roi)
        assert isinstance(pipe, ROIPipeline)


class TestVollCellSegFactory:
    def test_requires_cellpose(self):
        with pytest.raises(ValueError):
            VollCellSeg.from_models()

    def test_cellpose_only_returns_singleton(self):
        c = _FakePipeline()
        pipe = VollCellSeg.from_models(cellpose=c)
        assert pipe is c

    def test_with_nuclei_pipeline_wraps(self):
        from kapoorlabs_vollseg.pipelines.nuclei_cellpose import (
            NucleiSeededCellPosePipeline,
        )

        c = _FakePipeline()
        n = _FakePipeline()
        pipe = VollCellSeg.from_models(cellpose=c, nuclei_pipeline=n)
        assert isinstance(pipe, NucleiSeededCellPosePipeline)
