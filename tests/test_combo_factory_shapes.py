"""VollSeg.from_models routing for the user-asked combo modes:

- stardist + maskunet
- stardist + unet                       (seedpool=False)
- stardist + unet + maskunet            (seedpool=False)
- stardist + unet                       (seedpool=True)
- stardist + unet + maskunet            (seedpool=True)
"""

from __future__ import annotations

import numpy as np

from kapoorlabs_vollseg import Result, VollSeg
from kapoorlabs_vollseg.pipelines.roi import ROIPipeline
from kapoorlabs_vollseg.pipelines.unet_stardist import UNetStarDistPipeline


class _FakePipeline:
    def __init__(self, label_value: int = 1):
        self.label_value = label_value

    def predict(self, image, **kwargs) -> Result:
        return Result(
            labels=np.full(image.shape, self.label_value, dtype=np.uint32),
            semantic=np.ones(image.shape, dtype=bool),
            probability=np.ones(image.shape, dtype=np.float32),
        )


def test_stardist_plus_maskunet_is_roi_wrapped_stardist():
    s = _FakePipeline()
    r = _FakePipeline()
    pipe = VollSeg.from_models(stardist=s, roi_unet=r)
    assert isinstance(pipe, ROIPipeline)
    assert pipe.roi_unet is r
    assert pipe.downstream is s


def test_stardist_plus_unet_no_seedpool_is_unet_stardist_pipeline():
    s = _FakePipeline()
    u = _FakePipeline()
    pipe = VollSeg.from_models(stardist=s, unet=u, seedpool=False)
    assert isinstance(pipe, UNetStarDistPipeline)
    assert pipe.seedpool is False


def test_stardist_plus_unet_seedpool_true_is_unet_stardist_pipeline_with_seedpool():
    s = _FakePipeline()
    u = _FakePipeline()
    pipe = VollSeg.from_models(stardist=s, unet=u, seedpool=True)
    assert isinstance(pipe, UNetStarDistPipeline)
    assert pipe.seedpool is True


def test_full_combo_no_seedpool_is_roi_around_unet_stardist():
    s = _FakePipeline()
    u = _FakePipeline()
    r = _FakePipeline()
    pipe = VollSeg.from_models(stardist=s, unet=u, roi_unet=r, seedpool=False)
    assert isinstance(pipe, ROIPipeline)
    assert isinstance(pipe.downstream, UNetStarDistPipeline)
    assert pipe.downstream.seedpool is False


def test_full_combo_with_seedpool_is_roi_around_seedpool_fused_pipeline():
    s = _FakePipeline()
    u = _FakePipeline()
    r = _FakePipeline()
    pipe = VollSeg.from_models(stardist=s, unet=u, roi_unet=r, seedpool=True)
    assert isinstance(pipe, ROIPipeline)
    assert isinstance(pipe.downstream, UNetStarDistPipeline)
    assert pipe.downstream.seedpool is True
