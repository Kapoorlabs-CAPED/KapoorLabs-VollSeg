"""ROIPipeline must crop downstream input to the ROI bbox and paste labels back."""

from __future__ import annotations

import numpy as np

from kapoorlabs_vollseg import Result
from kapoorlabs_vollseg.pipelines.roi import ROIPipeline, _spatial_bbox


# ---------------------------------------------- helpers


class _FakeROI:
    """Returns a fixed binary mask as ``semantic`` regardless of input."""

    def __init__(self, mask: np.ndarray):
        self.mask = mask.astype(bool)

    def predict(self, image, **kwargs) -> Result:
        return Result(semantic=self.mask)


class _RecordingDownstream:
    """Captures the patch its caller hands to ``predict`` so the test can
    assert the patch is the bbox-cropped sub-image (not the whole thing)."""

    def __init__(self):
        self.seen_shapes: list[tuple] = []

    def predict(self, image, **kwargs) -> Result:
        self.seen_shapes.append(tuple(image.shape))
        # Mark every voxel of the patch with a unique label so we can
        # check the paste is at the bbox position.
        labels = np.arange(1, image.size + 1, dtype=np.uint32).reshape(image.shape)
        semantic = np.ones(image.shape, dtype=bool)
        return Result(labels=labels, semantic=semantic)


# ---------------------------------------------- _spatial_bbox


def test_spatial_bbox_2d_corner():
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 5:9] = True
    bbox = _spatial_bbox(mask)
    assert bbox == (slice(3, 7), slice(5, 9))


def test_spatial_bbox_empty_returns_none():
    assert _spatial_bbox(np.zeros((5, 5), dtype=bool)) is None


def test_spatial_bbox_z_broadcast_keeps_z_full():
    """A 3D mask that's the same across Z (the 2D-ROI-on-3D-image case
    we wired in MaskUNetSegmenter.predict) shouldn't crop Z."""
    mask = np.zeros((5, 10, 10), dtype=bool)
    mask[:, 2:6, 3:8] = True
    bbox = _spatial_bbox(mask)
    assert bbox[0] == slice(None)
    assert bbox[1] == slice(2, 6)
    assert bbox[2] == slice(3, 8)


# ---------------------------------------------- ROIPipeline.predict


def test_roi_pipeline_crops_to_bbox_2d():
    img = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 5:9] = True

    rec = _RecordingDownstream()
    pipe = ROIPipeline(_FakeROI(mask), rec)
    out = pipe.predict(img)

    # Downstream saw the bbox patch, not the full 10x10.
    assert rec.seen_shapes == [(4, 4)]
    # Labels pasted back into full 10x10, non-zero only inside the bbox.
    assert out.labels.shape == img.shape
    assert (out.labels[3:7, 5:9] > 0).all()
    assert (out.labels[~mask] == 0).all()
    # ROI mask is forwarded.
    assert np.array_equal(out.roi, mask)


def test_roi_pipeline_crops_yx_only_when_mask_is_z_broadcast():
    """3D image + Z-broadcast 2D ROI → downstream sees full Z but cropped YX."""
    img = np.zeros((5, 10, 10), dtype=np.float32)
    mask = np.zeros((5, 10, 10), dtype=bool)
    mask[:, 2:6, 3:8] = True

    rec = _RecordingDownstream()
    out = ROIPipeline(_FakeROI(mask), rec).predict(img)

    assert rec.seen_shapes == [(5, 4, 5)]
    assert out.labels.shape == img.shape
    assert (out.labels[:, 2:6, 3:8] > 0).all()
    assert (out.labels[:, :2, :] == 0).all()


def test_roi_pipeline_empty_mask_returns_zero():
    img = np.ones((4, 8, 8), dtype=np.float32)
    mask = np.zeros((4, 8, 8), dtype=bool)

    rec = _RecordingDownstream()
    out = ROIPipeline(_FakeROI(mask), rec).predict(img)

    # Downstream never called.
    assert rec.seen_shapes == []
    assert out.labels.shape == img.shape
    assert not out.labels.any()
    assert not out.semantic.any()
