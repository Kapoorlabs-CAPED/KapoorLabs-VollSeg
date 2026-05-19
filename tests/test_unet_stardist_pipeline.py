"""UNetStarDistPipeline modes — seedpool on/off."""

from __future__ import annotations

import numpy as np

from kapoorlabs_vollseg import Result
from kapoorlabs_vollseg.pipelines.unet_stardist import UNetStarDistPipeline


class _FakeStarDist:
    """Returns a fixed 3D instance label image."""

    def __init__(self, labels: np.ndarray):
        self.labels = labels.astype(np.uint32)

    def predict(self, image, **kwargs) -> Result:
        return Result(
            labels=self.labels, probability=np.ones(image.shape, dtype=np.float32)
        )


class _FakeUNet:
    """Returns a fixed binary U-Net mask."""

    def __init__(self, mask: np.ndarray):
        self.mask = mask.astype(bool)

    def predict(self, image, **kwargs) -> Result:
        return Result(
            labels=self.mask.astype(np.uint32),
            semantic=self.mask,
            probability=self.mask.astype(np.float32),
        )


def _two_centroid_volume():
    img = np.zeros((6, 16, 16), dtype=np.float32)
    star = np.zeros_like(img, dtype=np.uint32)
    # StarDist sees one cell at the top.
    star[2:5, 3:7, 3:7] = 1
    unet = np.zeros_like(img, dtype=bool)
    # U-Net sees both cells — top one matches StarDist, bottom one is missed.
    unet[2:5, 3:7, 3:7] = True
    unet[2:5, 10:14, 10:14] = True
    return img, star, unet


def test_seedpool_false_returns_stardist_labels_and_unet_semantic():
    img, star, unet = _two_centroid_volume()
    pipe = UNetStarDistPipeline(_FakeUNet(unet), _FakeStarDist(star), seedpool=False)
    out = pipe.predict(img)
    # Without seedpool, labels are exactly StarDist's labels.
    assert np.array_equal(out.labels, star)
    # Semantic is U-Net's binary mask (so the unmatched second cell shows
    # up in `semantic` even though `labels` doesn't have it).
    assert np.array_equal(out.semantic.astype(bool), unet)


def test_seedpool_true_pools_unet_seeds_for_missed_cells():
    img, star, unet = _two_centroid_volume()
    pipe = UNetStarDistPipeline(_FakeUNet(unet), _FakeStarDist(star), seedpool=True)
    out = pipe.predict(img)
    # With seedpool the second (U-Net-only) cell should now be present
    # in the fused labels — the StarDist-only path would have count 1.
    n_objects = int(out.labels.max())
    assert n_objects >= 2, f"seedpool should recover the missed cell; got {n_objects}"
