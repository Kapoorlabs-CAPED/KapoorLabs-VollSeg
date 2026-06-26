"""Tests for VollSeg.from_models composition logic.

Covers the full composition matrix documented in the top-level
``README.md`` Architecture section: which pipeline shape the factory
produces for every (care, roi_unet, unet, stardist, seedpool)
combination, plus the predict-time data-flow contract — denoise → ROI
→ segmentation core, with each ``Result.*`` stage output populated
only when the corresponding model runs.

The pipelines are tested with minimal in-memory fakes (no real
networks, no I/O) so the assertions exercise composition shape and
field plumbing only.
"""

from __future__ import annotations

import numpy as np
import pytest

from kapoorlabs_vollseg.pipelines.base import Pipeline, Result
from kapoorlabs_vollseg.pipelines.chunked import Chunked
from kapoorlabs_vollseg.pipelines.denoised import DenoisedPipeline
from kapoorlabs_vollseg.pipelines.factory import VollSeg
from kapoorlabs_vollseg.pipelines.roi import ROIPipeline
from kapoorlabs_vollseg.pipelines.unet_stardist import UNetStarDistPipeline


# ───────────────────────────── fakes ─────────────────────────────────


class FakeCARE(Pipeline):
    """Denoiser fake — outputs image+1 so seen-by-downstream is verifiable."""

    def __init__(self):
        self.last_input = None

    def predict(self, image, **kw):
        self.last_input = image
        return Result(denoised=image + 1)


class FakeROI(Pipeline):
    """ROI Mask-UNet fake — returns a centred half-volume mask."""

    def __init__(self):
        self.last_input = None

    def predict(self, image, **kw):
        self.last_input = image
        mask = np.zeros(image.shape, dtype=bool)
        s = tuple(slice(d // 4, 3 * d // 4) for d in image.shape)
        mask[s] = True
        return Result(semantic=mask, labels=mask.astype(np.uint32))


class FakeUNet(Pipeline):
    """U-Net fake — emits ones-everywhere binary mask + probability."""

    def __init__(self):
        self.last_input = None

    def predict(self, image, **kw):
        self.last_input = image
        return Result(
            semantic=np.ones(image.shape, dtype=bool),
            labels=np.ones(image.shape, dtype=np.uint32),
            probability=np.ones(image.shape, dtype=np.float32) * 0.7,
        )


class FakeStarDist(Pipeline):
    """StarDist fake — emits constant instance labels + polys metadata."""

    def __init__(self):
        self.last_input = None

    def predict(self, image, **kw):
        self.last_input = image
        return Result(
            labels=np.ones(image.shape, dtype=np.uint32),
            probability=np.ones(image.shape, dtype=np.float32) * 0.9,
            polys={"fake": True},
        )


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.normal(size=(8, 16, 16)).astype(np.float32)


# ─────────────────────────── failure mode ───────────────────────────


class TestFailureMode:
    """The only ``ValueError`` the factory ever raises is no-model-supplied."""

    def test_no_model_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            VollSeg.from_models()


# ───────────────────────── permissive seedpool ──────────────────────


class TestSeedpoolPermissive:
    """``seedpool=True`` is silently ignored when prerequisites aren't met."""

    def test_unet_only_seedpool_falls_back_to_bare_unet(self):
        u = FakeUNet()
        pipe = VollSeg.from_models(unet=u, seedpool=True)
        assert pipe is u  # bare U-Net singleton — no fusion possible

    def test_stardist_only_seedpool_falls_back_to_bare_stardist(self):
        s = FakeStarDist()
        pipe = VollSeg.from_models(stardist=s, seedpool=True)
        assert pipe is s  # bare StarDist singleton — no mask source

    def test_care_only_seedpool_falls_back_to_bare_care(self):
        c = FakeCARE()
        pipe = VollSeg.from_models(care=c, seedpool=True)
        assert pipe is c

    def test_roi_only_seedpool_falls_back_to_bare_roi(self):
        r = FakeROI()
        pipe = VollSeg.from_models(roi_unet=r, seedpool=True)
        assert pipe is r


# ──────────────────────────── singletons ────────────────────────────


class TestSingletons:
    """Single-model configs return the bare singleton — no wrapping."""

    def test_bare_stardist(self):
        s = FakeStarDist()
        assert VollSeg.from_models(stardist=s) is s

    def test_bare_unet(self):
        u = FakeUNet()
        assert VollSeg.from_models(unet=u) is u

    def test_bare_care(self):
        c = FakeCARE()
        assert VollSeg.from_models(care=c) is c

    def test_bare_roi(self):
        r = FakeROI()
        assert VollSeg.from_models(roi_unet=r) is r


# ──────────────────────── composition shapes ────────────────────────


class TestCompositionShape:
    """Outer-most wrapper is the right type for each multi-model combo."""

    def test_care_plus_stardist(self):
        pipe = VollSeg.from_models(care=FakeCARE(), stardist=FakeStarDist())
        assert isinstance(pipe, DenoisedPipeline)

    def test_care_plus_unet(self):
        pipe = VollSeg.from_models(care=FakeCARE(), unet=FakeUNet())
        assert isinstance(pipe, DenoisedPipeline)

    def test_roi_plus_stardist(self):
        pipe = VollSeg.from_models(roi_unet=FakeROI(), stardist=FakeStarDist())
        assert isinstance(pipe, ROIPipeline)

    def test_unet_plus_stardist_no_seedpool(self):
        pipe = VollSeg.from_models(unet=FakeUNet(), stardist=FakeStarDist())
        assert isinstance(pipe, UNetStarDistPipeline)
        assert pipe.seedpool is False

    def test_unet_plus_stardist_with_seedpool(self):
        pipe = VollSeg.from_models(
            unet=FakeUNet(), stardist=FakeStarDist(), seedpool=True
        )
        assert isinstance(pipe, UNetStarDistPipeline)
        assert pipe.seedpool is True

    def test_care_plus_stardist_with_seedpool_no_unet(self):
        """care + stardist + seedpool=True → Denoised(UNetStarDist(unet=None))."""
        pipe = VollSeg.from_models(
            care=FakeCARE(), stardist=FakeStarDist(), seedpool=True
        )
        assert isinstance(pipe, DenoisedPipeline)
        inner = pipe.downstream
        assert isinstance(inner, UNetStarDistPipeline)
        assert inner.unet is None
        assert inner.seedpool is True

    def test_full_stack(self):
        """care + roi + unet + stardist + seedpool → Denoised(ROI(UNetStarDist))."""
        pipe = VollSeg.from_models(
            care=FakeCARE(),
            roi_unet=FakeROI(),
            unet=FakeUNet(),
            stardist=FakeStarDist(),
            seedpool=True,
        )
        assert isinstance(pipe, DenoisedPipeline)
        assert isinstance(pipe.downstream, ROIPipeline)
        assert isinstance(pipe.downstream.downstream, UNetStarDistPipeline)

    def test_chunk_wraps_everything(self):
        pipe = VollSeg.from_models(stardist=FakeStarDist(), chunk=(4, 8, 8))
        assert isinstance(pipe, Chunked)


# ─────────────────────── decoration order ───────────────────────────


class TestDecorationOrder:
    """Verifies the predict-time order: denoise → ROI → segment.

    The factory composes as ``Denoised( ROI( inner ) )`` — so denoise
    runs first, then ROI operates on the denoised image. This guards
    against regressing to the earlier (incorrect) ``ROI( Denoised( ))``
    order where ROI saw the raw image.
    """

    def test_roi_sees_denoised_image(self, image):
        care = FakeCARE()
        roi = FakeROI()
        star = FakeStarDist()
        pipe = VollSeg.from_models(care=care, roi_unet=roi, stardist=star)

        pipe.predict(image)

        # CARE saw the raw input; ROI saw image+1 (the denoised output).
        np.testing.assert_array_equal(care.last_input, image)
        np.testing.assert_array_equal(roi.last_input, image + 1)


# ───────────────────────── Result field plumbing ────────────────────


class TestResultFields:
    """Each ``Result.*`` field is populated only when its model ran."""

    def test_full_stack_populates_every_field(self, image):
        pipe = VollSeg.from_models(
            care=FakeCARE(),
            roi_unet=FakeROI(),
            unet=FakeUNet(),
            stardist=FakeStarDist(),
            seedpool=True,
        )
        out = pipe.predict(image)
        assert out.labels is not None
        assert out.vollseg_labels is not None
        assert out.stardist_labels is not None
        assert out.unet_labels is not None
        assert out.semantic is not None
        assert out.denoised is not None
        assert out.roi is not None
        assert out.polys is not None
        np.testing.assert_array_equal(out.labels, out.vollseg_labels)

    def test_care_plus_stardist_no_unet_labels(self, image):
        pipe = VollSeg.from_models(care=FakeCARE(), stardist=FakeStarDist())
        out = pipe.predict(image)
        assert out.denoised is not None
        assert out.labels is not None
        assert out.unet_labels is None
        assert out.vollseg_labels is None  # no fusion ran
        assert out.roi is None  # no ROI model

    def test_care_plus_stardist_plus_seedpool_otsu_path(self, image):
        """No U-Net + seedpool → Otsu mask drives the fusion, no unet_labels."""
        pipe = VollSeg.from_models(
            care=FakeCARE(), stardist=FakeStarDist(), seedpool=True
        )
        out = pipe.predict(image)
        assert out.denoised is not None
        assert out.vollseg_labels is not None
        assert out.stardist_labels is not None
        assert out.semantic is not None  # Otsu-threshold mask
        assert out.unet_labels is None  # no U-Net ran

    def test_unet_plus_stardist_side_by_side(self, image):
        pipe = VollSeg.from_models(unet=FakeUNet(), stardist=FakeStarDist())
        out = pipe.predict(image)
        assert out.stardist_labels is not None
        assert out.unet_labels is not None
        assert out.semantic is not None
        assert out.vollseg_labels is None  # seedpool=False
        np.testing.assert_array_equal(out.labels, out.stardist_labels)

    def test_unet_plus_stardist_seedpool_fuses(self, image):
        pipe = VollSeg.from_models(
            unet=FakeUNet(), stardist=FakeStarDist(), seedpool=True
        )
        out = pipe.predict(image)
        assert out.vollseg_labels is not None
        assert out.stardist_labels is not None
        assert out.unet_labels is not None
        np.testing.assert_array_equal(out.labels, out.vollseg_labels)
