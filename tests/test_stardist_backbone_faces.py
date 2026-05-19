"""Tests for ``StarDistBackbone.faces`` — the ConvexHull triangulation
the inference rasteriser uses. Constructing the backbone with a real
:class:`StarDistModule` requires torch + careamics; gate accordingly.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("careamics")

from kapoorlabs_vollseg._backbones.stardist import StarDistBackbone  # noqa: E402
from kapoorlabs_vollseg.stardist import (  # noqa: E402
    StarDistModule,
    StarDistUNet,
    rays_2d,
    rays_3d_golden_spiral,
)


def _make_module(n_rays: int, conv_dims: int) -> StarDistModule:
    """Build a fresh (untrained) StarDistModule — only needed for its
    ``n_rays`` attribute and the eval() call in the backbone ctor."""
    network = StarDistUNet(
        n_rays=n_rays,
        conv_dims=conv_dims,
        in_channels=1,
        depth=2,
        num_channels_init=8,
        use_batch_norm=False,
    )
    return StarDistModule(network=network, optim_func=None)


class TestBackbonePopulatesFaces:
    def test_3d_backbone_has_triangulated_faces(self):
        n_rays = 48
        rays = rays_3d_golden_spiral(n_rays)
        module = _make_module(n_rays=n_rays, conv_dims=3)
        backbone = StarDistBackbone(module, rays)
        # 2n - 4 triangles for any ConvexHull of n points on a sphere.
        assert backbone.faces.shape == (2 * n_rays - 4, 3)
        # All indices must address into rays.
        assert backbone.faces.min() >= 0
        assert backbone.faces.max() < n_rays

    def test_2d_backbone_has_empty_faces(self):
        n_rays = 32
        rays = rays_2d(n_rays)
        module = _make_module(n_rays=n_rays, conv_dims=2)
        backbone = StarDistBackbone(module, rays)
        # 2D has no triangulation — empty (0, 3) so downstream takes the
        # cone-fallback path uniformly.
        assert backbone.faces.shape == (0, 3)

    def test_faces_match_module_rays_count(self):
        n_rays = 64
        rays = rays_3d_golden_spiral(n_rays)
        module = _make_module(n_rays=n_rays, conv_dims=3)
        backbone = StarDistBackbone(module, rays)
        # Every face index must be a valid ray index.
        assert set(np.unique(backbone.faces).tolist()) <= set(range(n_rays))
