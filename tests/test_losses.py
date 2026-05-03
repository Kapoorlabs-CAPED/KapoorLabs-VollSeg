"""Tests for vollseg.stardist losses — shapes and trivial-case values.

Skipped if torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vollseg.stardist import dist_loss, prob_loss, stardist_loss


@pytest.fixture
def fake_outputs():
    """Tiny (1, 1, 4, 4) prob_logits + (1, 8, 4, 4) dists fixtures."""
    torch.manual_seed(0)
    prob_logits = torch.zeros(1, 1, 4, 4)
    dists = torch.zeros(1, 8, 4, 4)
    return prob_logits, dists


@pytest.fixture
def fake_targets():
    prob_target = torch.zeros(1, 1, 4, 4)
    prob_target[..., 1:3, 1:3] = 1.0      # 2×2 foreground patch in the middle
    dist_target = torch.zeros(1, 8, 4, 4)
    dist_target[..., 1:3, 1:3] = 5.0
    return prob_target, dist_target


def test_prob_loss_shape(fake_outputs, fake_targets):
    prob_logits, _ = fake_outputs
    prob_target, _ = fake_targets
    loss = prob_loss(prob_logits, prob_target)
    assert loss.dim() == 0   # scalar
    assert loss.item() > 0   # logits=0 vs target≠0 → nonzero loss


def test_dist_loss_zero_when_perfect(fake_targets):
    """When predicted dist == target dist on the foreground, dist_loss = 0."""
    prob_target, dist_target = fake_targets
    loss = dist_loss(dist_target, dist_target, prob_target)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_dist_loss_ignores_background(fake_outputs, fake_targets):
    """Bad predictions outside the foreground shouldn't move the loss."""
    prob_logits, dists = fake_outputs
    prob_target, dist_target = fake_targets
    # Garbage in the background shouldn't increase loss vs zero everywhere.
    dist_pred_bad = dist_target.clone()
    dist_pred_bad[..., 0, 0] = 999.0     # corner is background → should be ignored
    loss_perfect = dist_loss(dist_target, dist_target, prob_target)
    loss_bad_bg = dist_loss(dist_pred_bad, dist_target, prob_target)
    assert loss_bad_bg.item() == pytest.approx(loss_perfect.item(), abs=1e-7)


def test_stardist_loss_returns_three_values(fake_outputs, fake_targets):
    prob_logits, dists = fake_outputs
    prob_target, dist_target = fake_targets
    total, p_term, d_term = stardist_loss(prob_logits, dists, prob_target, dist_target, lam=0.5)
    assert total.dim() == 0
    assert p_term.dim() == 0
    assert d_term.dim() == 0
    # total = p + lam·d
    np.testing.assert_allclose(
        total.item(), p_term.item() + 0.5 * d_term.item(), atol=1e-6
    )
