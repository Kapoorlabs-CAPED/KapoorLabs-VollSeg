"""Tests for the ``_backbones._config`` helpers that back every
singleton's ``from_folder`` constructor.

Singletons rely on these to:

- Locate the ``.ckpt`` inside an arbitrary model folder.
- Pull architecture kwargs (``conv_dims``, ``depth``, …) out of the
  Hydra-dumped ``training_config.json`` so the loader doesn't need them
  passed at the call site.
- Pick up optimised ``prob_thresh`` / ``nms_thresh`` that the
  threshold-optimisation script writes back into the same JSON.
"""

from __future__ import annotations

import json

from kapoorlabs_vollseg._backbones._config import (
    find_checkpoint,
    find_rays,
    read_thresholds,
    read_training_config,
)


class TestReadTrainingConfig:
    def test_preferred_training_config_json(self, tmp_path):
        (tmp_path / "training_config.json").write_text(
            json.dumps(
                {
                    "parameters": {
                        "conv_dims": 3,
                        "unet_depth": 4,
                        "num_channels_init": 32,
                        "use_batch_norm": True,
                        "in_channels": 1,
                    }
                }
            )
        )
        out = read_training_config(tmp_path)
        # unet_depth gets renamed to depth for from_checkpoint compatibility.
        assert out == {
            "conv_dims": 3,
            "depth": 4,
            "num_channels_init": 32,
            "use_batch_norm": True,
            "in_channels": 1,
        }

    def test_fallback_flat_json(self, tmp_path):
        # No training_config.json, but a per-experiment JSON is present.
        (tmp_path / "my_experiment.json").write_text(
            json.dumps({"conv_dims": 2, "unet_depth": 3, "num_channels_init": 16})
        )
        out = read_training_config(tmp_path)
        assert out == {"conv_dims": 2, "depth": 3, "num_channels_init": 16}

    def test_prefers_training_config_over_fallback(self, tmp_path):
        (tmp_path / "training_config.json").write_text(
            json.dumps({"parameters": {"conv_dims": 3}})
        )
        (tmp_path / "other.json").write_text(json.dumps({"conv_dims": 2}))
        out = read_training_config(tmp_path)
        assert out == {"conv_dims": 3}

    def test_no_json_returns_empty(self, tmp_path):
        out = read_training_config(tmp_path)
        assert out == {}

    def test_unknown_keys_filtered_out(self, tmp_path):
        # _ARCH_FIELDS whitelist: anything not in it must be dropped.
        (tmp_path / "training_config.json").write_text(
            json.dumps(
                {"parameters": {"conv_dims": 3, "learning_rate": 1e-3, "batch_size": 4}}
            )
        )
        out = read_training_config(tmp_path)
        assert out == {"conv_dims": 3}


class TestFindCheckpoint:
    def test_finds_top_level(self, tmp_path):
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"")
        assert find_checkpoint(tmp_path) == ckpt

    def test_finds_nested(self, tmp_path):
        nested = tmp_path / "lightning_logs" / "version_0" / "checkpoints"
        nested.mkdir(parents=True)
        ckpt = nested / "epoch=0-step=10.ckpt"
        ckpt.write_bytes(b"")
        assert find_checkpoint(tmp_path) == ckpt

    def test_raises_when_no_ckpt(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            find_checkpoint(tmp_path)


class TestFindRays:
    def test_canonical_name(self, tmp_path):
        rays = tmp_path / "rays.npy"
        rays.write_bytes(b"")
        assert find_rays(tmp_path) == rays

    def test_glob_variant(self, tmp_path):
        rays = tmp_path / "nuclei_rays.npy"
        rays.write_bytes(b"")
        assert find_rays(tmp_path) == rays

    def test_missing_returns_none(self, tmp_path):
        assert find_rays(tmp_path) is None


class TestReadThresholds:
    def test_present_in_training_config(self, tmp_path):
        (tmp_path / "training_config.json").write_text(
            json.dumps({"parameters": {"prob_thresh": 0.61, "nms_thresh": 0.3}})
        )
        out = read_thresholds(tmp_path)
        assert out["prob_thresh"] == 0.61
        assert out["nms_thresh"] == 0.3

    def test_partial(self, tmp_path):
        # Only one of the two thresholds present — the other key is absent.
        (tmp_path / "training_config.json").write_text(
            json.dumps({"parameters": {"prob_thresh": 0.5}})
        )
        out = read_thresholds(tmp_path)
        assert out == {"prob_thresh": 0.5}

    def test_absent_returns_empty(self, tmp_path):
        (tmp_path / "training_config.json").write_text(
            json.dumps({"parameters": {"conv_dims": 3}})
        )
        out = read_thresholds(tmp_path)
        assert out == {}

    def test_fallback_flat_json(self, tmp_path):
        # No training_config.json — should still pick up from flat JSON.
        (tmp_path / "exp.json").write_text(
            json.dumps({"prob_thresh": 0.45, "nms_thresh": 0.4})
        )
        out = read_thresholds(tmp_path)
        assert out == {"prob_thresh": 0.45, "nms_thresh": 0.4}
