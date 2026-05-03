"""Tests for vollseg.hub — registry lookup + ensure_model skip-when-present."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from vollseg import XENOPUS_MODELS, ensure_cellpose_checkpoint, ensure_model, hf_repo_for


class TestRegistry:
    def test_known_model(self):
        # The fixed Xenopus model names should always resolve.
        for name in (
            "membrane_edge_enhancement",
            "nuclei_xenopus_mari",
            "mem_mneongreen",
        ):
            assert hf_repo_for(name) is not None
            assert hf_repo_for(name).startswith("KapoorLabs-Copenhagen/")

    def test_unknown_model(self):
        assert hf_repo_for("nonsense_model_name_xyz") is None


class TestEnsureModel:
    def test_returns_existing_dir_without_download(self, tmp_path):
        target = tmp_path / "fake_model"
        target.mkdir()
        (target / "weights.h5").write_bytes(b"placeholder")
        # Even though "fake_model" isn't in the registry, since it exists
        # locally we never need to look it up — no exception, no HF call.
        with patch("huggingface_hub.snapshot_download") as snap:
            out = ensure_model(tmp_path, "fake_model")
        assert out == target
        snap.assert_not_called()

    def test_returns_existing_file_without_download(self, tmp_path):
        # Legacy CellPose layout: a single file at the configured path.
        target = tmp_path / "cellpose_ckpt"
        target.write_bytes(b"checkpoint")
        with patch("huggingface_hub.snapshot_download") as snap:
            out = ensure_model(tmp_path, "cellpose_ckpt")
        assert out == target
        snap.assert_not_called()

    def test_missing_unregistered_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ensure_model(tmp_path, "nonsense_unregistered_name")

    def test_missing_registered_calls_snapshot(self, tmp_path):
        # The download is mocked; we just verify the right repo id is used.
        with patch("huggingface_hub.snapshot_download") as snap:
            # Make the dir appear non-empty after the mocked download so
            # the function returns cleanly.
            def fake_download(*args, **kwargs):
                Path(kwargs["local_dir"]).joinpath("dummy.h5").write_bytes(b"x")
                return kwargs["local_dir"]
            snap.side_effect = fake_download

            out = ensure_model(tmp_path, "membrane_edge_enhancement")

        snap.assert_called_once()
        assert snap.call_args.kwargs["repo_id"] == XENOPUS_MODELS["membrane_edge_enhancement"]
        assert out.name == "membrane_edge_enhancement"


class TestEnsureCellposeCheckpoint:
    def test_returns_legacy_file_path(self, tmp_path):
        # Legacy: single file directly at model_dir/model_name.
        f = tmp_path / "mem_mneongreen"
        f.write_bytes(b"fake-cellpose-ckpt")
        out = ensure_cellpose_checkpoint(tmp_path, "mem_mneongreen")
        assert out == f
