"""Read an architecture config JSON out of a Lightning model folder.

Every checkpoint trained through the KapoorLabs-Lightning training
scripts is saved next to two JSON files:

- ``training_config.json``  — the full Hydra config dumped as JSON
  (``train_data_paths`` + ``parameters``). This is the canonical
  source of architecture knobs and is always preferred.
- ``{experiment_name}.json`` — a smaller per-experiment summary
  written by ``CareInception`` (``unet_depth``, ``num_channels_init``,
  ``n_tiles``, ``tile_overlap``, ``model_path``, ``model_name``).
  This is the fallback when ``training_config.json`` is missing — its
  contents are partial (no ``conv_dims`` or ``use_batch_norm``), so
  anything not present is auto-detected from the checkpoint state-dict
  by :func:`infer_arch_from_checkpoint`.

The reader returns a flat ``{kwarg: value}`` dict ready to be splat-ted
into ``from_checkpoint``; missing keys are simply absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

# Fields under ``parameters`` we forward to backbone ``from_checkpoint``.
_ARCH_FIELDS = (
    "conv_dims",
    "num_channels_init",
    "unet_depth",  # → renamed to "depth" below
    "use_batch_norm",
    "in_channels",
    "num_classes",
)

# How a parameters-key maps to the backbone kwarg name.
_RENAME = {"unet_depth": "depth"}


def _find_one(folder: Path, *names: str) -> Optional[Path]:
    """Return the first existing file from ``names``, searched in order."""
    for name in names:
        p = folder / name
        if p.is_file():
            return p
    return None


def read_training_config(folder: Union[str, Path]) -> dict[str, Any]:
    """Read architecture kwargs out of the JSON sidecars in ``folder``.

    Returns the picked-out arch fields with keys renamed to match the
    backbone ``from_checkpoint`` signature. An empty dict is returned
    when no JSON is found — callers should fall back to state-dict
    inference in that case.
    """
    folder = Path(folder)

    # Pass 1: canonical training_config.json (Hydra full dump).
    train_path = folder / "training_config.json"
    if train_path.is_file():
        with train_path.open() as fh:
            blob = json.load(fh)
        params = blob.get("parameters", {})
        return _extract(params)

    # Pass 2: fallback {experiment_name}.json — pick whichever JSON
    # isn't training_config.json; usually there's only one.
    candidates = [p for p in folder.glob("*.json") if p.name != "training_config.json"]
    if candidates:
        with candidates[0].open() as fh:
            blob = json.load(fh)
        # Fallback JSON is flat (not nested under ``parameters``).
        return _extract(blob)

    return {}


def _extract(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _ARCH_FIELDS:
        if key in params:
            out[_RENAME.get(key, key)] = params[key]
    return out


def find_checkpoint(folder: Union[str, Path]) -> Path:
    """Return the first ``.ckpt`` under ``folder`` (recursive). Errors if none."""
    folder = Path(folder)
    ckpts = sorted(folder.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt file under {folder}")
    return ckpts[0]


def find_rays(folder: Union[str, Path]) -> Optional[Path]:
    """Return ``rays.npy`` (or any ``*rays*.npy``) in ``folder``, or ``None``."""
    folder = Path(folder)
    direct = folder / "rays.npy"
    if direct.is_file():
        return direct
    for p in folder.glob("*rays*.npy"):
        return p
    return None
