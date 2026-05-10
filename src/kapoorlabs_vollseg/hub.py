"""HuggingFace Hub auto-download for the Xenopus model zoo.

The pretrained Xenopus models live as public repos under
``KapoorLabs-Copenhagen/`` on the HuggingFace Hub. This module:

1. Holds the canonical mapping between the **local model name** used in
   the hydra configs (e.g. ``"membrane_edge_enhancement"``) and the
   **HF repo id** (e.g.
   ``"KapoorLabs-Copenhagen/xenopus-care-membrane-edge-enhancement"``).
2. Exposes :func:`ensure_model` — given ``(model_dir, model_name)``,
   guarantees the directory ``model_dir / model_name`` exists locally,
   downloading it from HuggingFace if it doesn't.

Scripts call :func:`ensure_model` for each configured model before
constructing backbones, so users with a fresh checkout can run anything
without first manually fetching weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


# --- Model registry -------------------------------------------------------
# Each entry maps the local "model name" (the value of ``*_model_name`` in
# the hydra YAMLs) to the HuggingFace repo id that holds its weights.
#
# To add a model: upload it as a public model repo under
# KapoorLabs-Copenhagen/ (see ``scripts/_upload_models_to_hf.py``), then
# add the entry here.

XENOPUS_MODELS: dict[str, str] = {
    # CARE denoising
    "membrane_edge_enhancement": "KapoorLabs-Copenhagen/xenopus-care-membrane-edge-enhancement",
    # Unet3D — semantic segmentation
    "unet_nuclei_xenopus_mari": "KapoorLabs-Copenhagen/xenopus-unet3d-nuclei-mari",
    "unet_membrane_xenopus_mari": "KapoorLabs-Copenhagen/xenopus-unet3d-membrane-mari",
    # MASKUNET — ROI mask
    "unet_roi_nuclei_xenopus": "KapoorLabs-Copenhagen/xenopus-maskunet-roi-nuclei",
    # StarDist3D — instance segmentation
    "nuclei_xenopus_mari": "KapoorLabs-Copenhagen/xenopus-stardist3d-nuclei-mari",
    "membrane_xenopus_mari": "KapoorLabs-Copenhagen/xenopus-stardist3d-membrane-mari",
    # CellPose — only mem_mneongreen is supported (cellpose3D ignored).
    "mem_mneongreen": "KapoorLabs-Copenhagen/xenopus-cellpose-mem-mneongreen",
}


def hf_repo_for(model_name: str) -> Optional[str]:
    """Return the HF repo id for a registered model name, or None if unknown."""
    return XENOPUS_MODELS.get(model_name)


def ensure_model(
    model_dir: Union[str, Path],
    model_name: str,
    *,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    allow_patterns: Optional[list[str]] = None,
    force_download: bool = False,
) -> Path:
    """Make sure ``model_dir / model_name`` exists locally; fetch from HF if not.

    Parameters
    ----------
    model_dir
        Parent directory under which the model folder lives.
    model_name
        Folder name beneath ``model_dir``.
    repo_id
        Optional override of the registered HF repo id. If ``None``, looks
        up :data:`XENOPUS_MODELS` by ``model_name``.
    revision
        Optional HF revision (branch / tag / commit) to pin to.
    allow_patterns
        If provided, only files matching these patterns are downloaded
        (e.g. ``["*.h5", "config.json"]``).
    force_download
        Re-download even if the target appears to exist.

    Returns
    -------
    pathlib.Path
        The local path to the model folder.

    Raises
    ------
    FileNotFoundError
        If the local copy is missing and no HF mapping is registered.
    """
    target = Path(model_dir) / model_name
    if not force_download and target.exists():
        # File-shaped (e.g. legacy cellpose layout): caller knows what to do.
        if target.is_file():
            return target
        if target.is_dir() and any(target.iterdir()):
            return target

    repo = repo_id or hf_repo_for(model_name)
    if repo is None:
        raise FileNotFoundError(
            f"Model '{model_name}' not present at {target} and no HF repo "
            f"id registered. Add it to kapoorlabs_vollseg.hub.XENOPUS_MODELS or pass "
            f"repo_id=... explicitly."
        )

    # Lazy import — keeps huggingface_hub optional for users who already
    # have all weights on disk.
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    print(f"[kapoorlabs_vollseg.hub] downloading {repo} → {target}")
    snapshot_download(
        repo_id=repo,
        local_dir=str(target),
        revision=revision,
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )
    return target


def ensure_cellpose_checkpoint(
    model_dir: Union[str, Path],
    model_name: str,
    *,
    repo_id: Optional[str] = None,
) -> Path:
    """Same as :func:`ensure_model` but returns the checkpoint *file* path.

    CellPose stores its trained model as a single file rather than a
    folder. Two layouts are supported transparently:

    1. **Legacy on-disk** — the checkpoint sits as a file directly at
       ``model_dir / model_name``. Returned as-is.
    2. **Auto-downloaded** — ``model_dir / model_name`` is a folder
       holding the checkpoint file (named ``model_name``) plus a
       ``README.md``. The checkpoint inside is returned.
    """
    direct = Path(model_dir) / model_name
    if direct.is_file():
        return direct

    folder = ensure_model(model_dir, model_name, repo_id=repo_id)
    inside = folder / model_name
    if inside.is_file():
        return inside

    # Fallback: pick the only non-readme file in the folder.
    files = [p for p in folder.iterdir() if p.is_file() and p.name != "README.md"]
    if len(files) == 1:
        return files[0]
    raise FileNotFoundError(
        f"Could not locate cellpose checkpoint inside {folder}; "
        f"found: {[p.name for p in files]}"
    )
