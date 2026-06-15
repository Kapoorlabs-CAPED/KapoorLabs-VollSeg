"""One-shot helper: upload the freshly-trained PyTorch StarDist / U-Net /
CARE folders to public HuggingFace model repos under the ``KapoorLabs``
org.

Sibling of :mod:`scripts/legacy_segmentation_workflow/_upload_models_to_hf.py`
(which migrated the legacy keras Xenopus zoo); this one targets the
new PyTorch-Lightning checkpoints written by ``StarDistTrainer`` /
``UNetTrainer`` / ``CARETrainer`` (flat layout: ``last.ckpt`` +
``training_config.json`` next to each other; StarDist rays are
regenerated from the JSON at load time, no ``rays.npy`` sidecar).

Usage
=====

Step 0 — put the HF token in a .env file next to this script:

    echo 'HF_TOKEN=hf_xxx...' > scripts/.env

All one-liners below assume you ``cd`` into the repo root first. Append
``--dry-run`` to any of them to print exactly what would happen without
touching HF. ``--replace`` wipes the remote repo in the same commit as
the new upload so stale per-epoch checkpoints from a prior upload don't
linger; omit it for a first-time upload.

────────────────────────────────────────────────────────────────────────
StarDist (``models_stardist_pytorch`` → ``KapoorLabs/xenopus-stardist-pytorch``)
────────────────────────────────────────────────────────────────────────

First-time upload (folder named after the MODELS key under --source-root):

    python scripts/upload_pytorch_models_to_hf.py --source-root /home/debian/jean-zay --only models_stardist_pytorch

Replace existing remote with the same standard-layout folder:

    python scripts/upload_pytorch_models_to_hf.py --source-root /home/debian/jean-zay --only models_stardist_pytorch --replace

Replace existing remote with the winner of a sweep (non-standard folder
name) — use --source-folder and exactly one --only target:

    python scripts/upload_pytorch_models_to_hf.py --source-folder /home/debian/jean-zay/models_stardist_pytorch_sweep/stardist_sweep_adam_lr1p0e-3_cosine --only models_stardist_pytorch --replace

────────────────────────────────────────────────────────────────────────
U-Net (``models_unet_pytorch`` → ``KapoorLabs/xenopus-unet-pytorch``)
────────────────────────────────────────────────────────────────────────

First-time upload:

    python scripts/upload_pytorch_models_to_hf.py --source-root /home/debian/jean-zay --only models_unet_pytorch

Replace existing remote with the same standard-layout folder:

    python scripts/upload_pytorch_models_to_hf.py --source-root /home/debian/jean-zay --only models_unet_pytorch --replace

Replace existing remote with the winner of a sweep:

    python scripts/upload_pytorch_models_to_hf.py --source-folder /home/debian/jean-zay/models_unet_pytorch_sweep/unet_sweep_adam_lr1p0e-3_noscheduler --only models_unet_pytorch --replace

────────────────────────────────────────────────────────────────────────
CARE (``models_care_pytorch`` → ``KapoorLabs/xenopus-edge-pytorch``)
────────────────────────────────────────────────────────────────────────

First-time upload:

    python scripts/upload_pytorch_models_to_hf.py --source-root /home/debian/jean-zay --only models_edge_pytorch

Replace existing remote with the same standard-layout folder:

    python scripts/upload_pytorch_models_to_hf.py --source-root /home/debian/jean-zay --only models_edge_pytorch --replace

Replace existing remote with a specific checkpoint folder (CARE has no
sweep — single trained model — but the --source-folder form still works


────────────────────────────────────────────────────────────────────────
Batch (multiple models in one invocation)
────────────────────────────────────────────────────────────────────────

First-time upload of ALL models in :data:`MODELS`:

    python scripts/upload_pytorch_models_to_hf.py --source-root /home/debian/jean-zay

First-time upload of TWO specific models:

    python scripts/upload_pytorch_models_to_hf.py --source-root /home/debian/jean-zay --only models_stardist_pytorch models_unet_pytorch

Replace ALL models from the standard layout:

    python scripts/upload_pytorch_models_to_hf.py --source-root /home/debian/jean-zay --replace

Replace TWO specific models from the standard layout:

    python scripts/upload_pytorch_models_to_hf.py --source-root /home/debian/jean-zay --only models_unet_pytorch models_edge_pytorch --replace

────────────────────────────────────────────────────────────────────────
Adding a new model kind
────────────────────────────────────────────────────────────────────────

Extend :data:`MODELS` (the ``{local_folder_name : huggingface_repo_id}``
dict) — that's the single source of truth the CLI iterates over.
"""

from __future__ import annotations

import argparse
import os
import sys
from io import BytesIO
from pathlib import Path


# {local_folder_name : huggingface_repo_id} — extend when you train more.
MODELS: dict[str, str] = {
    "models_unet_pytorch": "KapoorLabs/xenopus-unet-pytorch",
    "models_stardist_pytorch": "KapoorLabs/xenopus-stardist-pytorch",
    "models_maskunet_pytorch": "KapoorLabs/xenopus-maskunet-pytorch",
    "models_edge_pytorch": "KapoorLabs/xenopus-edge-pytorch",
}


def _readme_for(folder_name: str, repo_id: str) -> str:
    return f"""---
license: bsd-3-clause
tags:
- biology
- microscopy
- segmentation
- xenopus
- vollseg
- pytorch
- lightning
---

# {repo_id.split('/')[-1]}

PyTorch-Lightning checkpoint trained with ``kapoorlabs_vollseg``
(``{folder_name}``). Flat layout — the folder ships:

- ``last.ckpt`` (and optionally per-epoch ``<model_name>-epoch=NNN.ckpt``)
- ``training_config.json`` — Hydra parameters block, what
  ``kapoorlabs_vollseg`` reads first to rebuild the architecture
- ``<model_name>.json`` — legacy ``CareInception`` fallback config

StarDist rays are regenerated deterministically from
``(conv_dims, n_rays, anisotropy)`` in the JSON; no ``rays.npy``
sidecar is needed.

## Loading

```python
# StarDist
from kapoorlabs_vollseg import StarDistSegmenter, ensure_model
folder = ensure_model("./local_models", "{folder_name}",
                      repo_id="{repo_id}")
star = StarDistSegmenter.from_folder(folder)
labels = star.predict(volume).labels

# U-Net
from kapoorlabs_vollseg import UNetSegmenter, ensure_model
folder = ensure_model("./local_models", "{folder_name}",
                      repo_id="{repo_id}")
unet = UNetSegmenter.from_folder(folder)
labels = unet.predict(volume).labels

# CARE
from kapoorlabs_vollseg import CAREDenoiser, ensure_model
folder = ensure_model("./local_models", "{folder_name}",
                      repo_id="{repo_id}")
care = CAREDenoiser.from_folder(folder)
denoised = care.predict(volume).denoised
```

See https://github.com/Kapoorlabs-CAPED/KapoorLabs-VollSeg for the
full segmentation pipeline.
"""


def upload_one(
    api,
    folder_name: str,
    source_root: Path,
    *,
    dry_run: bool,
    source_override: Path | None = None,
    replace: bool = False,
) -> bool:
    if folder_name not in MODELS:
        print(f"  skip: {folder_name} not in MODELS")
        return False

    repo_id = MODELS[folder_name]
    src = source_override if source_override is not None else source_root / folder_name
    if not src.is_dir():
        print(f"  skip: {folder_name} — source {src} not found")
        return False

    label = f"REPLACE from {src}" if replace else f"upload from {src}"
    print(f"  → {repo_id}  ({label})")
    if dry_run:
        return True

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    readme_text = _readme_for(folder_name, repo_id)
    commit = (
        f"Replace contents with {src.name}"
        if replace
        else f"Initial upload of {folder_name}"
    )

    readme = src / "README.md"
    if not readme.exists():
        readme.write_text(readme_text)
    else:
        api.upload_file(
            path_or_fileobj=BytesIO(readme_text.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update model card",
        )

    # ``delete_patterns=["*"]`` wipes every file in the repo as part of
    # the same commit, then uploads the local folder fresh — guarantees
    # no stale per-epoch checkpoints from a prior upload linger on the
    # remote. README is preserved because we upload it (above) in the
    # same commit batch.
    upload_kwargs = {
        "repo_id": repo_id,
        "folder_path": str(src),
        "repo_type": "model",
        "commit_message": commit,
    }
    if replace:
        upload_kwargs["delete_patterns"] = ["*"]

    api.upload_folder(**upload_kwargs)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Path containing the model folders, e.g. /home/debian/jean-zay. "
        "Either this or --source-folder must be provided.",
    )
    parser.add_argument(
        "--source-folder",
        type=Path,
        default=None,
        help="Explicit source folder to upload (overrides "
        "``source_root / folder_name`` lookup). Useful for pushing the "
        "winning run of a sweep without renaming it on disk. Requires "
        "exactly one ``--only`` target.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Wipe every file in the remote repo before uploading the "
        "new folder (single commit). Pair with --source-folder to "
        "swap in the winner of a sweep cleanly.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded; don't touch HF.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Restrict to these folder names (default: all in MODELS).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(__file__).resolve().parent / ".env",
        help="dotenv file to read HF_TOKEN from "
        "(default: scripts/.env next to this script).",
    )
    args = parser.parse_args()

    # Need at least one of source-root / source-folder, and source-folder
    # is per-target (a single directory maps to a single repo), so it
    # demands exactly one --only target to avoid silently mis-mapping.
    if args.source_root is None and args.source_folder is None:
        print(
            "Either --source-root or --source-folder must be provided.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.source_folder is not None:
        if not args.source_folder.is_dir():
            print(
                f"--source-folder {args.source_folder} does not exist",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.only or len(args.only) != 1:
            print(
                "--source-folder requires exactly one --only target "
                "so the mapping to a single HF repo is unambiguous.",
                file=sys.stderr,
            )
            sys.exit(1)
    if args.source_root is not None and not args.source_root.exists():
        print(f"--source-root {args.source_root} does not exist", file=sys.stderr)
        sys.exit(1)

    from dotenv import load_dotenv

    load_dotenv(dotenv_path=args.env_file)
    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print(
            f"HF_TOKEN not found in {args.env_file} or environment",
            file=sys.stderr,
        )
        sys.exit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    targets = args.only or list(MODELS.keys())
    print(f"{'DRY-RUN: ' if args.dry_run else ''}uploading {len(targets)} model(s)")
    ok = 0
    for name in targets:
        print(f"\n[{name}]")
        if upload_one(
            api,
            name,
            args.source_root,
            dry_run=args.dry_run,
            source_override=args.source_folder,
            replace=args.replace,
        ):
            ok += 1
    print(f"\nDone: {ok}/{len(targets)} succeeded")


if __name__ == "__main__":
    main()
