"""One-shot helper: upload the freshly-trained PyTorch StarDist / U-Net
folders to public HuggingFace model repos under the ``KapoorLabs`` org.

Sibling of :mod:`scripts/legacy_segmentation_workflow/_upload_models_to_hf.py`
(which migrated the legacy keras Xenopus zoo); this one targets the
new PyTorch-Lightning checkpoints written by ``StarDistTrainer`` /
``UNetTrainer`` (flat layout: ``last.ckpt`` + ``rays.npy`` +
``training_config.json`` next to each other).

Usage::

    # 1. Put the HF token in a .env file next to this script:
    #    HF_TOKEN=hf_xxx...

    # Standard layout — folders named after the MODELS keys live
    # directly under --source-root:
    python scripts/upload_pytorch_models_to_hf.py \\
        --source-root /home/debian/jean-zay [--dry-run]

    # Replace an existing HF model with the best of a sweep — point
    # --source-folder at the winning sweep directory and --only at
    # the repo you want to overwrite. Use --replace to first wipe the
    # remote repo so stale per-epoch checkpoints from the previous
    # upload don't linger:
    python scripts/upload_pytorch_models_to_hf.py \\
        --source-folder /home/debian/jean-zay/models_unet_pytorch_sweep/unet_sweep_adam_lr1p0e-3_noscheduler \\
        --only models_unet_pytorch \\
        --replace

Mapping is held in :data:`MODELS` — extend it when you train more.
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
- ``rays.npy`` (StarDist only) — the rays array used at training time

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
