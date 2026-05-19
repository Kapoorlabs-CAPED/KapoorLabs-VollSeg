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
    # 2. python scripts/upload_pytorch_models_to_hf.py \\
    #        --source-root /home/debian/jean-zay [--dry-run]

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
) -> bool:
    if folder_name not in MODELS:
        print(f"  skip: {folder_name} not in MODELS")
        return False

    repo_id = MODELS[folder_name]
    src = source_root / folder_name
    if not src.is_dir():
        print(f"  skip: {folder_name} — source {src} not found")
        return False

    print(f"  → {repo_id}  (from {src})")
    if dry_run:
        return True

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    readme_text = _readme_for(folder_name, repo_id)
    commit = f"Initial upload of {folder_name}"

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

    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(src),
        repo_type="model",
        commit_message=commit,
    )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Path containing the model folders, e.g. /home/debian/jean-zay",
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

    if not args.source_root.exists():
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
        if upload_one(api, name, args.source_root, dry_run=args.dry_run):
            ok += 1
    print(f"\nDone: {ok}/{len(targets)} succeeded")


if __name__ == "__main__":
    main()
