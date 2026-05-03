"""One-shot helper: publish each Xenopus model folder as its own public HF model repo.

Use this once to migrate the existing private dataset
``KapoorLabs-Copenhagen/Xenopus_Models`` into individual public model
repos. After that, ``vollseg.hub.ensure_model`` auto-downloads them.

Usage::

    huggingface-cli login                                # one-time
    python scripts/_upload_models_to_hf.py \\
        --source /path/to/Xenopus_Models/Mari_Models \\
        [--dry-run]

Mapping is read from :data:`vollseg.hub.XENOPUS_MODELS`. The local source
layout is expected to be:

    {source}/CARE/membrane_edge_enhancement/...
    {source}/Unet3D/unet_nuclei_xenopus_mari/...
    {source}/Unet3D/unet_membrane_xenopus_mari/...
    {source}/MASKUNET/unet_roi_nuclei_xenopus/...
    {source}/StarDist3D/nuclei_xenopus_mari/...
    {source}/StarDist3D/membrane_xenopus_mari/...
    {source}/CellPose/mem_mneongreen      (single file)

Each model becomes a public repo under ``KapoorLabs-Copenhagen/`` with
the matching name from :data:`vollseg.hub.XENOPUS_MODELS`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

from vollseg.hub import XENOPUS_MODELS


# Maps each registered model_name → the relative subfolder under --source
# where its weights currently live. Add entries here when XENOPUS_MODELS
# grows and the source layout has its own subfolder for it.
SOURCE_LAYOUT: dict[str, Tuple[str, str]] = {
    "membrane_edge_enhancement":  ("CARE",       "membrane_edge_enhancement"),
    "unet_nuclei_xenopus_mari":   ("Unet3D",     "unet_nuclei_xenopus_mari"),
    "unet_membrane_xenopus_mari": ("Unet3D",     "unet_membrane_xenopus_mari"),
    "unet_roi_nuclei_xenopus":    ("MASKUNET",   "unet_roi_nuclei_xenopus"),
    "nuclei_xenopus_mari":        ("StarDist3D", "nuclei_xenopus_mari"),
    "membrane_xenopus_mari":      ("StarDist3D", "membrane_xenopus_mari"),
    "mem_mneongreen":             ("CellPose",   "mem_mneongreen"),
}


def _readme_for(model_name: str, repo_id: str) -> str:
    return f"""---
license: bsd-3-clause
tags:
- biology
- microscopy
- segmentation
- xenopus
- vollseg
---

# {model_name}

Pretrained model from the **VollSeg** Xenopus zoo. Originally trained for the
Tolonen / Sedzinski Xenopus tissue segmentation pipeline; now distributed
under the BSD-3-Clause license for general use.

## Loading

```python
from vollseg import ensure_model

# Downloads weights from {repo_id} into <model_dir>/{model_name}/
ensure_model("./Mari_Models/<subfolder>", "{model_name}")
```

See https://github.com/Kapoorlabs-CAPED/KapoorLabs-VollSeg for the full
segmentation pipeline.
"""


def upload_one(
    api,
    model_name: str,
    source_root: Path,
    *,
    dry_run: bool,
) -> bool:
    if model_name not in XENOPUS_MODELS:
        print(f"  skip: {model_name} not in vollseg.hub.XENOPUS_MODELS")
        return False
    if model_name not in SOURCE_LAYOUT:
        print(f"  skip: {model_name} has no SOURCE_LAYOUT entry")
        return False

    repo_id = XENOPUS_MODELS[model_name]
    subdir, name = SOURCE_LAYOUT[model_name]
    src = source_root / subdir / name
    if not src.exists():
        print(f"  skip: {model_name} — source {src} not found")
        return False

    print(f"  → {repo_id}  (from {src})")
    if dry_run:
        return True

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    readme_text = _readme_for(model_name, repo_id)
    commit = f"Initial upload of {model_name}"

    if src.is_dir():
        # Folder layout (csbdeep / stardist / maskunet / care).
        readme = src / "README.md"
        if not readme.exists():
            readme.write_text(readme_text)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(src),
            repo_type="model",
            commit_message=commit,
        )
    else:
        # Single-file layout (cellpose checkpoint). Upload the file at the
        # repo root, then add a generated README via an in-memory upload.
        from io import BytesIO

        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=src.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit,
        )
        api.upload_file(
            path_or_fileobj=BytesIO(readme_text.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="Path to the local Mari_Models root containing CARE/, Unet3D/, etc.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be uploaded; don't touch HF.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Restrict to these model names (default: all registered).")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"--source {args.source} does not exist", file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import HfApi
    api = HfApi()

    targets = args.only or list(XENOPUS_MODELS.keys())
    print(f"{'DRY-RUN: ' if args.dry_run else ''}uploading {len(targets)} model(s)")
    ok = 0
    for name in targets:
        print(f"\n[{name}]")
        if upload_one(api, name, args.source, dry_run=args.dry_run):
            ok += 1
    print(f"\nDone: {ok}/{len(targets)} succeeded")


if __name__ == "__main__":
    main()
