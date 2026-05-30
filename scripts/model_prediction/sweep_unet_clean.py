"""Trim every ``unet_sweep_*/`` directory under ``sweep_root`` down to
its final-epoch checkpoint.

Same logic as ``scripts/clean_checkpoints.py`` but applied across the
whole sweep instead of a single folder. Use after ``sweep_unet_analyze.py``
when you've decided which sweep configuration won and just want the
last (=best, modulo overfit) checkpoint of every run on disk.

Edit the paths block below, then::

    # Dry run first to confirm what would be deleted.
    python sweep_unet_clean.py

Set ``dry_run = False`` to actually delete.
"""

# %%
from __future__ import annotations

import sys
from pathlib import Path

# Reuse the per-folder cleaner from ``scripts/clean_checkpoints.py``
# (two levels up from this file). Keeps a single source of truth for
# the keep_first / keep_middle / keep_last selection.
_HERE = Path(__file__).resolve().parent
_SCRIPTS_ROOT = _HERE.parent
sys.path.insert(0, str(_SCRIPTS_ROOT))
from clean_checkpoints import clean_ckpt_files  # noqa: E402


# %% ─── paths (edit per cluster) ─────────────────────────────────────
sweep_root = Path("/home/debian/jean-zay/models_unet_pytorch_sweep")

# Folder-name pattern produced by slurm_sweep_unet_jeanzay.sh.
dir_glob = "unet_sweep_*"

# Mirror clean_checkpoints.py defaults: keep only the final-epoch ckpt.
keep_n_first = 0
keep_n_middle = 0
keep_n_last = 1

# Set to ``False`` to actually delete; True does a dry-run.
dry_run = False


# %% ─── walk the sweep ──────────────────────────────────────────────
sweep_dirs = sorted(p for p in sweep_root.glob(dir_glob) if p.is_dir())
print(f"Sweep root: {sweep_root}")
print(f"Found {len(sweep_dirs)} sweep directories")
print(
    f"Keep policy: first={keep_n_first} middle={keep_n_middle} "
    f"last={keep_n_last}  dry_run={dry_run}"
)
print()

for i, d in enumerate(sweep_dirs):
    print(f"[{i + 1}/{len(sweep_dirs)}] {d.name}")
    clean_ckpt_files(
        str(d),
        keep_n_first=keep_n_first,
        keep_n_middle=keep_n_middle,
        keep_n_last=keep_n_last,
        dry_run=dry_run,
    )
    print()

print("Re-run with ``dry_run = False`` to actually delete." if dry_run else "Done.")
