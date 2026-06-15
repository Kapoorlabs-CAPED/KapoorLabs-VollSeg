"""Plot Lightning ``metrics.csv`` for a single U-Net model or a sweep tree.

Sibling of :mod:`plot_metrics` — same two entry points
(:func:`plot_csv_files_interactive`, :func:`plot_metrics_tree`), same
backup-folder failsafe, same overlay + boxplot artefacts. The only
difference is the default sweep root at the bottom points at the U-Net
sweep instead of the StarDist one.
"""

# %%
from __future__ import annotations

from pathlib import Path

from plot_metrics import plot_csv_files_interactive, plot_metrics_tree

__all__ = ["plot_csv_files_interactive", "plot_metrics_tree"]


# %% ─── entry point (edit paths per cluster) ───────────────────────
if __name__ == "__main__":
    # Single-model mode — uncomment to use:
    # csv_directory_current = Path("/mnt/jean-zay/models_unet_pytorch/")
    # backup_directory = csv_directory_current / "backup"
    # plot_csv_files_interactive(
    #     csv_directory_current,
    #     backup_directory=backup_directory,
    #     save_plots=True,
    #     page_output_dir=csv_directory_current.name,
    # )

    # Tree mode — point at the U-Net sweep root with one sub-folder per run.
    tree_root = Path("/mnt/jean-zay/models_unet_pytorch_sweep/")
    plot_metrics_tree(
        tree_root,
        output_dir=Path(__file__).resolve().parent / f"{tree_root.name}_summary",
    )

# %%
