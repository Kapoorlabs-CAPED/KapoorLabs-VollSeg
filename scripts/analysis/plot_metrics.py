# %%
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Mirror upstream's bokeh.palettes.Category10[10] without adding a
# bokeh dependency — matplotlib's 'tab10' is the same palette.
_PALETTE = plt.get_cmap("tab10").colors


def plot_csv_files_interactive(
    csv_directory,
    unwanted_substrings=("gpu", "memory"),
    page_output_dir="metrics",
    save_plots=False,
    show_plots=True,
):
    """Read ``metrics.csv`` from ``csv_directory`` and plot every metric
    column in a 4-column grid — same style as
    ``lightning_kietzmannlab.metrics.plot_npz_files_interactive``."""
    csv_directory = Path(csv_directory)
    csv_path = csv_directory / "metrics.csv"
    if not csv_path.is_file():
        # Fall back to a recursive search in case Lightning wrote into a
        # version_X subfolder.
        csvs = sorted(csv_directory.glob("**/metrics.csv"))
        if not csvs:
            raise FileNotFoundError(f"No metrics.csv under {csv_directory}")
        csv_path = csvs[0]

    Path(page_output_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"metrics.csv is empty: {csv_path}")
        return

    # X-axis: prefer 'epoch', else 'step'.
    if "epoch" in df.columns and df["epoch"].notna().any():
        x_col = "epoch"
    elif "step" in df.columns:
        x_col = "step"
    else:
        x_col = df.columns[0]

    grouped_keys = [
        c
        for c in df.columns
        if c not in (x_col, "epoch", "step")
        and not any(sub in c for sub in unwanted_substrings)
    ]
    if not grouped_keys:
        print("No plottable metric columns in", csv_path)
        return

    # Same grid shape as upstream: 4 columns, ceil(n / 4) rows.
    colors = itertools.cycle(_PALETTE)
    n_cols = 4
    n_plots = len(grouped_keys)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = np.atleast_2d(axes).flatten()

    for i, key in enumerate(grouped_keys):
        sub = df[[x_col, key]].dropna().sort_values(x_col)
        color = next(colors)
        ax = axes[i]
        ax.plot(sub[x_col].to_numpy(), sub[key].to_numpy(), label=key, color=color)
        ax.scatter(
            sub[x_col].to_numpy(),
            sub[key].to_numpy(),
            s=4,
            color=color,
            alpha=0.3,
        )
        ax.set_title(f"{key}")
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_plots:
        output_path = Path(page_output_dir) / "metrics_all_in_one.png"
        fig.savefig(output_path, dpi=300)
        print(f"Saved all-in-one plot to: {output_path}")
        if show_plots:
            plt.show()

    if show_plots:
        plt.show()


# %%
csv_directory_current = Path(
    "/lustre/fsn1/projects/rech/jsy/uzj81mi/models_stardist_pytorch/"
)
# csv_directory_backup = Path("/lustre/fsn1/projects/rech/jsy/uzj81mi/models_stardist_pytorch/backup/")


# %%

plot_csv_files_interactive(
    csv_directory_current,
    save_plots=True,
    page_output_dir=csv_directory_current.stem,
)

# %%
