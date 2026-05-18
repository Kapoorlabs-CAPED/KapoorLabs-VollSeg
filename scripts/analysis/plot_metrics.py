"""Plot Lightning-CSVLogger metrics in the same grid style as
``lightning_kietzmannlab.metrics.plot_npz_files_interactive`` — except
the metrics live in ``metrics.csv`` (one row per epoch, one column per
metric) rather than an NPZ of ``{steps, values}`` dicts.

Usage::

    # Most common — one log folder, latest training run.
    python plot_metrics.py /path/to/log_path

    # Or point at the CSV directly.
    python plot_metrics.py /path/to/log_path/metrics.csv

    # Optional output dir + behavior switches.
    python plot_metrics.py /path/to/log_path \\
        --out-dir /path/to/plots --save --no-show

Each non-x column gets one subplot in a 4-column grid: line + light
scatter, Category10 colors, key as title. The all-in-one PNG is
written to ``<out_dir>/metrics_all_in_one.png`` when ``--save`` is
on, mirroring the upstream behavior.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Mirror upstream's bokeh.palettes.Category10[10] without adding a
# bokeh dependency — matplotlib's 'tab10' is the same palette.
_PALETTE = plt.get_cmap("tab10").colors


def _resolve_csv(path: Path) -> Path:
    """Accept a CSV path *or* a Lightning log folder containing ``metrics.csv``."""
    if path.is_file():
        return path
    csvs = sorted(path.glob("metrics.csv")) + sorted(path.glob("**/metrics.csv"))
    if not csvs:
        raise FileNotFoundError(f"No metrics.csv under {path}")
    # The earliest match (top-level if present) wins.
    return csvs[0]


def _pick_x_axis(df: pd.DataFrame) -> str:
    """Prefer the 'epoch' column if Lightning wrote one (and it's monotonic);
    otherwise fall back to 'step'."""
    if "epoch" in df.columns and df["epoch"].notna().any():
        return "epoch"
    if "step" in df.columns:
        return "step"
    return df.columns[0]


def plot_metrics_csv(
    csv_path: Path,
    *,
    out_dir: Path,
    unwanted_substrings: tuple[str, ...] = ("gpu", "memory"),
    save: bool = True,
    show: bool = True,
) -> Path | None:
    """Read ``metrics.csv`` and emit one subplot per metric column.

    Returns the PNG path when ``save=True``, else ``None``.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"metrics.csv is empty: {csv_path}")
        return None

    x_col = _pick_x_axis(df)
    metric_cols = [
        c
        for c in df.columns
        if c not in (x_col, "epoch", "step")
        and not any(sub in c for sub in unwanted_substrings)
    ]
    if not metric_cols:
        print("No plottable metric columns in", csv_path)
        return None

    # Same grid shape as upstream: 4 columns, ceil(n / 4) rows.
    colors = itertools.cycle(_PALETTE)
    n_cols = 4
    n_plots = len(metric_cols)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = np.atleast_2d(axes).flatten()

    last_i = -1
    for i, key in enumerate(metric_cols):
        sub = df[[x_col, key]].dropna().sort_values(x_col)
        if sub.empty:
            continue
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
        ax.set_title(key)
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        last_i = i

    # Hide unused subplots in the grid (same as upstream).
    for j in range(last_i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = None
    if save:
        png_path = out_dir / "metrics_all_in_one.png"
        fig.savefig(png_path, dpi=300)
        print(f"Saved all-in-one plot to: {png_path}")
    if show:
        plt.show()
    plt.close(fig)
    return png_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot a Lightning CSVLogger metrics.csv in the "
        "lightning_kietzmannlab.metrics grid style.",
    )
    p.add_argument(
        "path",
        type=Path,
        help="Path to a metrics.csv OR the log folder that contains one.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write metrics_all_in_one.png "
        "(defaults to the CSV's parent folder).",
    )
    p.add_argument("--no-save", action="store_true", help="Don't write the PNG.")
    p.add_argument("--no-show", action="store_true", help="Don't pop the window.")
    p.add_argument(
        "--unwanted",
        nargs="*",
        default=["gpu", "memory"],
        help="Skip metric columns whose name contains any of these substrings.",
    )
    args = p.parse_args()

    csv_path = _resolve_csv(args.path)
    out_dir = args.out_dir or csv_path.parent
    plot_metrics_csv(
        csv_path,
        out_dir=out_dir,
        unwanted_substrings=tuple(args.unwanted),
        save=not args.no_save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
