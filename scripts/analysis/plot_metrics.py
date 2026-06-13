"""Plot Lightning ``metrics.csv`` for a single model or a whole tree.

Three entry points:

* :func:`plot_csv_files_interactive` — one model. Optionally takes a
  ``backup_directory``: if a SLURM job timed out and was re-launched,
  the pre-restart ``metrics.csv`` lives under that backup folder; the
  function concatenates ``backup`` → ``current`` so the curves are
  continuous. **Failsafe:** if the backup folder is absent the function
  silently treats the run as a single-shot training and plots just the
  one ``metrics.csv``.
* :func:`plot_metrics_tree` — top-level directory whose sub-folders are
  each a trained model with its own ``metrics.csv`` (+ optional
  ``backup/``). Writes per-model grids, a combined per-metric overlay
  with every model on the same axes, and a per-metric box-plot showing
  mean / variance of each model's training distribution.

Even though the docstring of upstream's
``lightning_kietzmannlab.metrics.plot_npz_files_interactive`` mentions
``.npz``, the on-disk artefact in this workspace is the CSVLogger's
``metrics.csv``; we standardise on that.
"""

# %%
from __future__ import annotations

import itertools
from pathlib import Path
from collections.abc import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("agg")

# Mirror upstream's bokeh.palettes.Category10[10] without adding a
# bokeh dependency — matplotlib's 'tab10' is the same palette.
_PALETTE = plt.get_cmap("tab10").colors


# %% ─── single-CSV loader (with optional backup concatenation) ──────
def _find_metrics_csv(directory: Path) -> Path | None:
    """Locate ``metrics.csv`` directly under ``directory`` or under any
    Lightning ``version_<n>/`` subfolder. Returns ``None`` if nothing
    matches — callers decide whether that's fatal."""
    direct = directory / "metrics.csv"
    if direct.is_file():
        return direct
    nested = sorted(directory.glob("**/metrics.csv"))
    return nested[0] if nested else None


def _load_metrics(
    csv_directory: Path,
    backup_directory: Path | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Read the model's ``metrics.csv`` and, when present, prepend the
    backup CSV so the resumed run plots as one continuous curve.

    Returns ``(df, sources)`` where ``sources`` lists the CSV paths
    that actually contributed — useful for the figure title.

    Concatenation order is ``backup`` → ``current``: when SLURM kills a
    run mid-train we save the partial ``metrics.csv`` into the backup
    folder before the resume overwrites it. If ``backup_directory`` is
    ``None`` *or* doesn't exist on disk, we silently treat the run as
    single-shot and return only the current CSV's frame.
    """
    csv_directory = Path(csv_directory)
    main_csv = _find_metrics_csv(csv_directory)
    if main_csv is None:
        raise FileNotFoundError(f"No metrics.csv under {csv_directory}")

    sources: list[Path] = []
    frames: list[pd.DataFrame] = []

    if backup_directory is not None:
        backup_directory = Path(backup_directory)
        if backup_directory.is_dir():
            backup_csv = _find_metrics_csv(backup_directory)
            if backup_csv is not None:
                frames.append(pd.read_csv(backup_csv))
                sources.append(backup_csv)
            else:
                print(
                    f"   backup folder {backup_directory} exists but has "
                    f"no metrics.csv — treating as single-shot"
                )
        # else: failsafe — backup folder missing → single-shot run.

    frames.append(pd.read_csv(main_csv))
    sources.append(main_csv)

    if len(frames) == 1:
        return frames[0], sources

    # When the resumed logger restarts ``epoch`` / ``step`` at 0, shift
    # the new frame so the concatenated axis is monotonic. We only
    # shift if the new frame's first epoch is <= the backup's last.
    head, tail = frames
    for col in ("epoch", "step"):
        if col in head.columns and col in tail.columns:
            head_last = head[col].dropna()
            tail_first = tail[col].dropna()
            if len(head_last) and len(tail_first):
                head_max = float(head_last.max())
                tail_min = float(tail_first.min())
                if tail_min <= head_max:
                    tail = tail.copy()
                    tail[col] = tail[col] + (head_max - tail_min + 1)
    df = pd.concat([head, tail], ignore_index=True)
    return df, sources


def _metric_columns(
    df: pd.DataFrame,
    unwanted_substrings: Iterable[str],
) -> tuple[str, list[str]]:
    """Pick the X-axis column (epoch ▸ step ▸ first column) and the set
    of metric columns to plot."""
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
    return x_col, grouped_keys


# %% ─── public: one model ──────────────────────────────────────────
def plot_csv_files_interactive(
    csv_directory,
    backup_directory=None,
    unwanted_substrings=("gpu", "memory"),
    page_output_dir="metrics",
    save_plots=True,
    show_plots=False,
):
    """Plot every metric column of one model's ``metrics.csv`` in a
    4-column grid — same style as
    ``lightning_kietzmannlab.metrics.plot_npz_files_interactive``.

    Args:
        csv_directory: Folder holding the current ``metrics.csv`` (or
            a Lightning ``version_<n>/`` subfolder of one).
        backup_directory: Optional folder holding the *previous*
            ``metrics.csv`` from a SLURM-killed run. When present and
            non-empty it is prepended to produce one continuous curve.
            **Failsafe:** when missing on disk we just plot the single
            current CSV — passing a backup path always is safe.
        unwanted_substrings: Drop metric columns whose name contains
            any of these (defaults skip the per-step GPU / memory
            telemetry that swamps the grid).
        page_output_dir: Where to write ``metrics_all_in_one.png``.
        save_plots / show_plots: as named.

    Returns:
        The ``Path`` of the saved figure when ``save_plots`` is True,
        else ``None``.
    """
    csv_directory = Path(csv_directory)
    Path(page_output_dir).mkdir(parents=True, exist_ok=True)

    df, sources = _load_metrics(csv_directory, backup_directory)
    if df.empty:
        print(f"metrics.csv is empty under {csv_directory}")
        return None

    x_col, grouped_keys = _metric_columns(df, unwanted_substrings)
    if not grouped_keys:
        print(f"No plottable metric columns under {csv_directory}")
        return None

    colors = itertools.cycle(_PALETTE)
    n_cols = 4
    n_plots = len(grouped_keys)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = np.atleast_2d(axes).flatten()

    i = -1
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
        ax.set_title(key)
        ax.set_xlabel(x_col.capitalize())
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    src_label = " + ".join(str(s.relative_to(s.parents[1])) for s in sources)
    fig.suptitle(f"{csv_directory.name}  ({src_label})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = None
    if save_plots:
        out_path = Path(page_output_dir) / "metrics_all_in_one.png"
        fig.savefig(out_path, dpi=300)
        print(f"Saved all-in-one plot to: {out_path}")
    if show_plots:
        plt.show()
    plt.close(fig)
    return out_path


# %% ─── public: tree of models ─────────────────────────────────────
def _discover_models(
    root: Path,
    backup_subfolder: str,
) -> list[tuple[Path, Path | None]]:
    """Return ``(model_dir, backup_dir_or_None)`` for every sub-folder
    of ``root`` that has a ``metrics.csv`` somewhere underneath.

    ``backup_dir`` is ``<model_dir>/<backup_subfolder>`` when that
    folder exists, else ``None`` (so the per-model call goes through
    the failsafe path)."""
    models: list[tuple[Path, Path | None]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if _find_metrics_csv(child) is None:
            continue
        backup = child / backup_subfolder
        models.append((child, backup if backup.is_dir() else None))
    return models


def plot_metrics_tree(
    root_directory,
    output_dir=None,
    backup_subfolder="backup",
    unwanted_substrings=("gpu", "memory"),
    per_model_subdir="per_model",
):
    """Plot ``metrics.csv`` for every model sub-folder of
    ``root_directory``.

    Layout assumed::

        root_directory/
          model_A/
            metrics.csv
            backup/            # optional — concatenated when present
              metrics.csv
          model_B/
            metrics.csv
          ...

    Three artefact families are written under ``output_dir`` (defaults
    to ``<root_directory.name>_summary/`` next to the script):

    * ``per_model/<model_name>/metrics_all_in_one.png`` — the same
      4-column grid as the single-model function.
    * ``combined_overlay_<metric>.png`` — one figure per metric column,
      every model on the same axes; lets you eyeball convergence
      differences between runs.
    * ``boxplot_<metric>.png`` — box-plot per metric showing each
      model's distribution over the whole training (mean marker +
      median + IQR + whiskers). Captures stability, not just final
      value.

    Args:
        root_directory: parent of the model folders.
        output_dir: where to write all artefacts (default
            ``<root.name>_summary/`` next to the CWD).
        backup_subfolder: name of the in-model backup folder (default
            ``backup``). Missing folders silently fall through to the
            single-shot path.
        unwanted_substrings: dropped metric-name substrings (same
            default as the single-model function).
        per_model_subdir: subfolder name for per-model grids.
    """
    root = Path(root_directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Tree root not found: {root}")
    output_dir = Path(output_dir) if output_dir else Path(f"{root.name}_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    models = _discover_models(root, backup_subfolder)
    if not models:
        print(f"No model sub-folders with metrics.csv under {root}")
        return
    print(f"Tree root:   {root}")
    print(f"Models:      {len(models)}")
    print(f"Output dir:  {output_dir}")

    # ── per-model grids + collect frames for downstream summaries ──
    model_frames: dict[str, pd.DataFrame] = {}
    model_xcol: dict[str, str] = {}
    for model_dir, backup_dir in models:
        name = model_dir.name
        print(f"\n• {name}  (backup={'yes' if backup_dir else 'no'})")
        per_model_out = output_dir / per_model_subdir / name
        plot_csv_files_interactive(
            model_dir,
            backup_directory=backup_dir,
            unwanted_substrings=unwanted_substrings,
            page_output_dir=per_model_out,
            save_plots=True,
            show_plots=False,
        )
        df, _ = _load_metrics(model_dir, backup_dir)
        x_col, _ = _metric_columns(df, unwanted_substrings)
        model_frames[name] = df
        model_xcol[name] = x_col

    # Union of metric columns across all models (so e.g. a metric that
    # only some models log still appears in the overlay).
    all_metrics: list[str] = []
    seen = set()
    for name, df in model_frames.items():
        _, keys = _metric_columns(df, unwanted_substrings)
        for k in keys:
            if k not in seen:
                seen.add(k)
                all_metrics.append(k)

    color_map = {
        name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(model_frames)
    }

    # ── combined overlay (one figure per metric, all models stacked) ──
    overlay_dir = output_dir / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for metric in all_metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = 0
        for name, df in model_frames.items():
            if metric not in df.columns:
                continue
            x_col = model_xcol[name]
            sub = df[[x_col, metric]].dropna().sort_values(x_col)
            if sub.empty:
                continue
            ax.plot(
                sub[x_col].to_numpy(),
                sub[metric].to_numpy(),
                label=name,
                color=color_map[name],
                linewidth=1.2,
                alpha=0.9,
            )
            plotted += 1
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(f"{metric} — all models")
        ax.set_xlabel("epoch / step")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best", framealpha=0.7)
        fig.tight_layout()
        safe = metric.replace("/", "_").replace(" ", "_")
        out = overlay_dir / f"combined_overlay_{safe}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"  overlay  → {out.name}")

    # ── boxplot summary (mean / variance per model per metric) ────
    box_dir = output_dir / "boxplots"
    box_dir.mkdir(parents=True, exist_ok=True)
    for metric in all_metrics:
        data, labels, colors = [], [], []
        for name, df in model_frames.items():
            if metric not in df.columns:
                continue
            vals = df[metric].dropna().to_numpy()
            if vals.size == 0:
                continue
            data.append(vals)
            labels.append(name)
            colors.append(color_map[name])
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(data) + 3), 5))
        bp = ax.boxplot(
            data,
            labels=labels,
            showmeans=True,
            meanline=False,
            patch_artist=True,
            widths=0.6,
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
            patch.set_edgecolor(c)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.2)
        ax.set_title(f"{metric} — distribution over training")
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        fig.tight_layout()
        safe = metric.replace("/", "_").replace(" ", "_")
        out = box_dir / f"boxplot_{safe}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"  boxplot  → {out.name}")


# %% ─── entry point (edit paths per cluster) ───────────────────────
if __name__ == "__main__":
    # Single-model mode — uncomment to use:
    # csv_directory_current = Path("/mnt/jean-zay/models_stardist_pytorch/")
    # backup_directory = csv_directory_current / "backup"
    # plot_csv_files_interactive(
    #     csv_directory_current,
    #     backup_directory=backup_directory,
    #     save_plots=True,
    #     page_output_dir=csv_directory_current.name,
    # )

    # Tree mode — point at the sweep root with one sub-folder per run.
    tree_root = Path("/mnt/jean-zay/models_stardist_pytorch_sweep/")
    plot_metrics_tree(
        tree_root,
        output_dir=Path(__file__).resolve().parent / f"{tree_root.name}_summary",
    )

# %%
