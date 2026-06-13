"""Sweep analysis for the StarDist optimizer × LR × scheduler grid.

Walks every ``stardist_sweep_<opt>_lr<lr_tag>_<sched>/`` directory
under ``sweep_root`` (one per combo written by
``slurm_sweep_stardist_jeanzay.sh``), reads each model's
``metrics.csv`` (the flat CSVLogger output the StarDist Lightning
trainer writes), and produces:

1. ``sweep_summary.csv`` with columns: ``experiment, optimizer,
   learning_rate, scheduler, val_loss_best, val_loss_final,
   val_prob_loss_final, val_dist_loss_final, train_loss_final,
   epochs_done``.
2. ``sweep_val_loss_curves.png`` — one panel per scheduler, val_loss
   vs.~epoch, colour by learning rate, line-style by optimizer.
3. ``sweep_best_models.txt`` — the top-K runs ranked by best (lowest)
   ``val_loss`` and (separately) by ``val_prob_loss_final`` /
   ``val_dist_loss_final`` so you can see which loss component a
   given combo is winning on.

Sibling of :mod:`sweep_unet_analyze.py` — same flow, different
metric names, different dir-prefix. For prediction-quality scoring
(IoU vs.~keras refs) use :mod:`sweep_predict_and_analyze.py`
instead; this script works off training-time metrics only.

Edit the paths block at the top, then::

    python sweep_stardist_analyze.py

Sibling cleanup: :mod:`sweep_stardist_clean.py`.
"""

# %%
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# %% ─── paths (edit per cluster) ─────────────────────────────────────
sweep_root = Path("/mnt/jean-zay/models_stardist_pytorch_sweep")

# Folder-name pattern produced by slurm_sweep_stardist_jeanzay.sh:
#   stardist_sweep_<opt>_lr<lr_tag>_<sched>
dir_glob = "stardist_sweep_*"

# Top-K to highlight in the printout / best-models file.
top_k = 5

# Primary metric ranking. Lower is better for every loss column.
primary_metric = "val_loss_best"

script_dir = Path(__file__).resolve().parent
results_folder = script_dir / "stardist_sweeps"
results_folder.mkdir(parents=True, exist_ok=True)
# Outputs (written next to sweep_root).
summary_csv = results_folder / "sweep_summary.csv"
plot_png = results_folder / "sweep_val_loss_curves.png"
best_txt = results_folder / "sweep_best_models.txt"


# %% ─── name parser ─────────────────────────────────────────────────
# LR tags: ``1p0e-3`` → 1.0e-3, ``1p0ep0`` → 1.0e+0, ``1p0ep1`` → 1.0e+1.
_DIR_RE = re.compile(
    r"stardist_sweep_(?P<opt>[a-z]+)_lr(?P<lr_tag>[\dp+\-e]+)_(?P<sched>\w+)"
)


def _decode_lr_tag(lr_tag: str) -> float:
    """Decode the SLURM-safe LR tag back to a float.

    The training script encodes a learning rate as e.g. ``1.0e-3`` →
    ``1p0e-3``; positive exponents use ``ep<N>`` so ``1.0`` is
    ``1p0ep0`` and ``10.0`` is ``1p0ep1``.
    """
    m = re.match(r"^(\d+)p(\d+)e(p?)(-?)(\d+)$", lr_tag)
    if m is None:
        return float("nan")
    int_part, frac_part, _plus, sign, exp_digits = m.groups()
    mantissa = float(f"{int_part}.{frac_part}")
    exp = int(f"{'-' if sign else ''}{exp_digits}")
    return mantissa * (10.0**exp)


def _parse_sweep_tags(model_dir: Path) -> dict:
    m = _DIR_RE.match(model_dir.name)
    if m is None:
        return {
            "experiment": model_dir.name,
            "optimizer": None,
            "learning_rate": None,
            "scheduler": None,
        }
    return {
        "experiment": model_dir.name,
        "optimizer": m.group("opt"),
        "learning_rate": _decode_lr_tag(m.group("lr_tag")),
        "scheduler": m.group("sched"),
    }


# %% ─── metrics reader ─────────────────────────────────────────────
def _read_metrics(model_dir: Path) -> pd.DataFrame:
    """Read ``metrics.csv`` and return a DataFrame indexed by epoch.

    The CSVLogger writes ``<name>_epoch`` and ``<name>_step`` pairs;
    we keep the ``_epoch`` row of each metric and rename it back to
    the bare metric name so downstream rank / plot code can work with
    one canonical schema regardless of Lightning version. Step-level
    NaN rows are dropped during the per-epoch ``max``.
    """
    csv_path = model_dir / "metrics.csv"
    if not csv_path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["epoch"]).copy()
    df["epoch"] = df["epoch"].astype(int)
    agg = df.groupby("epoch", as_index=False).max(numeric_only=True)
    rename_map = {
        c: c[: -len("_epoch")]
        for c in agg.columns
        if c.endswith("_epoch") and c != "epoch"
    }
    agg = agg.rename(columns=rename_map)
    return agg


def _final_summary(model_dir: Path) -> dict:
    """Best / final / epochs from the per-epoch DataFrame. Missing
    columns map to ``None`` so the CSV writer skips them cleanly."""
    agg = _read_metrics(model_dir)
    out = {
        "val_loss_best": None,
        "val_loss_final": None,
        "val_prob_loss_final": None,
        "val_dist_loss_final": None,
        "train_loss_final": None,
        "train_prob_loss_final": None,
        "train_dist_loss_final": None,
        "epochs_done": None,
    }
    if agg.empty:
        return out
    if "val_loss" in agg.columns:
        s = agg["val_loss"].dropna()
        if len(s):
            out["val_loss_best"] = float(s.min())
            out["val_loss_final"] = float(s.iloc[-1])
    for col, key in [
        ("val_prob_loss", "val_prob_loss_final"),
        ("val_dist_loss", "val_dist_loss_final"),
        ("train_loss", "train_loss_final"),
        ("train_prob_loss", "train_prob_loss_final"),
        ("train_dist_loss", "train_dist_loss_final"),
    ]:
        if col in agg.columns:
            s = agg[col].dropna()
            if len(s):
                out[key] = float(s.iloc[-1])
    out["epochs_done"] = int(agg["epoch"].max())
    return out


# %% ─── walk the sweep ──────────────────────────────────────────────
model_dirs = sorted(p for p in sweep_root.glob(dir_glob) if p.is_dir())
print(f"Sweep root:    {sweep_root}")
print(f"Models found:  {len(model_dirs)}")
print()

results = []
curves: dict[str, pd.DataFrame] = {}
for i, model_dir in enumerate(model_dirs):
    tags = _parse_sweep_tags(model_dir)
    summary = _final_summary(model_dir)
    print(
        f"[{i + 1}/{len(model_dirs)}] {tags['experiment']:60s} "
        f"epochs={summary['epochs_done']!s:>4} "
        f"val_loss_best={summary['val_loss_best']!s:>8}"
    )
    results.append({**tags, **summary})
    agg = _read_metrics(model_dir)
    if not agg.empty:
        curves[model_dir.name] = agg


# %% ─── write summary CSV ──────────────────────────────────────────
if not results:
    raise SystemExit(f"No models found in {sweep_root}")

ordered_keys = [
    "experiment",
    "optimizer",
    "learning_rate",
    "scheduler",
    "val_loss_best",
    "val_loss_final",
    "val_prob_loss_final",
    "val_dist_loss_final",
    "train_loss_final",
    "train_prob_loss_final",
    "train_dist_loss_final",
    "epochs_done",
]
with summary_csv.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=ordered_keys)
    w.writeheader()
    for r in results:
        w.writerow(r)
print(f"\nSummary CSV: {summary_csv}")


# %% ─── rank + write best-models text ──────────────────────────────
def _rank(rows, key, *, reverse=False):
    keep = [r for r in rows if r.get(key) is not None]
    return sorted(keep, key=lambda r: r[key], reverse=reverse)


by_val_loss = _rank(results, "val_loss_best", reverse=False)
by_val_prob = _rank(results, "val_prob_loss_final", reverse=False)
by_val_dist = _rank(results, "val_dist_loss_final", reverse=False)

lines = []
lines.append(f"── BEST BY VAL_LOSS (lower is better, top {top_k}) ──")
for r in by_val_loss[:top_k]:
    lines.append(
        f"  val_loss_best={r['val_loss_best']:.6f}  "
        f"{r['experiment']:<60} "
        f"opt={r['optimizer']!s:<6} lr={r['learning_rate']!s:<10} "
        f"sched={r['scheduler']!s:<14} epochs={r['epochs_done']}"
    )
if by_val_prob:
    lines.append("")
    lines.append(f"── BEST BY VAL_PROB_LOSS final (lower is better, top {top_k}) ──")
    for r in by_val_prob[:top_k]:
        lines.append(
            f"  val_prob_loss_final={r['val_prob_loss_final']:.6f}  "
            f"{r['experiment']:<60} "
            f"opt={r['optimizer']!s:<6} lr={r['learning_rate']!s:<10} "
            f"sched={r['scheduler']!s:<14}"
        )
if by_val_dist:
    lines.append("")
    lines.append(f"── BEST BY VAL_DIST_LOSS final (lower is better, top {top_k}) ──")
    for r in by_val_dist[:top_k]:
        lines.append(
            f"  val_dist_loss_final={r['val_dist_loss_final']:.6f}  "
            f"{r['experiment']:<60} "
            f"opt={r['optimizer']!s:<6} lr={r['learning_rate']!s:<10} "
            f"sched={r['scheduler']!s:<14}"
        )

print()
print("\n".join(lines))
best_txt.write_text("\n".join(lines) + "\n")
print(f"\nBest-models text: {best_txt}")


# %% ─── plot val_loss curves ───────────────────────────────────────
schedulers = sorted({r["scheduler"] for r in results if r["scheduler"]})
optimizers = sorted({r["optimizer"] for r in results if r["optimizer"]})

lrs_present = sorted(
    {r["learning_rate"] for r in results if r["learning_rate"] is not None}
)
cmap = plt.get_cmap("viridis")
lr_to_color = {
    lr: cmap(i / max(1, len(lrs_present) - 1)) for i, lr in enumerate(lrs_present)
}
opt_to_ls = {opt: ls for opt, ls in zip(optimizers, ["-", "--", "-.", ":"])}

n_panels = len(schedulers) or 1
fig, axes = plt.subplots(
    1, n_panels, figsize=(5.5 * n_panels, 4.0), sharey=True, squeeze=False
)
for ax, sched in zip(axes[0], schedulers):
    for r in results:
        if r["scheduler"] != sched:
            continue
        agg = curves.get(r["experiment"])
        if agg is None or "val_loss" not in agg.columns:
            continue
        s = agg.dropna(subset=["val_loss"])
        if s.empty:
            continue
        ax.plot(
            s["epoch"],
            s["val_loss"],
            color=lr_to_color.get(r["learning_rate"], "grey"),
            linestyle=opt_to_ls.get(r["optimizer"], "-"),
            linewidth=1.4,
            alpha=0.9,
        )
    ax.set_title(f"scheduler = {sched}", fontsize=10)
    ax.set_xlabel("epoch")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
axes[0][0].set_ylabel("val_loss (log)")

lr_handles = [
    plt.Line2D([0], [0], color=lr_to_color[lr], lw=2, label=f"lr={lr:g}")
    for lr in lrs_present
]
opt_handles = [
    plt.Line2D([0], [0], color="black", linestyle=opt_to_ls[opt], lw=1.5, label=opt)
    for opt in optimizers
]
axes[0][-1].legend(
    handles=lr_handles + opt_handles,
    fontsize=7,
    loc="upper right",
    ncol=1,
    framealpha=0.6,
)
fig.suptitle("StarDist sweep — val_loss curves", fontsize=11)
fig.tight_layout()
fig.savefig(plot_png, dpi=140)
plt.close(fig)
print(f"Curves plot: {plot_png}")


# %% ─── final ranking shown on stdout ─────────────────────────────
sorted_by_primary = _rank(results, primary_metric, reverse=False)
if sorted_by_primary:
    winner = sorted_by_primary[0]
    print(
        f"\nWinner by {primary_metric}: {winner['experiment']} "
        f"({primary_metric}={winner[primary_metric]:.6f})"
    )
    print(
        "→ pass this folder to predict-stardist via "
        "`experiment_data_paths.log_path=...`"
    )
    print(
        "→ for prediction-quality (IoU vs.~keras refs), run "
        "sweep_predict_and_analyze.py."
    )
    print(
        "→ run sweep_stardist_clean.py to trim every sweep dir down "
        "to its last checkpoint."
    )
