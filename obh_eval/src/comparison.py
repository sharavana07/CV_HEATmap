"""Cross-model comparison figures.

Builds one bar chart per metric (accuracy, precision, recall, f1, auc,
training time, inference time — whichever are available across the
discovered experiments) plus a ranked comparison table (by F1, tie-broken
by accuracy) saved as CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import config, style
from .io_utils import Experiment

log = logging.getLogger("obh_eval")

# (metric key as found in exp.metrics / exp.params, display label, higher_is_better)
METRIC_SPECS = [
    ("accuracy", "Accuracy", True),
    ("precision", "Precision", True),
    ("recall", "Recall", True),
    ("f1", "F1 Score", True),
    ("auc", "ROC-AUC", True),
    ("train_time_sec", "Training Time (s)", False),
    ("inference_time_ms", "Inference Time (ms/sample)", False),
]


def _collect_metric_table(experiments: list[Experiment]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        row = {"model": exp.name}
        row.update({k: exp.metrics.get(k) for k, _, _ in METRIC_SPECS if k in exp.metrics})
        row.update({k: exp.params.get(k) for k, _, _ in METRIC_SPECS if k in exp.params})
        row["num_parameters"] = exp.params.get("num_parameters")
        rows.append(row)
    return pd.DataFrame(rows)


def _bar_chart(df: pd.DataFrame, metric_key: str, label: str, higher_is_better: bool, out_dir: Path) -> None:
    if metric_key not in df.columns or df[metric_key].dropna().empty:
        log.warning("Metric '%s' not available for any model — skipping comparison chart.", metric_key)
        return
    sub = df[["model", metric_key]].dropna().sort_values(metric_key, ascending=not higher_is_better)

    fig, ax = plt.subplots(figsize=config.FIGSIZE_WIDE)
    colors = [config.PALETTE[i % len(config.PALETTE)] for i in range(len(sub))]
    bars = ax.bar(sub["model"], sub[metric_key], color=colors)
    for b, v in zip(bars, sub[metric_key]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}" if v < 10 else f"{v:.1f}",
                 ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(label)
    ax.set_title(f"Model Comparison — {label}")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    style.savefig(fig, out_dir / f"comparison_{metric_key}.png")


def generate(experiments: list[Experiment]) -> pd.DataFrame:
    style.apply_style()
    out_dir = config.OUTPUT_SUBDIRS["comparison"]
    df = _collect_metric_table(experiments)

    if df.empty:
        log.warning("No experiments with metrics found — skipping all comparison charts.")
        return df

    for key, label, higher_is_better in METRIC_SPECS:
        _bar_chart(df, key, label, higher_is_better, out_dir)

    # Ranked table: sort by F1 (falls back to accuracy if F1 unavailable)
    sort_key = "f1" if "f1" in df.columns and df["f1"].notna().any() else "accuracy"
    if sort_key in df.columns:
        ranked = df.sort_values(sort_key, ascending=False).reset_index(drop=True)
        ranked.insert(0, "rank", ranked.index + 1)
    else:
        ranked = df
    ranked.to_csv(config.OUTPUT_SUBDIRS["tables"] / "ranked_model_comparison.csv", index=False)
    return ranked