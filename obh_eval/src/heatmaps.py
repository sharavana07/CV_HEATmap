"""Qualitative prediction gallery.

For each model with both predictions.csv and a samples/ directory of saved
order-book heatmaps, builds a grid figure per category:
    - correct predictions
    - incorrect predictions
    - highest-confidence predictions
    - lowest-confidence (most uncertain) predictions
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, style
from .io_utils import Experiment, load_sample_array

log = logging.getLogger("obh_eval")

LABEL_NAMES = {0: "DOWN", 1: "UP"}


def _grid(exp: Experiment, subset: pd.DataFrame, title: str, out_path: Path) -> None:
    n = min(len(subset), config.GALLERY_N_PER_CATEGORY)
    if n == 0:
        return
    subset = subset.iloc[:n]
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    loaded_any = False
    for ax, (_, row) in zip(axes, subset.iterrows()):
        arr = load_sample_array(exp, int(row["sample_index"]))
        if arr is not None:
            ax.imshow(arr, cmap="gray", aspect="auto")
            loaded_any = True
        else:
            ax.text(0.5, 0.5, "sample\nnot found", ha="center", va="center", fontsize=8)
            ax.set_facecolor("#f2f2f2")
        true_lbl = LABEL_NAMES.get(int(row["true_label"]), row["true_label"])
        pred_lbl = LABEL_NAMES.get(int(row["predicted_label"]), row["predicted_label"])
        prob = row.get("probability_up", np.nan)
        correct = row["true_label"] == row["predicted_label"]
        color = "#009E73" if correct else "#D55E00"
        subtitle = f"GT: {true_lbl} | Pred: {pred_lbl}"
        if not np.isnan(prob):
            subtitle += f"\nP(UP)={prob:.2f}"
        ax.set_title(subtitle, fontsize=9, color=color)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"{title} — {exp.name}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if not loaded_any:
        log.warning(
            "[%s] no sample images found under %s — grid saved with placeholders only.",
            exp.name, exp.samples_dir,
        )
    style.savefig(fig, out_path)


def generate(experiments: list[Experiment]) -> None:
    style.apply_style()
    base_out = config.OUTPUT_SUBDIRS["heatmaps"]

    for exp in experiments:
        if not exp.has_predictions:
            continue
        if exp.samples_dir is None:
            log.info("[%s] no samples/ directory — skipping heatmap gallery (metrics-only run).", exp.name)
            continue

        preds = exp.predictions.copy()
        if "probability_up" in preds.columns:
            preds["confidence"] = (preds["probability_up"] - 0.5).abs() * 2
        else:
            preds["confidence"] = np.nan

        correct = preds[preds["true_label"] == preds["predicted_label"]]
        incorrect = preds[preds["true_label"] != preds["predicted_label"]]
        high_conf = preds.sort_values("confidence", ascending=False) if preds["confidence"].notna().any() else preds.iloc[0:0]
        low_conf = preds.sort_values("confidence", ascending=True) if preds["confidence"].notna().any() else preds.iloc[0:0]

        out_dir = base_out / exp.name
        _grid(exp, correct.sample(frac=1, random_state=config.RANDOM_SEED) if len(correct) else correct,
              "Correct Predictions", out_dir / "correct.png")
        _grid(exp, incorrect.sample(frac=1, random_state=config.RANDOM_SEED) if len(incorrect) else incorrect,
              "Incorrect Predictions", out_dir / "incorrect.png")
        _grid(exp, high_conf, "High-Confidence Predictions", out_dir / "high_confidence.png")
        _grid(exp, low_conf, "Low-Confidence Predictions", out_dir / "low_confidence.png")