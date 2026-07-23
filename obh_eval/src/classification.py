"""Per-model classification performance figures.

For every experiment with predictions.csv, generates:
  - confusion_matrix.png
  - roc_curve.png (with AUC)
  - pr_curve.png (Precision-Recall)
  - prf1_bar.png (Precision / Recall / F1 bar chart)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config, style
from .io_utils import Experiment

log = logging.getLogger("obh_eval")

LABELS = ["DOWN (0)", "UP (1)"]


def _confusion_matrix(exp: Experiment, out_dir: Path) -> None:
    y_true = exp.predictions["true_label"].to_numpy()
    y_pred = exp.predictions["predicted_label"].to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(f"Confusion Matrix — {exp.name}")
    style.savefig(fig, out_dir / "confusion_matrix.png")


def _roc_curve(exp: Experiment, out_dir: Path) -> None:
    if "probability_up" not in exp.predictions.columns:
        log.warning("[%s] no probability_up column — skipping ROC curve.", exp.name)
        return
    y_true = exp.predictions["true_label"].to_numpy()
    y_score = exp.predictions["probability_up"].to_numpy()
    if len(set(y_true)) < 2:
        log.warning("[%s] only one class present — skipping ROC curve.", exp.name)
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)
    ax.plot(fpr, tpr, color=config.PALETTE[0], label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {exp.name}")
    ax.legend(loc="lower right")
    style.savefig(fig, out_dir / "roc_curve.png")


def _pr_curve(exp: Experiment, out_dir: Path) -> None:
    if "probability_up" not in exp.predictions.columns:
        log.warning("[%s] no probability_up column — skipping PR curve.", exp.name)
        return
    y_true = exp.predictions["true_label"].to_numpy()
    y_score = exp.predictions["probability_up"].to_numpy()
    if len(set(y_true)) < 2:
        log.warning("[%s] only one class present — skipping PR curve.", exp.name)
        return
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=config.FIGSIZE_SQUARE)
    ax.plot(recall, precision, color=config.PALETTE[1])
    baseline = y_true.mean()
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall Curve — {exp.name}")
    ax.legend(loc="lower left")
    style.savefig(fig, out_dir / "pr_curve.png")


def _prf1_bar(exp: Experiment, out_dir: Path) -> None:
    y_true = exp.predictions["true_label"].to_numpy()
    y_pred = exp.predictions["predicted_label"].to_numpy()
    values = [
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
    ]
    names = ["Precision", "Recall", "F1"]

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    bars = ax.bar(names, values, color=config.PALETTE[:3])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(f"Precision / Recall / F1 — {exp.name}")
    style.savefig(fig, out_dir / "prf1_bar.png")


def generate(experiments: list[Experiment]) -> None:
    style.apply_style()
    base_out = config.OUTPUT_SUBDIRS["classification"]
    for exp in experiments:
        if not exp.has_predictions:
            log.warning("[%s] no predictions.csv — skipping classification-performance figures.", exp.name)
            continue
        out_dir = base_out / exp.name
        _confusion_matrix(exp, out_dir)
        _roc_curve(exp, out_dir)
        _pr_curve(exp, out_dir)
        _prf1_bar(exp, out_dir)