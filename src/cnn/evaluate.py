"""
evaluate.py — Comprehensive evaluation of the trained OrderBookCNN.

Outputs
───────
  outputs/results/predictions.csv          — per-sample predictions for backtest
  outputs/results/classification_report.txt
  outputs/plots/confusion_matrix.png
  outputs/plots/roc_curve.png
  outputs/plots/confidence_distribution.png
  outputs/plots/loss_curves.png
  outputs/plots/accuracy_curves.png

All plots are saved automatically; the script can also be imported and its
functions called from a Jupyter notebook.

Index alignment
────────────────
dataset.py is the single source of truth for FLAT removal, binary label
conversion, and (global_idx, binary_label) pairing (see Sample in
cnn/dataset.py). This script NEVER recomputes indices or re-runs a temporal
split against the raw label array — doing so would use a different index
space (raw 3-class length) than the one build_dataloaders() actually split
on (post-FLAT-removal binary sample count), silently misaligning
predictions.csv against the wrong heatmaps.

Instead, the test split's global heatmap indices are read directly off the
test_loader's own dataset: test_loader.dataset.samples is the exact list of
Sample(global_idx, label) pairs build_dataloaders() constructed internally.
That is the only place this script gets indices from.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn import config as cfg
from cnn.models import get_model
from cnn.dataset import build_dataloaders
from cnn.utils import get_device

log = logging.getLogger(__name__)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":     120,
    "font.family":    "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})
COLORS = {"train": "#2196F3", "val": "#FF5722", "test": "#4CAF50"}


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:  nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model over a DataLoader.

    Returns
    -------
    true_labels   : (N,) int
    pred_labels   : (N,) int
    prob_up       : (N,) float  — probability of class 1 (UP)
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)

        logits = model(imgs)
        probs  = F.softmax(logits, dim=1)   # (B, 2)

        pred   = logits.argmax(dim=1).cpu().numpy()
        prob1  = probs[:, 1].cpu().numpy()   # P(UP)
        true   = labels.numpy()

        all_true.append(true)
        all_pred.append(pred)
        all_prob.append(prob1)

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
) -> Dict[str, float]:
    fpr, tpr, _ = roc_curve(true, prob)
    roc_auc = auc(fpr, tpr)

    metrics = {
        "accuracy":  accuracy_score(true, pred),
        "precision": precision_score(true, pred, zero_division=0),
        "recall":    recall_score(true, pred, zero_division=0),
        "f1":        f1_score(true, pred, zero_division=0),
        "auc":       roc_auc,
    }
    return metrics, fpr, tpr


def print_metrics(metrics: Dict[str, float], split: str = "test") -> None:
    print(f"\n{'='*55}")
    print(f"  Evaluation — {split.upper()} SET")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    true: np.ndarray,
    pred: np.ndarray,
    path: Path = cfg.PLOT_DIR / "confusion_matrix.png",
) -> None:
    cm   = confusion_matrix(true, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["DOWN (0)", "UP (1)"])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


def plot_roc_curve(
    true:    np.ndarray,
    prob:    np.ndarray,
    roc_auc: float,
    path:    Path = cfg.PLOT_DIR / "roc_curve.png",
) -> None:
    fpr, tpr, _ = roc_curve(true, prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=COLORS["test"], lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


def plot_confidence_distribution(
    prob_up: np.ndarray,
    true:    np.ndarray,
    path:    Path = cfg.PLOT_DIR / "confidence_distribution.png",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(prob_up[true == 0], bins=50, alpha=0.65, color=COLORS["val"],
            label="True DOWN (0)", density=True)
    ax.hist(prob_up[true == 1], bins=50, alpha=0.65, color=COLORS["train"],
            label="True UP (1)",   density=True)
    ax.axvline(0.5, color="k", linestyle="--", lw=1, label="Decision boundary")
    ax.set_xlabel("P(UP)")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Confidence Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


def plot_training_curves(
    history: Dict[str, List],
    loss_path: Path = cfg.PLOT_DIR / "loss_curves.png",
    acc_path:  Path = cfg.PLOT_DIR / "accuracy_curves.png",
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Loss ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], color=COLORS["train"], label="Train loss")
    ax.plot(epochs, history["val_loss"],   color=COLORS["val"],   label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", loss_path)

    # ── Accuracy ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [v * 100 for v in history["train_acc"]],
            color=COLORS["train"], label="Train acc")
    ax.plot(epochs, [v * 100 for v in history["val_acc"]],
            color=COLORS["val"],   label="Val acc")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(acc_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", acc_path)


# ─────────────────────────────────────────────────────────────────────────────
# Predictions CSV (for backtest)
# ─────────────────────────────────────────────────────────────────────────────

def save_predictions_csv(
    indices:    np.ndarray,
    true:       np.ndarray,
    pred:       np.ndarray,
    prob_up:    np.ndarray,
    path:       Path = cfg.PREDICTIONS_CSV,
) -> None:
    """
    Export per-sample predictions in a format ready for the trading backtest.

    Columns
    ───────
    sample_index   : the ORIGINAL global heatmap index for this row, i.e.
                     hm_{sample_index:06d}.npy — taken verbatim from each
                     test-split Sample.global_idx, never recomputed here.
    true_label     : 0 = DOWN, 1 = UP
    predicted_label: model prediction
    probability_up : P(UP) from softmax
    """
    df = pd.DataFrame({
        "sample_index":    indices,
        "true_label":      true.astype(int),
        "predicted_label": pred.astype(int),
        "probability_up":  prob_up,
    })
    df.to_csv(path, index=False)
    log.info("Predictions CSV saved → %s  (%d rows)", path, len(df))


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model:        torch.nn.Module,
    test_loader:  torch.utils.data.DataLoader,
    test_indices: np.ndarray,
    history:      Optional[Dict[str, List]] = None,
) -> Dict[str, float]:
    """
    Run full evaluation: metrics, plots, predictions CSV.

    Parameters
    ----------
    model        : trained OrderBookCNN (best checkpoint already loaded)
    test_loader  : DataLoader for the held-out test split, as returned by
                   cnn.dataset.build_dataloaders()
    test_indices : global heatmap indices for the test split, i.e.
                   [s.global_idx for s in test_loader.dataset.samples] —
                   must come from the test dataset itself, never recomputed
    history      : training history dict (from train.py); if supplied,
                   loss/accuracy curves are plotted
    """
    device = get_device()
    model  = model.to(device)

    # ── Inference ─────────────────────────────────────────────────────
    log.info("Running inference on test set …")
    true, pred, prob_up = run_inference(model, test_loader, device)

    if len(test_indices) != len(true):
        raise ValueError(
            f"test_indices length ({len(test_indices)}) does not match the "
            f"number of test predictions ({len(true)}). test_indices must "
            "come from test_loader.dataset.samples, not a separately "
            "computed split."
        )

    # ── Metrics ───────────────────────────────────────────────────────
    metrics, fpr, tpr = compute_metrics(true, pred, prob_up)
    print_metrics(metrics)

    # Classification report (per-class precision/recall)
    report = classification_report(
        true, pred,
        target_names=["DOWN (0)", "UP (1)"],
        digits=4,
    )
    print(report)
    rpt_path = cfg.RESULTS_DIR / "classification_report.txt"
    rpt_path.write_text(report)
    log.info("Classification report saved → %s", rpt_path)

    # Class distribution
    log.info(
        "Test class distribution — 0 (DOWN): %d  1 (UP): %d",
        (true == 0).sum(), (true == 1).sum(),
    )

    # ── Plots ─────────────────────────────────────────────────────────
    plot_confusion_matrix(true, pred)
    plot_roc_curve(true, prob_up, metrics["auc"])
    plot_confidence_distribution(prob_up, true)

    if history is not None:
        plot_training_curves(history)

    # ── Predictions CSV ───────────────────────────────────────────────
    save_predictions_csv(test_indices, true, pred, prob_up)

    # ── Summary JSON ──────────────────────────────────────────────────
    summary_path = cfg.RESULTS_DIR / "test_metrics.json"
    with open(summary_path, "w") as f:
        json.dump({k: round(float(v), 6) for k, v in metrics.items()}, f, indent=2)
    log.info("Test metrics saved → %s", summary_path)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Evaluate a trained OrderBookCNN model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[cfg.MODEL_CNN, cfg.MODEL_CNN_SE, cfg.MODEL_DAFNET],
        help="Which model to evaluate (cnn, cnn_se, dafnet)",
    )
    args = parser.parse_args()

    cfg.MODEL_NAME = args.model

    checkpoint_map = {
        cfg.MODEL_CNN: cfg.BEST_CNN_MODEL,
        cfg.MODEL_CNN_SE: cfg.BEST_CNN_SE_MODEL,
        cfg.MODEL_DAFNET: cfg.BEST_DAFNET_MODEL,
    }
    checkpoint_path = checkpoint_map[args.model]

    device = get_device()
    model = get_model(args.model).to(device)

    if not checkpoint_path.exists():
        log.error("No checkpoint found at %s — run train.py for this model first.", checkpoint_path)
        raise SystemExit(1)

    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    log.info("Loaded checkpoint: %s", checkpoint_path)

    lbl_path = cfg.LABELS_PATH if cfg.LABELS_PATH.exists() else cfg.LABELS_ALT_PATH
    raw_labels = np.load(str(lbl_path))

    if cfg.HEATMAP_DIR.exists():
        _, _, test_loader, _, _ = build_dataloaders(raw_labels, heatmap_dir=cfg.HEATMAP_DIR)
    else:
        log.warning("Heatmap dir not found — using synthetic data.")
        arr = np.random.rand(len(raw_labels), cfg.IMG_HEIGHT, cfg.IMG_WIDTH).astype(np.float32)
        _, _, test_loader, _, _ = build_dataloaders(raw_labels, heatmap_array=arr)

    test_indices = np.array(
        [sample.global_idx for sample in test_loader.dataset.samples],
        dtype=np.int64,
    )

    hist_path = cfg.RESULTS_DIR / f"training_history_{args.model}.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else None

    evaluate(model, test_loader, test_indices, history=history)