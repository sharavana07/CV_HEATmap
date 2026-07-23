"""
Central configuration for the evaluation framework.

Edit the paths below (or override via environment variables / CLI flags in
generate_report.py) to point at your real experiment outputs. Nothing else
in the pipeline needs to change when you add a new model — drop a new
folder under EXPERIMENTS_DIR that follows the layout described in the
README and it will be picked up automatically.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Top-level paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where per-model training logs / predictions / metrics live. Each
# sub-directory of this folder is treated as one experiment (one model run).
EXPERIMENTS_DIR = Path(os.environ.get("OBH_EXPERIMENTS_DIR", PROJECT_ROOT / "experiments"))

# Where every generated artifact is written.
OUTPUT_DIR = Path(os.environ.get("OBH_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

OUTPUT_SUBDIRS = {
    "training": OUTPUT_DIR / "evaluation" / "training",
    "classification": OUTPUT_DIR / "evaluation" / "classification",
    "comparison": OUTPUT_DIR / "evaluation" / "comparison",
    "heatmaps": OUTPUT_DIR / "evaluation" / "heatmaps",
    "feature_maps": OUTPUT_DIR / "evaluation" / "feature_maps",
    "gradcam": OUTPUT_DIR / "evaluation" / "gradcam",
    "tables": OUTPUT_DIR / "evaluation" / "tables",
    "report": OUTPUT_DIR / "evaluation" / "report",
}

# ---------------------------------------------------------------------------
# Expected filenames inside each experiments/<model_name>/ folder.
# All are optional except history.csv / predictions.csv are needed for the
# figures that depend on them; the pipeline degrades gracefully (skips a
# figure + logs why) whenever a required file is missing.
# ---------------------------------------------------------------------------

FILE_HISTORY = "history.json"                # train_loss, val_loss, train_acc, val_acc, lr, epoch_time, gpu_util
FILE_PREDICTIONS = "predictions.csv"         # sample_index, true_label, predicted_label, probability_up
FILE_METRICS = "metrics.json"                # accuracy, precision, recall, f1, auc  (metrics.csv also supported)
FILE_CLASSIFICATION_REPORT = "classification_report.txt"
FILE_PARAMS = "params.json"                  # num_parameters, train_time_sec, inference_time_ms (all optional)
FILE_CHECKPOINT = "checkpoint.pt"            # optional torch state_dict, only needed for feature maps / Grad-CAM
SAMPLES_DIR = "samples"                      # optional folder of heatmap images/arrays for gallery + Grad-CAM

# Image / array files inside samples/ are matched to predictions.csv rows by
# sample_index, trying these patterns in order:
SAMPLE_FILENAME_PATTERNS = [
    "hm_{idx:06d}.npy",
    "sample_{idx:04d}.npy",
    "sample_{idx}.npy",
    "sample_{idx:04d}.png",
    "sample_{idx}.png",
]

# ---------------------------------------------------------------------------
# Figure / style constants
# ---------------------------------------------------------------------------

DPI = 400  # within the requested 300-600 DPI publication range
FIGSIZE_WIDE = (7.0, 4.2)
FIGSIZE_SQUARE = (5.5, 5.0)
FONT_SIZE_BASE = 11

# Colorblind-friendly, high-contrast palette (Okabe-Ito).
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

RANDOM_SEED = 42

# Number of qualitative examples to show per gallery category
GALLERY_N_PER_CATEGORY = 8