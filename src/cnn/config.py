"""
config.py — Central configuration for CV_HEATmap CNN training system.

All hyperparameters, paths, and runtime settings live here.
Modify this file to run experiments without touching training logic.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────────────────────────────────────

# Root of the CV_HEATmap project (two levels up from src/cnn/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR        = PROJECT_ROOT / "src" / "data"
HEATMAP_DIR     = DATA_DIR / "heatmaps_npy"          # folder of .npy heatmap files
LABELS_PATH     = DATA_DIR / "labels_final.npy"  # primary label source
LABELS_ALT_PATH = DATA_DIR / "labels.npy"         # fallback
MID_PRICES_PATH = DATA_DIR / "mid_prices.npy"

OUTPUT_DIR      = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR  = OUTPUT_DIR / "checkpoints"
PLOT_DIR        = OUTPUT_DIR / "plots"
RESULTS_DIR     = OUTPUT_DIR / "results"

# Auto-create output folders
for _d in [OUTPUT_DIR, CHECKPOINT_DIR, PLOT_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH     = CHECKPOINT_DIR / "best_model.pth"
PREDICTIONS_CSV     = RESULTS_DIR    / "predictions.csv"

# ─────────────────────────────────────────────────────────────────────────────
# INPUT DIMENSIONS
# ─────────────────────────────────────────────────────────────────────────────

IMG_HEIGHT   = 64     # heatmap rows  (price levels)
IMG_WIDTH    = 100    # heatmap cols  (time steps / depth)
IN_CHANNELS  = 1      # grayscale order-book heatmap
NUM_CLASSES  = 2      # 0 = DOWN, 1 = UP

# ─────────────────────────────────────────────────────────────────────────────
# DATA SPLIT  (temporal — no shuffle)
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15    # 1 - TRAIN_RATIO - VAL_RATIO

# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE          = 64
NUM_WORKERS         = 2      # safe for a laptop; raise to 4 if RAM allows
PIN_MEMORY          = True   # faster CPU→GPU transfer
PERSISTENT_WORKERS  = True   # keep workers alive between epochs
PREFETCH_FACTOR     = 2      # batches prefetched per worker

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

# Conv blocks: (out_channels, kernel_size, pool_size)
CONV_BLOCKS = [
    (32, 3, 2),
    (64, 3, 2),
    (128, 3, 2),
]
FC_HIDDEN    = 256
DROPOUT_RATE = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-4      # L2 regularisation (AdamW-style)
NUM_EPOCHS       = 50
GRAD_CLIP_NORM   = 1.0       # max gradient norm; None to disable

# Mixed precision (AMP) — requires CUDA
USE_AMP = True
LABEL_SMOOTHING = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

# ReduceLROnPlateau — reduce LR when val_loss stagnates
LR_PATIENCE   = 5
LR_FACTOR     = 0.5
LR_MIN        = 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────

EARLY_STOP_PATIENCE = 10     # epochs without val_loss improvement → stop
EARLY_STOP_DELTA    = 1e-4   # minimum improvement to count as progress

# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION  (computed from training split; these are placeholders)
# ─────────────────────────────────────────────────────────────────────────────

# Set to None to compute from training data at runtime (recommended).
# Supply pre-computed values here to skip the scan on subsequent runs.
NORM_MEAN = None   # e.g. 0.0412
NORM_STD  = None   # e.g. 0.1237

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME FLAGS
# ─────────────────────────────────────────────────────────────────────────────

CUDNN_BENCHMARK = True   # faster conv ops when input size is fixed
VERBOSE         = True   # print per-epoch stats
LOG_INTERVAL    = 50     # print training loss every N batches