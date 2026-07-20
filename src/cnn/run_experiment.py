"""
run_experiment.py — Master entry point for CV_HEATmap CNN training system.

This script wires together every module in the correct order:
  logger → preprocess (optional) → dataset → train → checkpoint → evaluate → visualize

It is the SINGLE command you run to go from raw data to trained model
and all evaluation outputs.

Resume support
──────────────
  If outputs/checkpoints/full_checkpoint.pth already exists, training
  automatically resumes from the last saved epoch rather than restarting.
  Pass --no-resume to force a fresh training run.

Experiment tracking
────────────────────
  An ExperimentLogger writes a JSON manifest at the start and updates it
  at the end with final metrics, git commit, GPU info, and training history.
  Find it at: outputs/results/experiment_<timestamp>.json

Usage
─────
  # Full pipeline from scratch:
  python src/cnn/run_experiment.py

  # Resume interrupted training:
  python src/cnn/run_experiment.py

  # Force fresh training (ignore existing checkpoint):
  python src/cnn/run_experiment.py --no-resume

  # Skip visualisation (faster, useful on headless servers):
  python src/cnn/run_experiment.py --no-viz

  # Custom experiment name (appears in manifest filename):
  python src/cnn/run_experiment.py --name ablation_dropout0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Path bootstrap (must come before any cnn.* imports) ──────────────────────
_SRC = Path(__file__).resolve().parents[1]   # CV_HEATmap/src/
sys.path.insert(0, str(_SRC))

from cnn import config as cfg
from cnn.logger import setup_logging, ExperimentLogger, log_gpu_memory
from cnn.utils import set_seed, get_device, EarlyStopping
from cnn.model import OrderBookCNN, print_model_summary
from cnn.dataset import build_dataloaders, temporal_split
from cnn.checkpoint import (
    save_checkpoint, load_checkpoint, restore_full_state,
    checkpoint_summary, save_config_json, FULL_CKPT_PATH,
)
from cnn.train import _train_epoch, _eval_epoch, _try_get_gpu_util
from cnn.evaluate import evaluate
from cnn.visualize import (
    plot_conv_filters, plot_feature_maps, plot_sample_grid, plot_gradcam_grid,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CV_HEATmap — full CNN training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--name", type=str, default="cv_heatmap_cnn",
                   help="Experiment name (used in log/manifest filenames)")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore existing checkpoint and train from scratch")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip visualisation plots (faster on headless servers)")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training, only run evaluation on existing checkpoint")
    p.add_argument("--n-viz-samples", type=int, default=8,
                   help="Number of samples for Grad-CAM / sample grid plots")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"],
                   help="Console log verbosity")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_data(experiment: ExperimentLogger):
    """Resolve labels and heatmaps, return DataLoaders + split indices."""
    # Labels
    lbl_path = cfg.LABELS_PATH if cfg.LABELS_PATH.exists() else cfg.LABELS_ALT_PATH
    if not lbl_path.exists():
        raise FileNotFoundError(
            f"No label file at {cfg.LABELS_PATH} or {cfg.LABELS_ALT_PATH}.\n"
            "Run generate_labels.py first."
        )
    labels = np.load(str(lbl_path))
    log.info(
        "Labels loaded: %s  DOWN=%d  UP=%d",
        labels.shape, (labels == 0).sum(), (labels == 1).sum(),
    )

    # Heatmaps
    if cfg.HEATMAP_DIR.exists() and len(list(cfg.HEATMAP_DIR.glob("*.npy"))) > 0:
        log.info("Using per-sample heatmaps from %s", cfg.HEATMAP_DIR)
        train_loader, val_loader, test_loader, mean, std = build_dataloaders(
            labels, heatmap_dir=cfg.HEATMAP_DIR
        )
    else:
        log.warning(
            "Heatmap directory not found or empty (%s).\n"
            "→ Using synthetic random data for pipeline testing.\n"
            "→ Run preprocess.py to convert your real heatmaps.",
            cfg.HEATMAP_DIR,
        )
        arr = np.random.rand(len(labels), cfg.IMG_HEIGHT, cfg.IMG_WIDTH).astype(np.float32)
        train_loader, val_loader, test_loader, mean, std = build_dataloaders(
            labels, heatmap_array=arr
        )

    _, _, test_idx = temporal_split(len(labels))
    log.info("Normalisation — mean: %.5f  std: %.5f", mean, std)
    return train_loader, val_loader, test_loader, test_idx, mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Training loop with resume support
# ─────────────────────────────────────────────────────────────────────────────

def _build_training_objects(device: torch.device):
    """Instantiate model, optimiser, scheduler, scaler, stopper."""
    model     = OrderBookCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.NUM_EPOCHS))
    _amp_device = device.type if device.type == "cuda" else "cpu"
    scaler  = GradScaler(_amp_device, enabled=cfg.USE_AMP and device.type == "cuda")
    stopper = EarlyStopping(
        patience=cfg.EARLY_STOP_PATIENCE,
        delta=cfg.EARLY_STOP_DELTA,
        path=cfg.BEST_MODEL_PATH,
    )
    return model, optimizer, scheduler, scaler, stopper


def _run_training(
    train_loader,
    val_loader,
    resume: bool = True,
) -> tuple[nn.Module, Dict[str, List]]:
    """
    Full training loop with optional resume from full_checkpoint.pth.

    Returns the best model (weights loaded) and the complete history dict.
    """
    set_seed(cfg.SEED)
    device = get_device()
    log_gpu_memory("before model init")

    model, optimizer, scheduler, scaler, stopper = _build_training_objects(device)
    print_model_summary(model)
    log_gpu_memory("after model init")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)

    history: Dict[str, List] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr": [], "epoch_time": [], "gpu_util": [],
    }
    start_epoch = 1

    # ── Resume? ───────────────────────────────────────────────────────
    if resume and FULL_CKPT_PATH.exists():
        log.info("Resuming from checkpoint: %s", FULL_CKPT_PATH)
        checkpoint_summary(FULL_CKPT_PATH)
        state = load_checkpoint(FULL_CKPT_PATH, device=device)
        start_epoch = restore_full_state(state, model, optimizer, scheduler, scaler, stopper)
        history = state.get("history", history)
    else:
        if resume and not FULL_CKPT_PATH.exists():
            log.info("No checkpoint found — starting fresh training.")
        elif not resume:
            log.info("--no-resume flag set — starting fresh training.")

    # ── Main loop ─────────────────────────────────────────────────────
    log.info(
        "Training: epochs %d → %d  |  device: %s  |  AMP: %s",
        start_epoch, cfg.NUM_EPOCHS, device, cfg.USE_AMP,
    )

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        t0 = time.perf_counter()

        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = _eval_epoch(model, val_loader, criterion, device)

        elapsed  = time.perf_counter() - t0
        lr_now   = optimizer.param_groups[0]["lr"]
        gpu_util = _try_get_gpu_util()

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)
        history["epoch_time"].append(elapsed)
        history["gpu_util"].append(gpu_util)

        gpu_str = f"  GPU:{gpu_util:.0f}%" if gpu_util is not None else ""
        log.info(
            "Epoch %3d/%d  train_loss=%.4f  train_acc=%.4f  "
            "val_loss=%.4f  val_acc=%.4f  lr=%.2e  %.1fs%s",
            epoch, cfg.NUM_EPOCHS,
            train_loss, train_acc, val_loss, val_acc,
            lr_now, elapsed, gpu_str,
        )

        # Save full checkpoint every epoch (enables resume at any point)
        save_checkpoint(
            epoch, model, optimizer, scheduler, scaler, stopper, history
        )

        # Early stopping (also saves best_model.pth internally)
        if stopper(val_loss, model, epoch=epoch):
            log.info("Early stopping at epoch %d (best epoch: %d).",
                     epoch, stopper.best_epoch)
            break

    # ── Load best weights ─────────────────────────────────────────────
    log.info("Loading best model ← %s", cfg.BEST_MODEL_PATH)
    model.load_state_dict(
        torch.load(str(cfg.BEST_MODEL_PATH), map_location=device)
    )

    # Persist history
    hist_path = cfg.RESULTS_DIR / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Training history saved → %s", hist_path)

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _run_visualisations(
    model:       nn.Module,
    test_loader: torch.utils.data.DataLoader,
    n_samples:   int,
) -> None:
    """Plot filters, feature maps, sample predictions, and Grad-CAM."""
    device = next(model.parameters()).device

    # Collect a mini-batch from test set
    batch_imgs, batch_true = next(iter(test_loader))
    batch_imgs  = batch_imgs[:n_samples]
    batch_true  = batch_true[:n_samples].numpy()

    with torch.no_grad():
        import torch.nn.functional as F
        logits   = model(batch_imgs.to(device))
        probs    = F.softmax(logits, dim=1).cpu().numpy()
        pred_lbl = logits.argmax(dim=1).cpu().numpy()
    prob_up = probs[:, 1]

    log.info("Generating visualisation plots …")

    try:
        plot_conv_filters(model, layer_idx=0)
    except Exception as e:
        log.warning("Filter plot failed: %s", e)

    try:
        plot_feature_maps(model, batch_imgs[:1])
    except Exception as e:
        log.warning("Feature map plot failed: %s", e)

    try:
        plot_sample_grid(batch_imgs, batch_true, pred_lbl, prob_up)
    except Exception as e:
        log.warning("Sample grid plot failed: %s", e)

    try:
        plot_gradcam_grid(model, batch_imgs, batch_true, pred_lbl, prob_up)
    except Exception as e:
        log.warning("Grad-CAM plot failed: %s", e)

    log.info("Visualisation plots saved → %s", cfg.PLOT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── 1. Logging ────────────────────────────────────────────────────
    from datetime import datetime
    exp_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = setup_logging(
        level=getattr(logging, args.log_level),
        experiment_id=exp_id,
    )

    experiment = ExperimentLogger(experiment_name=args.name, experiment_id=exp_id)
    experiment.start()

    # ── 2. Config snapshot ────────────────────────────────────────────
    save_config_json()

    try:
        # ── 3. Data ───────────────────────────────────────────────────
        train_loader, val_loader, test_loader, test_idx, mean, std = _load_data(experiment)

        # ── 4. Eval-only shortcut ─────────────────────────────────────
        if args.eval_only:
            log.info("--eval-only: skipping training.")
            if not cfg.BEST_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"No model at {cfg.BEST_MODEL_PATH}. Run training first."
                )
            device = get_device()
            model  = OrderBookCNN().to(device)
            model.load_state_dict(
                torch.load(str(cfg.BEST_MODEL_PATH), map_location=device)
            )
            history = None
            hist_p  = cfg.RESULTS_DIR / "training_history.json"
            if hist_p.exists():
                history = json.loads(hist_p.read_text())
        else:
            # ── 5. Training ───────────────────────────────────────────
            model, history = _run_training(
                train_loader, val_loader,
                resume=not args.no_resume,
            )
            experiment.log_history(history)

        # ── 6. Evaluation ─────────────────────────────────────────────
        log.info("Running evaluation on test set …")
        hist_p = cfg.RESULTS_DIR / "training_history.json"
        history = json.loads(hist_p.read_text()) if hist_p.exists() else None

        metrics = evaluate(model, test_loader, test_idx, history=history)
        experiment.finish(metrics=metrics)

        # ── 7. Visualisation ──────────────────────────────────────────
        if not args.no_viz:
            _run_visualisations(model, test_loader, n_samples=args.n_viz_samples)
        else:
            log.info("--no-viz: skipping visualisation plots.")

        # ── 8. Final summary ──────────────────────────────────────────
        log.info("")
        log.info("=" * 60)
        log.info("  EXPERIMENT COMPLETE")
        log.info("=" * 60)
        for k, v in metrics.items():
            log.info("  %-12s: %.4f", k, v)
        log.info("")
        log.info("  Outputs:")
        log.info("    Model     → %s", cfg.BEST_MODEL_PATH)
        log.info("    Checkpoint→ %s", FULL_CKPT_PATH)
        log.info("    Plots     → %s", cfg.PLOT_DIR)
        log.info("    Results   → %s", cfg.RESULTS_DIR)
        log.info("    Log file  → %s", log_path)
        log.info("=" * 60)

    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        experiment.mark_failed("KeyboardInterrupt")
        sys.exit(130)
    except Exception as exc:
        log.exception("Experiment failed: %s", exc)
        experiment.mark_failed(str(exc))
        raise


if __name__ == "__main__":
    main()