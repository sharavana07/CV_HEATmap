"""
train.py — Full training loop, now fully model‑aware.

The pipeline itself (AMP, scheduler, optimiser, early stopping,
dataloaders, normalisation) remains architecture‑agnostic.
Only checkpoint paths and the history file are now derived
from `cfg.MODEL_NAME`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.cnn import config as cfg
    from src.cnn.models import get_model
    from src.cnn.model import print_model_summary  # generic, works on any nn.Module
    from src.cnn.dataset import build_dataloaders
    from src.cnn.utils import set_seed, get_device, EarlyStopping
except ImportError:  # pragma: no cover - fallback for direct script execution
    from cnn import config as cfg
    from cnn.models import get_model
    from cnn.model import print_model_summary  # generic, works on any nn.Module
    from cnn.dataset import build_dataloaders
    from cnn.utils import set_seed, get_device, EarlyStopping

log = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for runtime model selection."""
    parser = argparse.ArgumentParser(description="Train the CNN-based heatmap classifier")
    parser.add_argument(
        "--model",
        choices=[cfg.MODEL_CNN, cfg.MODEL_CNN_SE, cfg.MODEL_DAFNET],
        default=cfg.MODEL_NAME,
        help="Model architecture to train",
    )
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# GPU utilisation (optional — requires pynvml)
# ─────────────────────────────────────────────────────────────────────────────

def _try_get_gpu_util() -> Optional[float]:
    """Return GPU utilisation % or None if pynvml is unavailable."""
    try:
        import pynvml
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helper: pick the correct best‑model path for the active architecture
# ─────────────────────────────────────────────────────────────────────────────

def _get_best_model_path(model_name: Optional[str] = None) -> Path:
    """Return the best-checkpoint path for the chosen architecture."""
    selected_model_name = model_name or cfg.MODEL_NAME
    mapping = {
        cfg.MODEL_CNN:     cfg.BEST_CNN_MODEL,
        cfg.MODEL_CNN_SE:  cfg.BEST_CNN_SE_MODEL,
        cfg.MODEL_DAFNET:  cfg.BEST_DAFNET_MODEL,
    }
    try:
        return mapping[selected_model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown MODEL_NAME '{selected_model_name}'. "
            f"Expected one of {list(mapping.keys())}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# One epoch helpers
# ─────────────────────────────────────────────────────────────────────────────

def _train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    n_samples  = 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device.type, enabled=cfg.USE_AMP and device.type == "cuda"):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()

        # Gradient clipping (unscale first so clip sees real grad magnitudes)
        if cfg.GRAD_CLIP_NORM is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()

        bs          = imgs.size(0)
        total_loss += loss.item() * bs
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        n_samples  += bs

        if cfg.VERBOSE and (batch_idx + 1) % cfg.LOG_INTERVAL == 0:
            log.debug(
                "  batch %d/%d  loss: %.4f",
                batch_idx + 1, len(loader), loss.item(),
            )

    return total_loss / n_samples, correct / n_samples


def _eval_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float]:
    """Run one evaluation epoch. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    n_samples  = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device.type, enabled=cfg.USE_AMP and device.type == "cuda"):
                logits = model(imgs)
                loss   = criterion(logits, labels)

            bs          = imgs.size(0)
            total_loss += loss.item() * bs
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            n_samples  += bs

    return total_loss / n_samples, correct / n_samples


# ─────────────────────────────────────────────────────────────────────────────
# Main train function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    model_name: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, List]]:
    """
    Full training loop.

    Returns
    -------
    model   : best checkpoint (lowest val_loss) loaded back
    history : dict with lists: train_loss, val_loss, train_acc, val_acc,
              lr, epoch_time, gpu_util
    """
    set_seed(cfg.SEED)
    device = get_device()

    # ── Model ─────────────────────────────────────────────────────────
    selected_model_name = model_name or cfg.MODEL_NAME
    model = get_model(selected_model_name).to(device)
    log.info("Architecture: %s", selected_model_name)
    if cfg.VERBOSE:
        print_model_summary(model)

    # ── Loss / Optimiser ──────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.LR_FACTOR,
        patience=cfg.LR_PATIENCE,
        min_lr=cfg.LR_MIN,
    )

    _amp_device = device.type if device.type == "cuda" else "cpu"
    scaler  = GradScaler(_amp_device, enabled=cfg.USE_AMP and device.type == "cuda")

    # Model-aware checkpoint path
    best_ckpt_path = _get_best_model_path(model_name=selected_model_name)
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    stopper = EarlyStopping(
        patience=cfg.EARLY_STOP_PATIENCE,
        delta=cfg.EARLY_STOP_DELTA,
        path=best_ckpt_path,
    )

    # ── History ───────────────────────────────────────────────────────
    history: Dict[str, List] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr":         [], "epoch_time": [], "gpu_util": [],
    }

    log.info("Starting training for up to %d epochs on %s", cfg.NUM_EPOCHS, device)
    log.info("AMP: %s  |  grad_clip: %s  |  batch_size: %d",
             cfg.USE_AMP, cfg.GRAD_CLIP_NORM, cfg.BATCH_SIZE)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.perf_counter()

        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = _eval_epoch(model, val_loader, criterion, device)

        elapsed   = time.perf_counter() - t0

        # Scheduler step BEFORE reading LR so logged value matches next epoch
        scheduler.step(val_loss)
        lr_now    = optimizer.param_groups[0]["lr"]
        gpu_util  = _try_get_gpu_util()

        # Record
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)
        history["epoch_time"].append(elapsed)
        history["gpu_util"].append(gpu_util)

        gpu_str = f"  GPU: {gpu_util:.0f}%" if gpu_util is not None else ""
        log.info(
            "Epoch %3d/%d  |  train_loss: %.4f  train_acc: %.4f  "
            "|  val_loss: %.4f  val_acc: %.4f  |  lr: %.2e  |  %.1fs%s",
            epoch, cfg.NUM_EPOCHS,
            train_loss, train_acc,
            val_loss,   val_acc,
            lr_now, elapsed, gpu_str,
        )

        # Early stopping (saves best checkpoint internally)
        if stopper(val_loss, model):
            log.info("Early stopping triggered at epoch %d.", epoch)
            break

    # ── Load best checkpoint ──────────────────────────────────────────
    log.info("Loading best model from %s", best_ckpt_path)
    model.load_state_dict(torch.load(str(best_ckpt_path), map_location=device))

    # ── Save history (model‑specific filename) ────────────────────────
    hist_path = cfg.RESULTS_DIR / f"training_history_{selected_model_name}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Training history saved → %s", hist_path)

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Resolve labels ────────────────────────────────────────────────
    lbl_path = cfg.LABELS_PATH if cfg.LABELS_PATH.exists() else cfg.LABELS_ALT_PATH
    if not lbl_path.exists():
        log.error("Label file not found at %s or %s", cfg.LABELS_PATH, cfg.LABELS_ALT_PATH)
        raise SystemExit(1)

    raw_labels = np.load(str(lbl_path))
    log.info(
        "Raw labels loaded: %s  (DOWN=0: %d  FLAT=1: %d  UP=2: %d)",
        raw_labels.shape,
        (raw_labels == 0).sum(),
        (raw_labels == 1).sum(),
        (raw_labels == 2).sum(),
    )

    # ── Resolve heatmaps ──────────────────────────────────────────────
    if cfg.HEATMAP_DIR.exists():
        train_loader, val_loader, test_loader, mean, std = build_dataloaders(
            raw_labels, heatmap_dir=cfg.HEATMAP_DIR
        )
    else:
        log.warning("Heatmap dir not found — using synthetic data for testing.")
        arr = np.random.rand(len(raw_labels), cfg.IMG_HEIGHT, cfg.IMG_WIDTH).astype(np.float32)
        train_loader, val_loader, test_loader, mean, std = build_dataloaders(
            raw_labels, heatmap_array=arr
        )

    # ── Save normalisation statistics for inference ───────────────────
    norm_stats = {
        "mean": mean.tolist() if isinstance(mean, (np.ndarray, torch.Tensor)) else mean,
        "std":  std.tolist()  if isinstance(std,  (np.ndarray, torch.Tensor)) else std,
    }
    norm_path = cfg.RESULTS_DIR / "norm_stats.json"
    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    log.info("Normalisation stats saved → %s", norm_path)

    # ── Train ─────────────────────────────────────────────────────────
    model, history = train(train_loader, val_loader, model_name=args.model)

    best_ckpt = _get_best_model_path(model_name=args.model)
    log.info("Training complete. Best model → %s", best_ckpt)
    log.info(
        "Final val_loss: %.4f  val_acc: %.4f",
        min(history["val_loss"]),
        max(history["val_acc"]),
    )