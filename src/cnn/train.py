"""
train.py — Full training loop for OrderBookCNN.

Features
────────
  • CUDA + mixed-precision AMP (torch.cuda.amp) for RTX 2050
  • Gradient clipping
  • ReduceLROnPlateau learning-rate scheduler
  • Early stopping (validation-loss-based)
  • Best-model checkpoint saving (best_model.pth)
  • Per-epoch metrics: loss, accuracy, LR, epoch time
  • GPU utilisation logging (via pynvml if available)
  • Training history saved to outputs/results/training_history.json

Label ownership
────────────────
train.py does NOT do any label filtering, remapping, or index alignment.
dataset.build_dataloaders() owns that entirely: it accepts the ORIGINAL raw
3-class label array (0=DOWN, 1=FLAT, 2=UP), drops FLAT rows, remaps
DOWN/UP -> 0/1, and pairs each surviving label with its correct global
heatmap index internally. train.py only ever loads raw_labels and hands
them off — see the __main__ block below.
"""

from __future__ import annotations

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
from cnn import config as cfg
from cnn.model import OrderBookCNN, print_model_summary
from cnn.dataset import build_dataloaders
from cnn.utils import set_seed, get_device, EarlyStopping

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GPU utilisation (optional — requires pynvml)
# ─────────────────────────────────────────────────────────────────────────────

def _try_get_gpu_util() -> Optional[float]:
    """Return GPU utilisation % or None if pynvml unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return None


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
) -> Tuple[OrderBookCNN, Dict[str, List]]:
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
    model = OrderBookCNN().to(device)
    if cfg.VERBOSE:
        print_model_summary(model)

    # ── Loss / Optimiser ──────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    # FIX 1: Use AdamW for correct weight-decoupling
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
    stopper = EarlyStopping(
        patience=cfg.EARLY_STOP_PATIENCE,
        delta=cfg.EARLY_STOP_DELTA,
        path=cfg.BEST_MODEL_PATH,
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

        # FIX 2: scheduler step BEFORE reading LR so logged value matches next epoch
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
    log.info("Loading best model from %s", cfg.BEST_MODEL_PATH)
    model.load_state_dict(torch.load(str(cfg.BEST_MODEL_PATH), map_location=device))

    # ── Save history ──────────────────────────────────────────────────
    hist_path = cfg.RESULTS_DIR / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Training history saved → %s", hist_path)

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
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

    # Raw, 3-class, file-order-aligned labels (0=DOWN, 1=FLAT, 2=UP).
    # No filtering or remapping here — dataset.build_dataloaders() owns that
    # (it needs the untouched array so global heatmap indices stay correct
    # once FLAT rows are dropped internally).
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
    model, history = train(train_loader, val_loader)

    log.info("Training complete. Best model → %s", cfg.BEST_MODEL_PATH)
    log.info(
        "Final val_loss: %.4f  val_acc: %.4f",
        min(history["val_loss"]),
        max(history["val_acc"]),
    )