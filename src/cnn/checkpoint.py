"""
checkpoint.py — Full training-state checkpointing for CV_HEATmap.

Why save more than just model weights?
───────────────────────────────────────
best_model.pth (saved by EarlyStopping) stores only the model state_dict —
enough to run inference.  But to *resume* training after an interruption you
also need:
  • Optimizer state      — momentum buffers, Adam m/v estimates
  • Scheduler state      — current LR, patience counter
  • GradScaler state     — AMP loss-scale history
  • EarlyStopping state  — best_loss, counter, best_epoch
  • Training history     — loss/accuracy lists for all completed epochs
  • Epoch number         — so the loop resumes from the right step
  • Config snapshot      — records the exact hyperparameters used

This module saves all of the above in a single ``full_checkpoint.pth`` file
alongside the model-only ``best_model.pth`` that EarlyStopping manages.

Usage
─────
  # Save at end of every epoch (or every N epochs):
  from cnn.checkpoint import save_checkpoint, load_checkpoint
  save_checkpoint(epoch, model, optimizer, scheduler, scaler, stopper, history)

  # Resume:
  state = load_checkpoint(path)
  model.load_state_dict(state["model"])
  optimizer.load_state_dict(state["optimizer"])
  ...
  start_epoch = state["epoch"] + 1
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import _LRScheduler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn import config as cfg
from cnn.utils import EarlyStopping

log = logging.getLogger(__name__)

# Default path for the full resumable checkpoint
FULL_CKPT_PATH = cfg.CHECKPOINT_DIR / "full_checkpoint.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    epoch:      int,
    model:      nn.Module,
    optimizer:  torch.optim.Optimizer,
    scheduler:  _LRScheduler,
    scaler:     GradScaler,
    stopper:    EarlyStopping,
    history:    Dict[str, List],
    path:       Path = FULL_CKPT_PATH,
    config_snapshot: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save the complete training state to a single .pth file.

    Parameters
    ----------
    epoch            : Last completed epoch number (1-indexed).
    model            : The model being trained.
    optimizer        : Adam / SGD optimiser instance.
    scheduler        : ReduceLROnPlateau instance.
    scaler           : AMP GradScaler instance.
    stopper          : EarlyStopping instance.
    history          : Training metrics dict accumulated so far.
    path             : Output path (default: outputs/checkpoints/full_checkpoint.pth).
    config_snapshot  : Optional dict of hyperparameters to embed in the file.
    """
    if config_snapshot is None:
        config_snapshot = _build_config_snapshot()

    payload = {
        "epoch":           epoch,
        "model":           model.state_dict(),
        "optimizer":       optimizer.state_dict(),
        "scheduler":       scheduler.state_dict(),
        "scaler":          scaler.state_dict(),
        # EarlyStopping plain-Python state (not a PyTorch object)
        "early_stopping": {
            "best_loss":  stopper.best_loss,
            "counter":    stopper.counter,
            "best_epoch": stopper.best_epoch,
            "patience":   stopper.patience,
            "delta":      stopper.delta,
        },
        "history":         history,
        "config":          config_snapshot,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
    log.info("Full checkpoint saved → %s  (epoch %d)", path, epoch)


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(
    path:   Path = FULL_CKPT_PATH,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a full checkpoint saved by save_checkpoint().

    Returns the raw payload dict.  Callers are responsible for calling
    .load_state_dict() on each object individually so they control the
    device placement.

    Example
    -------
    state = load_checkpoint()
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    start_epoch = state["epoch"] + 1

    es_state = state["early_stopping"]
    stopper.best_loss  = es_state["best_loss"]
    stopper.counter    = es_state["counter"]
    stopper.best_epoch = es_state["best_epoch"]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device if device is not None else torch.device("cpu")
    payload = torch.load(str(path), map_location=map_location)
    log.info(
        "Checkpoint loaded ← %s  (epoch %d  best_loss: %.6f)",
        path,
        payload["epoch"],
        payload["early_stopping"]["best_loss"],
    )
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Restore helpers
# ─────────────────────────────────────────────────────────────────────────────

def restore_early_stopping(stopper: EarlyStopping, state: Dict) -> None:
    """Restore EarlyStopping fields from a checkpoint payload dict."""
    es = state["early_stopping"]
    stopper.best_loss  = es["best_loss"]
    stopper.counter    = es["counter"]
    stopper.best_epoch = es["best_epoch"]
    log.info(
        "EarlyStopping restored — best_loss: %.6f  counter: %d  best_epoch: %d",
        stopper.best_loss, stopper.counter, stopper.best_epoch,
    )


def restore_full_state(
    state:     Dict[str, Any],
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: _LRScheduler,
    scaler:    GradScaler,
    stopper:   EarlyStopping,
) -> int:
    """
    Convenience function: restore all training objects from a checkpoint.

    Returns
    -------
    start_epoch : int — the epoch to resume from (last_epoch + 1).
    """
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    restore_early_stopping(stopper, state)
    start_epoch = state["epoch"] + 1
    log.info("Training will resume from epoch %d", start_epoch)
    return start_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint listing / management
# ─────────────────────────────────────────────────────────────────────────────

def list_checkpoints(directory: Path = cfg.CHECKPOINT_DIR) -> List[Path]:
    """Return all .pth files in the checkpoint directory, newest first."""
    ckpts = sorted(Path(directory).glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        log.info("Found %d checkpoint(s) in %s:", len(ckpts), directory)
        for c in ckpts:
            size_mb = c.stat().st_size / 1024 ** 2
            log.info("  %s  (%.1f MB)", c.name, size_mb)
    else:
        log.info("No checkpoints found in %s", directory)
    return ckpts


def checkpoint_summary(path: Path = FULL_CKPT_PATH) -> None:
    """Print a human-readable summary of a saved checkpoint."""
    state = load_checkpoint(path)
    es    = state["early_stopping"]
    hist  = state["history"]

    print("=" * 60)
    print(f"  Checkpoint: {Path(path).name}")
    print("=" * 60)
    print(f"  Last epoch        : {state['epoch']}")
    print(f"  Best val_loss     : {es['best_loss']:.6f}  (epoch {es['best_epoch']})")
    print(f"  Early-stop counter: {es['counter']} / {es['patience']}")
    print(f"  Epochs recorded   : {len(hist.get('train_loss', []))}")
    if hist.get("val_acc"):
        best_acc = max(hist["val_acc"])
        print(f"  Best val_acc      : {best_acc:.4f}")
    print(f"  Config keys       : {list(state.get('config', {}).keys())}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Config snapshot
# ─────────────────────────────────────────────────────────────────────────────

def _build_config_snapshot() -> Dict[str, Any]:
    """Extract serialisable hyperparameters from cfg for embedding in checkpoints."""
    return {
        "BATCH_SIZE":           cfg.BATCH_SIZE,
        "LEARNING_RATE":        cfg.LEARNING_RATE,
        "WEIGHT_DECAY":         cfg.WEIGHT_DECAY,
        "NUM_EPOCHS":           cfg.NUM_EPOCHS,
        "DROPOUT_RATE":         cfg.DROPOUT_RATE,
        "FC_HIDDEN":            cfg.FC_HIDDEN,
        "CONV_BLOCKS":          cfg.CONV_BLOCKS,
        "GRAD_CLIP_NORM":       cfg.GRAD_CLIP_NORM,
        "LR_PATIENCE":          cfg.LR_PATIENCE,
        "LR_FACTOR":            cfg.LR_FACTOR,
        "EARLY_STOP_PATIENCE":  cfg.EARLY_STOP_PATIENCE,
        "EARLY_STOP_DELTA":     cfg.EARLY_STOP_DELTA,
        "SEED":                 cfg.SEED,
        "IMG_HEIGHT":           cfg.IMG_HEIGHT,
        "IMG_WIDTH":            cfg.IMG_WIDTH,
        "USE_AMP":              cfg.USE_AMP,
        "TRAIN_RATIO":          cfg.TRAIN_RATIO,
        "VAL_RATIO":            cfg.VAL_RATIO,
    }


def save_config_json(path: Optional[Path] = None) -> None:
    """Persist a JSON copy of the config snapshot for experiment tracking."""
    path = path or (cfg.RESULTS_DIR / "config_snapshot.json")
    snap = _build_config_snapshot()
    with open(path, "w") as f:
        json.dump(snap, f, indent=2)
    log.info("Config snapshot saved → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    import tempfile
    import torch.optim as optim
    from torch.amp import GradScaler
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from cnn.model import OrderBookCNN
    from cnn.utils import EarlyStopping, get_device

    device = get_device()
    model  = OrderBookCNN().to(device)
    opt    = optim.Adam(model.parameters(), lr=1e-3)
    sched  = ReduceLROnPlateau(opt, mode="min", patience=5)
    scaler = GradScaler("cpu", enabled=False)

    with tempfile.TemporaryDirectory() as tmp:
        p       = Path(tmp) / "full_checkpoint.pth"
        stopper = EarlyStopping(patience=10, path=Path(tmp) / "best.pth")
        history = {"train_loss": [0.7, 0.6], "val_loss": [0.75, 0.65],
                   "train_acc":  [0.5, 0.6], "val_acc":  [0.48, 0.58],
                   "lr": [1e-3, 1e-3], "epoch_time": [5.1, 4.9], "gpu_util": [None, None]}

        # Save
        save_checkpoint(1, model, opt, sched, scaler, stopper, history, path=p)
        assert p.exists()

        # Load & restore
        state = load_checkpoint(p, device=device)
        stopper2 = EarlyStopping(patience=10, path=Path(tmp) / "best2.pth")
        start_ep = restore_full_state(state, model, opt, sched, scaler, stopper2)
        assert start_ep == 2, f"Expected start_epoch=2, got {start_ep}"

        checkpoint_summary(p)

    log.info("checkpoint.py self-test passed ✓")