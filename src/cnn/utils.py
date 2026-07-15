"""
utils.py — Shared utilities for the CV_HEATmap CNN training system.

Provides three components imported across the codebase:
  • set_seed(seed)       — full reproducibility (Python / NumPy / PyTorch / CUDA)
  • get_device()         — CUDA-first device selection with cuDNN benchmark flag
  • EarlyStopping        — patience-based early stopping with best-model saving

This module has zero imports from other cnn.* modules to avoid circular deps.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Pin all random sources for reproducible experiments.

    Covers Python built-in random, NumPy, PyTorch CPU and GPU RNGs.
    Sets PYTHONHASHSEED so dict/set ordering is deterministic across processes.

    Note: full reproducibility with multi-threaded DataLoader (num_workers > 0)
    also requires worker_init_fn — handled by get_worker_init_fn() below.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # multi-GPU safety
    # Deterministic conv ops (slight speed cost — disable for production speed runs)
    torch.backends.cudnn.deterministic = False   # keep False for RTX 2050 speed
    log.debug("Random seed set to %d", seed)


def get_worker_init_fn(seed: int = 42):
    """
    Return a DataLoader worker_init_fn that seeds each worker independently.
    Pass as worker_init_fn=get_worker_init_fn() in DataLoader if needed.
    """
    def _init(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Return the best available device and configure cuDNN.

    Priority: CUDA (RTX 2050) → MPS (Apple Silicon) → CPU.

    Sets torch.backends.cudnn.benchmark = True when CUDA is available.
    benchmark=True lets cuDNN auto-select the fastest convolution algorithm
    for the fixed input size (64×100) — typically 10–30 % faster on RTX cards.
    """
    if torch.cuda.is_available():
        # Import cfg lazily to avoid circular import at module load time
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from cnn import config as cfg
            torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
        except Exception:
            torch.backends.cudnn.benchmark = True

        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        log.info(
            "GPU: %s  |  VRAM: %.0f MB  |  CUDA %s  |  cuDNN benchmark: %s",
            props.name,
            props.total_memory / 1024 ** 2,
            torch.version.cuda,
            torch.backends.cudnn.benchmark,
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Device: Apple MPS (GPU acceleration)")
    else:
        device = torch.device("cpu")
        log.warning("CUDA not available — running on CPU (training will be slow).")

    return device


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Monitors validation loss and halts training when improvement stalls.

    Behaviour
    ─────────
    • Every call with a new val_loss is compared against the best seen so far.
    • If the improvement is less than ``delta`` for ``patience`` consecutive
      epochs, ``__call__`` returns True (signal to stop).
    • When a new best is found the model state_dict is saved to ``path``.
    • Designed to be called once per epoch, after validation.

    Parameters
    ----------
    patience : int
        Epochs to wait for improvement before stopping.
    delta : float
        Minimum absolute improvement in val_loss to count as progress.
    path : Path
        Where to save the best model checkpoint (best_model.pth).

    Example
    -------
    stopper = EarlyStopping(patience=10, delta=1e-4, path=cfg.BEST_MODEL_PATH)
    for epoch in range(MAX_EPOCHS):
        ...
        if stopper(val_loss, model):
            break
    """

    def __init__(
        self,
        patience: int  = 10,
        delta:    float = 1e-4,
        path:     Path  = Path("best_model.pth"),
    ) -> None:
        self.patience  = patience
        self.delta     = delta
        self.path      = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.best_loss:  float = float("inf")
        self.counter:    int   = 0
        self.best_epoch: int   = 0

    # ------------------------------------------------------------------
    def __call__(self, val_loss: float, model: nn.Module, epoch: int = 0) -> bool:
        """
        Returns True when training should stop.

        Saves model checkpoint whenever a new best val_loss is reached.
        """
        improved = val_loss < self.best_loss - self.delta

        if improved:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
            self._save(model)
            log.info(
                "EarlyStopping: val_loss improved to %.6f — checkpoint saved.",
                val_loss,
            )
        else:
            self.counter += 1
            log.debug(
                "EarlyStopping: no improvement for %d / %d epochs  "
                "(best: %.6f  current: %.6f)",
                self.counter, self.patience, self.best_loss, val_loss,
            )

        return self.counter >= self.patience

    # ------------------------------------------------------------------
    def _save(self, model: nn.Module) -> None:
        torch.save(model.state_dict(), str(self.path))

    # ------------------------------------------------------------------
    @property
    def status(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, "
            f"counter={self.counter}, best_loss={self.best_loss:.6f}, "
            f"best_epoch={self.best_epoch})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Miscellaneous helpers
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    """Return a dict with total, trainable, and frozen parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def bytes_to_mb(n_bytes: int) -> float:
    return n_bytes / (1024 ** 2)


def vram_usage_mb() -> Optional[float]:
    """Return current CUDA VRAM usage in MB, or None if unavailable."""
    if not torch.cuda.is_available():
        return None
    return bytes_to_mb(torch.cuda.memory_allocated())


def vram_reserved_mb() -> Optional[float]:
    """Return CUDA VRAM reserved by the caching allocator in MB."""
    if not torch.cuda.is_available():
        return None
    return bytes_to_mb(torch.cuda.memory_reserved())


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    set_seed(42)
    log.info("set_seed(42) OK")

    device = get_device()
    log.info("get_device() → %s", device)

    # EarlyStopping smoke test
    import tempfile, torch.nn as nn
    with tempfile.TemporaryDirectory() as tmp:
        m = nn.Linear(4, 2)
        es = EarlyStopping(patience=3, delta=0.01, path=Path(tmp) / "best.pth")
        losses = [1.0, 0.9, 0.85, 0.84, 0.84, 0.84, 0.84]
        for i, loss in enumerate(losses):
            stopped = es(loss, m, epoch=i + 1)
            log.info("Epoch %d  loss=%.3f  counter=%d  stopped=%s",
                     i + 1, loss, es.counter, stopped)
        assert stopped, "EarlyStopping should have triggered"
        assert (Path(tmp) / "best.pth").exists(), "Checkpoint not saved"

    log.info("utils.py self-test passed ✓")