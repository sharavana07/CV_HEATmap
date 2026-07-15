"""
logger.py — Structured logging system for CV_HEATmap experiments.

Features
────────
  • Dual-sink: coloured console handler + rotating file handler.
  • All modules call logging.getLogger(__name__) — this module configures
    the root logger so every module is covered automatically.
  • Experiment ID (timestamp-based) stamped on every log line and used
    as a folder prefix for outputs, enabling parallel experiment runs.
  • JSON experiment manifest written at start and updated at end with
    final metrics and elapsed wall-clock time.
  • GPU memory snapshot logged after model creation.
  • Compatible with the existing log = logging.getLogger(__name__) pattern
    used in every other cnn.* module.

Usage
─────
  # At the top of your entry-point script (run_experiment.py / train.py):
  from cnn.logger import setup_logging, ExperimentLogger
  setup_logging()                          # configure root logger
  exp = ExperimentLogger(experiment_name="baseline_v1")
  exp.start()
  ...training...
  exp.finish(metrics={"accuracy": 0.72, "auc": 0.78})

  # All other modules just do:
  import logging
  log = logging.getLogger(__name__)
  log.info("message")                      # automatically captured
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Insert src/ so 'from cnn import config' resolves
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers (console only — stripped from file logs)
# ─────────────────────────────────────────────────────────────────────────────

class _ColourFormatter(logging.Formatter):
    """Add ANSI colours to console log lines by level."""

    LEVEL_COLOURS = {
        logging.DEBUG:    "\033[36m",   # cyan
        logging.INFO:     "\033[32m",   # green
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[35m",   # magenta
    }
    RESET = "\033[0m"
    GREY  = "\033[90m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.LEVEL_COLOURS.get(record.levelno, "")
        record.levelname = f"{colour}{record.levelname:<8}{self.RESET}"
        record.name      = f"{self.GREY}{record.name}{self.RESET}"
        return super().format(record)


# ─────────────────────────────────────────────────────────────────────────────
# Root logger setup
# ─────────────────────────────────────────────────────────────────────────────

_LOG_DIR = cfg.OUTPUT_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_CONFIGURED = False   # guard against double-initialisation


def setup_logging(
    level:       int  = logging.INFO,
    log_dir:     Path = _LOG_DIR,
    experiment_id: Optional[str] = None,
    max_bytes:   int  = 10 * 1024 * 1024,   # 10 MB per file
    backup_count: int = 5,
) -> Path:
    """
    Configure the root logger with:
      • ColourFormatter on stderr (console).
      • RotatingFileHandler writing plain text to logs/<experiment_id>.log.

    Safe to call multiple times — subsequent calls are no-ops.

    Returns
    -------
    log_path : Path to the active log file.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return log_dir / f"{experiment_id or 'experiment'}.log"

    if experiment_id is None:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = log_dir / f"{experiment_id}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # ── Console handler ───────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(_ColourFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(ch)

    # ── Rotating file handler ─────────────────────────────────────────
    fh = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)   # file captures debug too
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    # Silence noisy third-party loggers
    for noisy in ("matplotlib", "PIL", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).info(
        "Logging initialised — level: %s  file: %s",
        logging.getLevelName(level), log_path,
    )
    return log_path


# ─────────────────────────────────────────────────────────────────────────────
# Experiment tracking
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentLogger:
    """
    Lightweight experiment tracker that writes a JSON manifest.

    The manifest captures:
      • experiment_id, name, start_time
      • git commit hash (if available)
      • Python / PyTorch / CUDA versions
      • GPU name and VRAM
      • All config hyperparameters
      • Training history (written at the end)
      • Final test metrics (written at the end)
      • Total wall-clock duration

    The JSON file lives at:
      outputs/results/experiment_<id>.json

    This is a lightweight alternative to MLflow / W&B — no server needed,
    suitable for a laptop-based research project, and easy to diff in git.

    Parameters
    ----------
    experiment_name : str
        Human-readable label embedded in the manifest filename.
    experiment_id   : str | None
        Timestamp-based ID generated automatically if not supplied.
    """

    def __init__(
        self,
        experiment_name: str = "cv_heatmap_cnn",
        experiment_id:   Optional[str] = None,
    ) -> None:
        self.name = experiment_name
        self.id   = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._start_time: Optional[float] = None
        self._manifest: Dict[str, Any]    = {}
        self._path = cfg.RESULTS_DIR / f"experiment_{self.id}.json"
        self.log   = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Record start metadata and write initial manifest."""
        self._start_time = time.perf_counter()
        self._manifest   = {
            "experiment_id":   self.id,
            "experiment_name": self.name,
            "start_time":      datetime.now().isoformat(),
            "end_time":        None,
            "duration_seconds": None,
            "status":          "running",
            "python_version":  sys.version,
            "pytorch_version": torch.__version__,
            "cuda_version":    torch.version.cuda,
            "gpu":             self._gpu_info(),
            "config":          self._config_snapshot(),
            "history":         {},
            "metrics":         {},
            "git_commit":      self._git_commit(),
        }
        self._write()
        self.log.info(
            "Experiment '%s' started  [ID: %s]", self.name, self.id
        )
        self.log.info("Manifest → %s", self._path)

    # ------------------------------------------------------------------
    def log_history(self, history: Dict[str, list]) -> None:
        """Persist training history into the manifest (call after training)."""
        self._manifest["history"] = history
        self._write()

    # ------------------------------------------------------------------
    def finish(self, metrics: Optional[Dict[str, float]] = None) -> None:
        """Mark experiment as complete, record final metrics and duration."""
        elapsed = time.perf_counter() - (self._start_time or time.perf_counter())
        self._manifest["end_time"]        = datetime.now().isoformat()
        self._manifest["duration_seconds"] = round(elapsed, 2)
        self._manifest["status"]          = "completed"
        if metrics:
            self._manifest["metrics"] = {k: round(float(v), 6) for k, v in metrics.items()}
        self._write()
        self.log.info(
            "Experiment '%s' finished in %.1f s  →  %s",
            self.name, elapsed, self._path,
        )
        if metrics:
            self.log.info(
                "Final metrics: %s",
                "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items()),
            )

    # ------------------------------------------------------------------
    def mark_failed(self, error: str) -> None:
        """Record a failure in the manifest (call inside except block)."""
        self._manifest["status"] = "failed"
        self._manifest["error"]  = error
        self._write()
        self.log.error("Experiment '%s' FAILED: %s", self.name, error)

    # ------------------------------------------------------------------
    def _write(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._manifest, f, indent=2, default=str)

    # ------------------------------------------------------------------
    @staticmethod
    def _gpu_info() -> Dict[str, Any]:
        if not torch.cuda.is_available():
            return {"available": False}
        props = torch.cuda.get_device_properties(0)
        return {
            "available":   True,
            "name":        props.name,
            "vram_mb":     round(props.total_memory / 1024 ** 2),
            "compute":     f"{props.major}.{props.minor}",
            "cuda_cores":  props.multi_processor_count,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _config_snapshot() -> Dict[str, Any]:
        return {
            "BATCH_SIZE":           cfg.BATCH_SIZE,
            "LEARNING_RATE":        cfg.LEARNING_RATE,
            "NUM_EPOCHS":           cfg.NUM_EPOCHS,
            "WEIGHT_DECAY":         cfg.WEIGHT_DECAY,
            "DROPOUT_RATE":         cfg.DROPOUT_RATE,
            "FC_HIDDEN":            cfg.FC_HIDDEN,
            "CONV_BLOCKS":          cfg.CONV_BLOCKS,
            "EARLY_STOP_PATIENCE":  cfg.EARLY_STOP_PATIENCE,
            "LR_PATIENCE":          cfg.LR_PATIENCE,
            "USE_AMP":              cfg.USE_AMP,
            "SEED":                 cfg.SEED,
            "TRAIN_RATIO":          cfg.TRAIN_RATIO,
            "VAL_RATIO":            cfg.VAL_RATIO,
            "IMG_HEIGHT":           cfg.IMG_HEIGHT,
            "IMG_WIDTH":            cfg.IMG_WIDTH,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _git_commit() -> Optional[str]:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, cwd=str(cfg.PROJECT_ROOT),
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self._path


# ─────────────────────────────────────────────────────────────────────────────
# GPU memory logger (call after model.to(device))
# ─────────────────────────────────────────────────────────────────────────────

def log_gpu_memory(label: str = "") -> None:
    """Log current CUDA memory allocation and reservation."""
    if not torch.cuda.is_available():
        return
    alloc   = torch.cuda.memory_allocated()  / 1024 ** 2
    reserved = torch.cuda.memory_reserved()  / 1024 ** 2
    total    = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    logging.getLogger(__name__).info(
        "GPU memory%s — allocated: %.0f MB  reserved: %.0f MB  total: %.0f MB",
        f" [{label}]" if label else "",
        alloc, reserved, total,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log_path = setup_logging(level=logging.DEBUG, experiment_id="selftest")
    log = logging.getLogger(__name__)
    log.debug("debug message")
    log.info("info message")
    log.warning("warning message")

    exp = ExperimentLogger(experiment_name="selftest_run", experiment_id="selftest")
    exp.start()
    log_gpu_memory("after model load")
    exp.log_history({"train_loss": [0.7, 0.6], "val_loss": [0.75, 0.65]})
    exp.finish(metrics={"accuracy": 0.72, "auc": 0.78})

    assert exp.manifest_path.exists(), "Manifest not written"
    log.info("logger.py self-test passed ✓  [log file: %s]", log_path)