"""
inference.py — Live inference engine for CV_HEATmap.

Use cases
─────────
  1. Single heatmap   : predict UP/DOWN from one .npy file captured live.
  2. Batch directory  : score all .npy files in a folder, write results CSV.
  3. NumPy array      : accept a raw (H, W) array directly (WebSocket pipeline).
  4. Streaming hook   : InferenceEngine.predict() returns a structured dict
                        suitable for injection into the live trading system.

The inference pipeline mirrors the training preprocessing exactly:
  load → validate shape → float32 → normalize (same mean/std as training)
  → add batch/channel dims → forward pass → softmax → threshold

Usage
─────
  # CLI — single file:
  python src/cnn/inference.py --input path/to/heatmap.npy

  # CLI — batch folder:
  python src/cnn/inference.py --input path/to/folder/ --output predictions.csv

  # Python API:
  from cnn.inference import InferenceEngine
  engine = InferenceEngine()
  result = engine.predict_array(my_heatmap_array)
  print(result)   # {"label": 1, "direction": "UP", "prob_up": 0.73, ...}
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn import config as cfg
from cnn.model import OrderBookCNN
from cnn.utils import get_device

log = logging.getLogger(__name__)

LABEL_MAP = {0: "DOWN", 1: "UP"}


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing (mirrors dataset.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(
    arr:  np.ndarray,
    mean: float,
    std:  float,
) -> torch.Tensor:
    """
    Convert a raw (H, W) or (1, H, W) array to a normalised (1, 1, H, W) tensor.

    Mirrors HeatmapDataset.__getitem__ preprocessing so inference is consistent
    with training.  Discrepancy here is the single most common source of
    train/inference skew in production ML systems.
    """
    # Squeeze channel dim if present
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(
            f"Expected (H, W) or (1, H, W) array, got shape {arr.shape}."
        )

    expected = (cfg.IMG_HEIGHT, cfg.IMG_WIDTH)
    if arr.shape != expected:
        raise ValueError(
            f"Heatmap shape {arr.shape} != expected {expected}. "
            "Check IMG_HEIGHT / IMG_WIDTH in config.py."
        )

    arr = arr.astype(np.float32)
    arr = (arr - mean) / (std + 1e-8)             # normalise
    tensor = torch.from_numpy(arr)                 # (H, W)
    tensor = tensor.unsqueeze(0).unsqueeze(0)      # (1, 1, H, W)
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Inference engine
# ─────────────────────────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Self-contained inference engine.

    Loads the trained model and normalisation statistics once, then provides
    fast repeated predictions without reloading weights.

    Parameters
    ----------
    checkpoint_path : Path to best_model.pth (or full_checkpoint.pth).
    norm_stats_path : JSON file containing {"mean": float, "std": float}.
                      If None, mean=0 / std=1 (no normalisation) — only safe
                      if the model was trained without normalisation.
    device          : torch.device (auto-selected if None).
    confidence_threshold : P(UP) threshold above which prediction = UP (1).
    """

    def __init__(
        self,
        checkpoint_path:      Path  = cfg.BEST_MODEL_PATH,
        norm_stats_path:      Optional[Path] = None,
        device:               Optional[torch.device] = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.device    = device or get_device()
        self.threshold = confidence_threshold
        self.model     = self._load_model(checkpoint_path)
        self.mean, self.std = self._load_norm_stats(norm_stats_path)

        log.info(
            "InferenceEngine ready — device: %s  threshold: %.2f  "
            "mean: %.5f  std: %.5f",
            self.device, self.threshold, self.mean, self.std,
        )

    # ------------------------------------------------------------------
    def _load_model(self, path: Path) -> nn.Module:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}\n"
                "Run train.py first to generate best_model.pth."
            )

        model = OrderBookCNN().to(self.device)

        # Handle both model-only and full-checkpoint formats
        payload = torch.load(str(path), map_location=self.device)
        if isinstance(payload, dict) and "model" in payload:
            # full_checkpoint.pth
            model.load_state_dict(payload["model"])
            log.info("Loaded full checkpoint (epoch %d) ← %s", payload.get("epoch", "?"), path)
        else:
            # best_model.pth (state_dict only)
            model.load_state_dict(payload)
            log.info("Loaded model weights ← %s", path)

        model.eval()
        return model

    # ------------------------------------------------------------------
    def _load_norm_stats(
        self,
        path: Optional[Path],
    ) -> tuple[float, float]:
        # 1. Explicit path
        if path is not None and Path(path).exists():
            with open(path) as f:
                stats = json.load(f)
            return float(stats["mean"]), float(stats["std"])

        # 2. Auto-detect norm_stats.json in results dir
        auto = cfg.RESULTS_DIR / "norm_stats.json"
        if auto.exists():
            with open(auto) as f:
                stats = json.load(f)
            log.info("Norm stats loaded ← %s", auto)
            return float(stats["mean"]), float(stats["std"])

        # 3. Config-level constants
        if cfg.NORM_MEAN is not None and cfg.NORM_STD is not None:
            log.info("Using config NORM_MEAN=%.5f  NORM_STD=%.5f", cfg.NORM_MEAN, cfg.NORM_STD)
            return cfg.NORM_MEAN, cfg.NORM_STD

        # 4. Fallback — identity normalisation
        log.warning(
            "No normalisation stats found.  Using mean=0, std=1 (identity). "
            "Run preprocess.py to compute proper stats."
        )
        return 0.0, 1.0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_array(self, arr: np.ndarray) -> Dict[str, Any]:
        """
        Predict price direction from a raw numpy heatmap array.

        Parameters
        ----------
        arr : np.ndarray of shape (H, W) or (1, H, W).

        Returns
        -------
        {
          "label":      int   — 0 (DOWN) or 1 (UP)
          "direction":  str   — "DOWN" or "UP"
          "prob_up":    float — P(UP) in [0, 1]
          "prob_down":  float — P(DOWN) in [0, 1]
          "confidence": float — max(prob_up, prob_down)
          "latency_ms": float — inference latency in milliseconds
        }
        """
        t0     = time.perf_counter()
        tensor = _preprocess(arr, self.mean, self.std).to(self.device)

        logits = self.model(tensor)               # (1, 2)
        probs  = F.softmax(logits, dim=1)[0]      # (2,)
        prob_up   = float(probs[1].item())
        prob_down = float(probs[0].item())

        label = 1 if prob_up >= self.threshold else 0
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "label":      label,
            "direction":  LABEL_MAP[label],
            "prob_up":    round(prob_up,   4),
            "prob_down":  round(prob_down, 4),
            "confidence": round(max(prob_up, prob_down), 4),
            "latency_ms": round(latency_ms, 2),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_file(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load a .npy file and predict."""
        path = Path(path)
        arr  = np.load(str(path))
        result = self.predict_array(arr)
        result["file"] = str(path)
        log.debug(
            "%s → %s  P(UP)=%.4f  (%.1f ms)",
            path.name, result["direction"], result["prob_up"], result["latency_ms"],
        )
        return result

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_batch_files(
        self,
        paths: List[Union[str, Path]],
        batch_size: int = cfg.BATCH_SIZE,
    ) -> List[Dict[str, Any]]:
        """
        Score a list of .npy file paths in batches for throughput.

        Returns a list of result dicts in the same order as ``paths``.
        """
        results:  List[Dict[str, Any]] = []
        tensors:  List[torch.Tensor]   = []
        meta:     List[Path]           = []

        def _flush():
            if not tensors:
                return
            batch = torch.cat(tensors, dim=0).to(self.device)
            logits = self.model(batch)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            for i, p in enumerate(meta):
                pu = float(probs[i, 1])
                pd = float(probs[i, 0])
                lbl = 1 if pu >= self.threshold else 0
                results.append({
                    "file":       str(p),
                    "label":      lbl,
                    "direction":  LABEL_MAP[lbl],
                    "prob_up":    round(pu, 4),
                    "prob_down":  round(pd, 4),
                    "confidence": round(max(pu, pd), 4),
                })
            tensors.clear()
            meta.clear()

        t0 = time.perf_counter()
        for path in paths:
            arr = np.load(str(path))
            tensors.append(_preprocess(arr, self.mean, self.std))
            meta.append(Path(path))
            if len(tensors) >= batch_size:
                _flush()
        _flush()

        total_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "Batch inference — %d files in %.1f ms  (%.2f ms/file)",
            len(results), total_ms, total_ms / max(len(results), 1),
        )
        return results

    # ------------------------------------------------------------------
    def predict_directory(
        self,
        directory: Union[str, Path],
        output_csv: Optional[Path] = None,
        batch_size: int = cfg.BATCH_SIZE,
    ) -> pd.DataFrame:
        """
        Score all .npy files in a directory.

        Returns a DataFrame; optionally saves a CSV.

        Columns
        ───────
        file | label | direction | prob_up | prob_down | confidence
        """
        directory = Path(directory)
        paths     = sorted(directory.glob("*.npy"), key=lambda p: p.stem)
        if not paths:
            raise FileNotFoundError(f"No .npy files found in {directory}")

        log.info("Scoring %d files in %s …", len(paths), directory)
        results = self.predict_batch_files(paths, batch_size=batch_size)
        df      = pd.DataFrame(results)

        if output_csv is not None:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
            log.info("Results saved → %s", output_csv)

        return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CV_HEATmap live inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Single file   : python inference.py --input snap.npy\n"
            "  Batch folder  : python inference.py --input data/heatmaps/ --output out.csv\n"
            "  Custom model  : python inference.py --input snap.npy "
            "--model outputs/checkpoints/full_checkpoint.pth\n"
        ),
    )
    p.add_argument("--input", type=Path, required=True,
                   help=".npy file or directory of .npy files")
    p.add_argument("--model", type=Path, default=cfg.BEST_MODEL_PATH,
                   help="Path to checkpoint (default: best_model.pth)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output CSV path for batch mode")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Confidence threshold for UP prediction (default: 0.5)")
    p.add_argument("--norm-stats", type=Path, default=None,
                   help="Path to norm_stats.json (auto-detected if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    import torch.nn as nn  # needed for type annotation in _load_model
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    engine = InferenceEngine(
        checkpoint_path=args.model,
        norm_stats_path=args.norm_stats,
        confidence_threshold=args.threshold,
    )

    if args.input.is_dir():
        # ── Batch mode ────────────────────────────────────────────────
        df = engine.predict_directory(args.input, output_csv=args.output)
        print(df.to_string(index=False))
        print(f"\nTotal samples: {len(df)}")
        print(df["direction"].value_counts().to_string())

    elif args.input.is_file():
        # ── Single file mode ──────────────────────────────────────────
        result = engine.predict_file(args.input)
        print("\n" + "=" * 45)
        print("  Inference Result")
        print("=" * 45)
        for k, v in result.items():
            print(f"  {k:<15}: {v}")
        print("=" * 45)

    else:
        log.error("--input must be a .npy file or a directory: %s", args.input)
        sys.exit(1)