"""
preprocess.py — Data validation and preprocessing pipeline for CV_HEATmap.

Responsibilities
─────────────────
  1. Validate that source data files exist and are well-formed.
  2. Accept heatmaps stored in any of three common layouts:
       A) Monolithic array:  heatmaps.npy  shape (N, H, W)  or (N, 1, H, W)
       B) Pre-split arrays:  heatmaps_train.npy / heatmaps_val.npy / heatmaps_test.npy
       C) Already-split per-sample files in src/data/heatmaps/0.npy … N.npy
  3. Normalise shapes to (H, W) float32.
  4. Write one .npy file per sample into src/data/heatmaps/ so HeatmapDataset
     can use its lazy-loading mode.
  5. Validate label alignment (len(labels) == n_heatmaps).
  6. Compute and cache normalisation statistics (mean / std) from the training
     split, writing them back to config-compatible values.
  7. Print a dataset summary report.

Run this once before train.py if your heatmaps are not yet split into
per-sample files.  It is safe to re-run — existing files are skipped.

Usage
─────
  python src/cnn/preprocess.py                        # auto-detect source
  python src/cnn/preprocess.py --source heatmaps.npy  # explicit source
  python src/cnn/preprocess.py --dry-run              # validate only, no writes
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn import config as cfg
from cnn.dataset import temporal_split

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shape normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _to_hw(arr: np.ndarray, idx: int) -> np.ndarray:
    """
    Normalise a single heatmap array to shape (H, W) float32.

    Accepted input shapes:
      (H, W)      → returned as-is
      (1, H, W)   → channel dim squeezed
      (H, W, 1)   → last dim squeezed
    """
    if arr.ndim == 2:
        pass
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    else:
        raise ValueError(
            f"Sample {idx}: unexpected shape {arr.shape}. "
            "Expected (H, W), (1, H, W), or (H, W, 1)."
        )

    expected = (cfg.IMG_HEIGHT, cfg.IMG_WIDTH)
    if arr.shape != expected:
        raise ValueError(
            f"Sample {idx}: shape {arr.shape} != expected {expected}. "
            "Update IMG_HEIGHT/IMG_WIDTH in config.py if intentional."
        )

    return arr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Source detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_heatmap_source(data_dir: Path = cfg.DATA_DIR) -> Tuple[str, Path]:
    """
    Auto-detect the layout of heatmap data in data_dir.

    Returns
    -------
    (layout, path)
      layout: "monolithic" | "per_sample"
      path  : resolved Path to the source file or directory
    """
    # Priority 1: already-split per-sample directory
    heatmap_dir = data_dir / "heatmaps"
    if heatmap_dir.exists():
        n_files = len(list(heatmap_dir.glob("*.npy")))
        if n_files > 0:
            log.info("Detected per-sample layout: %d files in %s", n_files, heatmap_dir)
            return "per_sample", heatmap_dir

    # Priority 2: monolithic array
    for candidate in ["heatmaps.npy", "heatmaps_all.npy"]:
        p = data_dir / candidate
        if p.exists():
            log.info("Detected monolithic layout: %s", p)
            return "monolithic", p

    raise FileNotFoundError(
        f"No heatmap data found in {data_dir}.\n"
        "Expected one of:\n"
        "  • src/data/heatmaps/0.npy  …  N.npy   (per-sample)\n"
        "  • src/data/heatmaps.npy               (monolithic array)\n"
        "Run ws_capture.py → generate_heatmap.py first."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_labels(labels: np.ndarray, n_heatmaps: int) -> None:
    """Assert that labels are well-formed and align with heatmap count."""
    assert labels.ndim == 1, f"Labels must be 1-D, got shape {labels.shape}"
    assert len(labels) == n_heatmaps, (
        f"Label count ({len(labels)}) != heatmap count ({n_heatmaps})."
    )
    unique = np.unique(labels)
    assert set(unique).issubset({0, 1}), (
        f"Labels should be binary {{0, 1}}, found {unique}."
    )
    n0, n1 = (labels == 0).sum(), (labels == 1).sum()
    ratio  = n1 / len(labels)
    log.info(
        "Labels OK — N=%d  DOWN(0)=%d  UP(1)=%d  UP_ratio=%.3f",
        len(labels), n0, n1, ratio,
    )
    if ratio < 0.35 or ratio > 0.65:
        log.warning(
            "Class imbalance detected (UP ratio=%.3f).  "
            "Consider class-weighted loss or oversampling.", ratio,
        )


def validate_per_sample_dir(heatmap_dir: Path, n_expected: int) -> int:
    """Check that 0.npy … (n_expected-1).npy all exist and have correct shape."""
    missing  = []
    wrong_sh = []

    for i in range(n_expected):
        p = heatmap_dir / f"{i}.npy"
        if not p.exists():
            missing.append(i)
        else:
            arr = np.load(str(p))
            try:
                _to_hw(arr, i)   # shape check only
            except ValueError as e:
                wrong_sh.append((i, str(e)))

    if missing:
        log.error("Missing heatmap files: indices %s … (first 10 shown)", missing[:10])
        raise FileNotFoundError(f"{len(missing)} heatmap files missing.")
    if wrong_sh:
        for i, msg in wrong_sh[:5]:
            log.error("Shape error at %d: %s", i, msg)
        raise ValueError(f"{len(wrong_sh)} heatmaps have wrong shapes.")

    log.info("Per-sample validation passed: %d files, all shapes OK.", n_expected)
    return n_expected


# ─────────────────────────────────────────────────────────────────────────────
# Conversion: monolithic → per-sample
# ─────────────────────────────────────────────────────────────────────────────

def monolithic_to_per_sample(
    source_path: Path,
    out_dir:     Path = cfg.HEATMAP_DIR,
    dry_run:     bool = False,
    skip_existing: bool = True,
) -> int:
    """
    Split a monolithic heatmaps.npy array into per-sample .npy files.

    Parameters
    ----------
    source_path   : Path to the (N, H, W) or (N, 1, H, W) array.
    out_dir       : Destination directory (created if needed).
    dry_run       : If True, validate only — do not write files.
    skip_existing : Skip files that already exist (idempotent re-runs).

    Returns
    -------
    n_written : Number of files written (0 in dry_run mode).
    """
    log.info("Loading monolithic array from %s …", source_path)
    arr = np.load(str(source_path), mmap_mode="r")   # memory-map for RAM safety
    log.info("  shape: %s  dtype: %s", arr.shape, arr.dtype)

    if arr.ndim == 4:
        # (N, 1, H, W) — channel dim present
        assert arr.shape[1] == 1, f"Expected 1 channel, got {arr.shape[1]}"
        n = arr.shape[0]
    elif arr.ndim == 3:
        n = arr.shape[0]
    else:
        raise ValueError(f"Unexpected array ndim={arr.ndim}")

    if dry_run:
        log.info("[DRY RUN] Would write %d files to %s", n, out_dir)
        # Still validate a sample
        _to_hw(np.array(arr[0]), 0)
        log.info("[DRY RUN] Shape validation passed.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0

    for i in range(n):
        out_path = out_dir / f"{i}.npy"
        if skip_existing and out_path.exists():
            continue
        img = _to_hw(np.array(arr[i]), i)
        np.save(str(out_path), img)
        n_written += 1

        if (i + 1) % 1000 == 0 or (i + 1) == n:
            log.info("  Written %d / %d", i + 1, n)

    log.info("Conversion complete — %d files written to %s", n_written, out_dir)
    return n_written


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_and_save_norm_stats(
    labels:       np.ndarray,
    heatmap_dir:  Path = cfg.HEATMAP_DIR,
    sample_limit: int  = 5000,
    out_path:     Optional[Path] = None,
) -> Tuple[float, float]:
    """
    Compute mean/std from the training split and save them to a JSON file.

    Loading a subset of training files is safer than loading all 22 k
    heatmaps into RAM at once.

    Returns
    -------
    (mean, std) — ready to paste into config.py as NORM_MEAN / NORM_STD.
    """
    train_idx, _, _ = temporal_split(len(labels))
    n       = min(len(train_idx), sample_limit)
    rng     = np.random.default_rng(cfg.SEED)
    indices = rng.choice(train_idx, size=n, replace=False)

    pixels: list[np.ndarray] = []
    for i in indices:
        arr = np.load(str(heatmap_dir / f"{int(i)}.npy"))
        pixels.append(arr.ravel().astype(np.float32))

    flat = np.concatenate(pixels)
    mean = float(flat.mean())
    std  = float(flat.std())

    log.info(
        "Normalisation stats (from %d train samples) — mean: %.6f  std: %.6f",
        n, mean, std,
    )
    log.info(
        "→ Set in config.py:  NORM_MEAN = %.6f   NORM_STD = %.6f",
        mean, std,
    )

    # Save to JSON for reference
    if out_path is None:
        out_path = cfg.RESULTS_DIR / "norm_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_path, "w") as f:
        json.dump({"mean": mean, "std": std, "n_samples": int(n)}, f, indent=2)
    log.info("Saved → %s", out_path)

    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Dataset summary report
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_summary(
    labels:      np.ndarray,
    heatmap_dir: Path = cfg.HEATMAP_DIR,
) -> None:
    """Print a concise summary of the dataset to stdout and log."""
    n = len(labels)
    train_idx, val_idx, test_idx = temporal_split(n)

    n0, n1 = (labels == 0).sum(), (labels == 1).sum()
    border = "=" * 60

    print(border)
    print("  CV_HEATmap — Dataset Summary")
    print(border)
    print(f"  Total samples     : {n:,}")
    print(f"  Heatmap shape     : ({cfg.IMG_HEIGHT}, {cfg.IMG_WIDTH})")
    print(f"  Label DOWN (0)    : {n0:,}  ({100*n0/n:.1f}%)")
    print(f"  Label UP   (1)    : {n1:,}  ({100*n1/n:.1f}%)")
    print()
    print(f"  Train split       : {len(train_idx):,}  (idx 0 → {train_idx[-1]})")
    print(f"  Val split         : {len(val_idx):,}  (idx {val_idx[0]} → {val_idx[-1]})")
    print(f"  Test split        : {len(test_idx):,}  (idx {test_idx[0]} → {test_idx[-1]})")
    print()
    if heatmap_dir.exists():
        n_files = len(list(heatmap_dir.glob("*.npy")))
        print(f"  Heatmap files     : {n_files:,}  in {heatmap_dir}")
    else:
        print(f"  Heatmap dir       : NOT FOUND ({heatmap_dir})")
    print(border)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CV_HEATmap preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source", type=Path, default=None,
        help="Path to monolithic heatmaps .npy file (auto-detected if omitted)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate source data without writing files",
    )
    p.add_argument(
        "--skip-norm", action="store_true",
        help="Skip normalisation statistics computation",
    )
    p.add_argument(
        "--validate-only", action="store_true",
        help="Only validate existing per-sample files, no conversion",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    # ── Labels ────────────────────────────────────────────────────────
    lbl_path = cfg.LABELS_PATH if cfg.LABELS_PATH.exists() else cfg.LABELS_ALT_PATH
    if not lbl_path.exists():
        log.error("No label file found at %s or %s", cfg.LABELS_PATH, cfg.LABELS_ALT_PATH)
        sys.exit(1)

    labels = np.load(str(lbl_path))
    log.info("Labels loaded: %s  dtype=%s", labels.shape, labels.dtype)
    validate_labels(labels, len(labels))

    # ── Per-sample validation only ────────────────────────────────────
    if args.validate_only:
        if not cfg.HEATMAP_DIR.exists():
            log.error("Per-sample dir not found: %s", cfg.HEATMAP_DIR)
            sys.exit(1)
        validate_per_sample_dir(cfg.HEATMAP_DIR, len(labels))
        print_dataset_summary(labels)
        return

    # ── Detect or use supplied source ─────────────────────────────────
    if args.source is not None:
        layout, src_path = "monolithic", args.source
    else:
        try:
            layout, src_path = detect_heatmap_source()
        except FileNotFoundError as e:
            log.error(str(e))
            sys.exit(1)

    # ── Conversion ────────────────────────────────────────────────────
    if layout == "monolithic":
        monolithic_to_per_sample(
            source_path=src_path,
            out_dir=cfg.HEATMAP_DIR,
            dry_run=args.dry_run,
        )
    elif layout == "per_sample":
        log.info("Per-sample files already exist — running validation only.")
        validate_per_sample_dir(cfg.HEATMAP_DIR, len(labels))

    # ── Norm stats ────────────────────────────────────────────────────
    if not args.dry_run and not args.skip_norm and cfg.HEATMAP_DIR.exists():
        compute_and_save_norm_stats(labels)

    print_dataset_summary(labels)
    log.info("Preprocessing pipeline complete ✓")


if __name__ == "__main__":
    main()