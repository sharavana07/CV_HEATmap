"""
generate_labels.py
==================
Generates classification labels (UP / FLAT / DOWN) from financial mid-price
data stored in a NumPy `.npy` file.

Label encoding
--------------
  0  →  DOWN  (forward return < -threshold)
  1  →  FLAT  (|forward return| ≤ threshold)
  2  →  UP    (forward return >  threshold)

Usage
-----
  # Default settings
  python generate_labels.py

  # Custom paths and parameters
  python generate_labels.py \
      --input  data/mid_prices.npy \
      --output data/labels.npy \
      --lookaheads 10 20 50 \
      --thresholds 0.0003 0.0005 \
      --verbose

  # Emit a CSV with timestamps alongside labels
  python generate_labels.py --timestamps data/timestamps.npy --csv-output data/labels.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT_PATH = "data/mid_prices.npy"
DEFAULT_OUTPUT_PATH = "data/labels.npy"
DEFAULT_LOOKAHEAD = 20
# DEFAULT_THRESHOLD = 0.0005  # 0.05 %
DEFAULT_THRESHOLD = 0.0001 # 0.01 %

CLASS_NAMES = {0: "DOWN", 1: "FLAT", 2: "UP"}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    """Configure root logger verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


# ---------------------------------------------------------------------------
# Data I/O and validation
# ---------------------------------------------------------------------------

def load_mid_prices(path: str | Path) -> np.ndarray:
    """
    Load and validate mid-price data from a NumPy `.npy` file.

    Parameters
    ----------
    path:
        Path to the `.npy` file containing a 1-D array of mid prices.

    Returns
    -------
    np.ndarray
        Validated 1-D float64 array of mid prices (NaN rows dropped with a
        warning).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the array is not 1-D, not numeric, or is empty after NaN removal.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    logger.debug("Loading mid prices from '%s' …", path)
    prices: np.ndarray = np.load(path, allow_pickle=False)

    if prices.ndim != 1:
        raise ValueError(
            f"Expected a 1-D array, got shape {prices.shape}. "
            "Flatten the array before passing it to this script."
        )

    if not np.issubdtype(prices.dtype, np.floating) and not np.issubdtype(
        prices.dtype, np.integer
    ):
        raise ValueError(
            f"Array dtype '{prices.dtype}' is not numeric. "
            "Provide float or integer data."
        )

    prices = prices.astype(np.float64)

    nan_count = int(np.isnan(prices).sum())
    if nan_count:
        logger.warning(
            "%d NaN value(s) found; dropping affected indices before labelling.",
            nan_count,
        )
        prices = prices[~np.isnan(prices)]

    if prices.size == 0:
        raise ValueError("Mid-price array is empty after NaN removal.")

    logger.info("Loaded %d mid-price observations.", len(prices))
    return prices


# ---------------------------------------------------------------------------
# Core labelling logic (fully vectorised)
# ---------------------------------------------------------------------------

def compute_labels(
    mid_prices: np.ndarray,
    lookahead: int,
    threshold: float,
) -> np.ndarray:
    """
    Compute forward-return classification labels without a Python loop.

    The forward return at index *i* is::

        ret[i] = (mid_prices[i + lookahead] - mid_prices[i]) / mid_prices[i]

    The label is then:

    * **2 (UP)**   if ret[i] >  +threshold
    * **0 (DOWN)** if ret[i] < -threshold
    * **1 (FLAT)** otherwise

    Only indices ``[0, len(mid_prices) - lookahead)`` produce a label, so the
    output array is shorter than the input by exactly *lookahead* elements.

    Parameters
    ----------
    mid_prices:
        1-D float64 array of mid prices. Must contain no NaNs.
    lookahead:
        Number of bars to look ahead when computing the return.
    threshold:
        Symmetric return threshold (absolute value) for UP / DOWN labelling.

    Returns
    -------
    np.ndarray
        1-D int64 array of class labels with length
        ``max(0, len(mid_prices) - lookahead)``.

    Raises
    ------
    ValueError
        If *lookahead* ≤ 0, *threshold* ≤ 0, or if the price array is too
        short to produce even a single label.
    """
    if lookahead <= 0:
        raise ValueError(f"lookahead must be a positive integer, got {lookahead}.")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}.")

    n = len(mid_prices)
    n_labels = n - lookahead

    if n_labels <= 0:
        raise ValueError(
            f"Array length ({n}) is too short for lookahead={lookahead}. "
            "Need at least lookahead+1 observations."
        )

    # Vectorised forward-return computation (no Python loop)
    current = mid_prices[:n_labels]           # shape (n_labels,)
    future  = mid_prices[lookahead:n]         # shape (n_labels,)
    returns = (future - current) / current    # shape (n_labels,)

    # Classify using NumPy conditions (still no Python loop)
    labels = np.ones(n_labels, dtype=np.int64)        # default: FLAT = 1
    labels[returns >  threshold] = 2                  # UP
    labels[returns < -threshold] = 0                  # DOWN

    logger.debug(
        "Labelled %d observations (lookahead=%d, threshold=%.4f%%)",
        n_labels,
        lookahead,
        threshold * 100,
    )
    return labels


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_distribution(labels: np.ndarray, lookahead: int, threshold: float) -> None:
    """
    Log the class distribution and basic statistics for a label array.

    Parameters
    ----------
    labels:
        1-D int64 label array produced by :func:`compute_labels`.
    lookahead:
        Lookahead used (for display only).
    threshold:
        Threshold used (for display only).
    """
    total = len(labels)
    unique, counts = np.unique(labels, return_counts=True)

    logger.info(
        "─── Label Distribution  [lookahead=%d, threshold=%.4f%%] ───",
        lookahead,
        threshold * 100,
    )
    for cls, cnt in zip(unique, counts):
        pct = 100.0 * cnt / total
        logger.info("  %-6s  %7d  (%5.1f %%)", CLASS_NAMES.get(cls, str(cls)), cnt, pct)
    logger.info("  %-6s  %7d  (100.0 %%)", "TOTAL", total)


# ---------------------------------------------------------------------------
# Multi-horizon / multi-threshold convenience wrapper
# ---------------------------------------------------------------------------

def generate_all_labels(
    mid_prices: np.ndarray,
    lookaheads: Sequence[int],
    thresholds: Sequence[float],
    output_dir: Path,
    base_stem: str = "labels",
) -> dict[tuple[int, float], np.ndarray]:
    """
    Generate and save labels for every combination of *lookaheads* and
    *thresholds*.

    Output files are named::

        <output_dir>/<base_stem>_L<lookahead>_T<threshold_bps>bps.npy

    Parameters
    ----------
    mid_prices:
        Validated mid-price array.
    lookaheads:
        Sequence of lookahead values to iterate over.
    thresholds:
        Sequence of threshold values to iterate over.
    output_dir:
        Directory where label files are saved.
    base_stem:
        Prefix for output file names.

    Returns
    -------
    dict
        Mapping of ``(lookahead, threshold)`` → label array for all
        successfully computed combinations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[int, float], np.ndarray] = {}

    total_combos = len(lookaheads) * len(thresholds)
    logger.info(
        "Generating labels for %d combination(s) …", total_combos
    )

    for lookahead in lookaheads:
        for threshold in thresholds:
            try:
                labels = compute_labels(mid_prices, lookahead, threshold)
            except ValueError as exc:
                logger.warning(
                    "Skipping (lookahead=%d, threshold=%.6f): %s",
                    lookahead, threshold, exc,
                )
                continue

            threshold_bps = round(threshold * 10_000)
            stem = f"{base_stem}_L{lookahead}_T{threshold_bps}bps"
            out_path = output_dir / f"{stem}.npy"
            np.save(out_path, labels)
            logger.info("Saved → %s", out_path)

            report_distribution(labels, lookahead, threshold)
            results[(lookahead, threshold)] = labels

    return results


# ---------------------------------------------------------------------------
# Optional CSV output with timestamps
# ---------------------------------------------------------------------------

def save_csv(
    labels: np.ndarray,
    output_path: Path,
    timestamps: np.ndarray | None = None,
    lookahead: int = DEFAULT_LOOKAHEAD,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    """
    Save labels (and optionally timestamps) to a CSV file.

    Parameters
    ----------
    labels:
        Label array to export.
    output_path:
        Destination `.csv` path.
    timestamps:
        Optional 1-D array of timestamps aligned to *labels*. If provided,
        the CSV includes a ``timestamp`` column.
    lookahead:
        Lookahead used (written to the CSV header comment).
    threshold:
        Threshold used (written to the CSV header comment).
    """
    import csv

    n = len(labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as fh:
        # Human-readable header comment
        fh.write(
            f"# generated by generate_labels.py | "
            f"lookahead={lookahead} | threshold={threshold:.6f}\n"
        )
        writer = csv.writer(fh)

        if timestamps is not None:
            if len(timestamps) < n:
                raise ValueError(
                    f"Timestamp array length ({len(timestamps)}) is shorter "
                    f"than label array length ({n})."
                )
            writer.writerow(["timestamp", "label", "label_name"])
            for i in range(n):
                writer.writerow(
                    [timestamps[i], int(labels[i]), CLASS_NAMES[int(labels[i])]]
                )
        else:
            writer.writerow(["index", "label", "label_name"])
            for i in range(n):
                writer.writerow([i, int(labels[i]), CLASS_NAMES[int(labels[i])]])

    logger.info("CSV saved → %s  (%d rows)", output_path, n)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate UP/FLAT/DOWN classification labels from mid-price data."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT_PATH,
        help="Path to the input .npy mid-price array.",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Path for the primary output .npy label file "
            "(used only when a single lookahead/threshold is given)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for multi-horizon output files. "
            "Defaults to the directory of --output."
        ),
    )
    parser.add_argument(
        "--timestamps",
        default=None,
        help="Optional .npy file with timestamps (aligned to mid prices).",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional path to write a CSV copy of the (primary) labels.",
    )

    # Label parameters
    parser.add_argument(
        "--lookaheads", "-l",
        nargs="+",
        type=int,
        default=[DEFAULT_LOOKAHEAD],
        metavar="N",
        help="One or more lookahead bar counts.",
    )
    parser.add_argument(
        "--thresholds", "-t",
        nargs="+",
        type=float,
        default=[DEFAULT_THRESHOLD],
        metavar="T",
        help="One or more return thresholds (e.g. 0.0005 for 0.05%%).",
    )

    # Behaviour
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """
    Main entry point.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    # ── Load data ──────────────────────────────────────────────────────────
    try:
        mid_prices = load_mid_prices(args.input)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Data loading failed: %s", exc)
        return 1

    # ── Optional timestamps ────────────────────────────────────────────────
    timestamps: np.ndarray | None = None
    if args.timestamps:
        ts_path = Path(args.timestamps)
        if not ts_path.exists():
            logger.error("Timestamp file not found: %s", ts_path)
            return 1
        timestamps = np.load(ts_path, allow_pickle=True)
        logger.info("Loaded %d timestamps.", len(timestamps))

    # ── Single vs. multi combination ──────────────────────────────────────
    output_path = Path(args.output)
    output_dir  = Path(args.output_dir) if args.output_dir else output_path.parent
    single_mode = len(args.lookaheads) == 1 and len(args.thresholds) == 1

    try:
        if single_mode:
            # Fast path: one combination, save to the explicit --output path
            lookahead = args.lookaheads[0]
            threshold = args.thresholds[0]
            labels = compute_labels(mid_prices, lookahead, threshold)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, labels)
            logger.info("Labels saved → %s", output_path)
            report_distribution(labels, lookahead, threshold)

            # Optional CSV
            if args.csv_output:
                # Align timestamps to labels if provided
                ts_slice = timestamps[:len(labels)] if timestamps is not None else None
                save_csv(
                    labels,
                    Path(args.csv_output),
                    timestamps=ts_slice,
                    lookahead=lookahead,
                    threshold=threshold,
                )
        else:
            # Multi-combination path
            generate_all_labels(
                mid_prices,
                lookaheads=args.lookaheads,
                thresholds=args.thresholds,
                output_dir=output_dir,
                base_stem=output_path.stem,
            )
    except ValueError as exc:
        logger.error("Label generation failed: %s", exc)
        return 1

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())