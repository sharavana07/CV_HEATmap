"""
dataset.py — PyTorch Dataset & DataLoader factory for CV_HEATmap.

Design decisions
────────────────
* Lazy loading: heatmaps are read from individual .npy files on demand,
  avoiding a monolithic RAM allocation that would OOM on a laptop.
* Temporal split: financial time-series must NOT be shuffled before splitting.
  Shuffling causes look-ahead bias because a validation sample recorded at
  t+Δ would leak future market state into the training window.
* Normalisation: computed from the training split only, then applied to all
  splits — mirrors the real deployment scenario where future statistics are
  unknown.

Index alignment (IMPORTANT — read before touching this file)
──────────────────────────────────────────────────────────────
Heatmap filenames encode the ORIGINAL global index, e.g. hm_000042.npy was
generated from order-book snapshot #42. Raw labels are 3-class:

    0 = DOWN, 1 = FLAT, 2 = UP

FLAT rows are dropped before training (binary CNN: DOWN=0, UP=1). This means
"position in the filtered/binary label array" and "global heatmap index" are
DIFFERENT numbers as soon as even one FLAT sample is removed:

    global_idx   raw_label      binary position   binary_label
    0            DOWN (0)       0                 0
    1            FLAT (1)       — dropped —        —
    2            UP   (2)       1                 1
    3            DOWN (0)       2                 0

A dataset that does `hm_{position:06d}.npy` (position = row in the filtered
label array) will silently load the WRONG heatmap for every sample after the
first dropped FLAT — exactly the bug this file previously had.

The fix: we never again derive a heatmap filename from "position in some
label array". Instead, we build ONE canonical list of
(global_heatmap_index, binary_label) pairs — a `Sample` — a single time,
while raw labels are still 3-class and still in original file order. Every
downstream step (temporal split, normalisation, Dataset.__getitem__) consumes
that list directly. The global heatmap index is carried in the sample tuple
itself, so there is no arithmetic anyone could get wrong later.

Expected on-disk layout
────────────────────────
  src/data/heatmaps/
      hm_000000.npy   # shape (64, 100) or (1, 64, 100) — float32 or uint8
      hm_000001.npy
      ...
      hm_022302.npy
  src/data/labels_final.npy   # shape (22303,) — int64 {0=DOWN, 1=FLAT, 2=UP}

Alternative flat layout
────────────────────────
If all heatmaps are stored in a single array (heatmaps.npy with shape
[N, 64, 100]), set HEATMAP_DIR = None and pass heatmap_array directly to
build_dataloaders(). heatmap_array[global_idx] must still correspond to the
same global_idx used in the raw label array — the array's row order IS its
global index space, exactly like the hm_{idx:06d}.npy filenames are.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from cnn import config as cfg

log = logging.getLogger(__name__)

# Raw 3-class label convention (input to build_binary_samples).
RAW_DOWN, RAW_FLAT, RAW_UP = 0, 1, 2

# Binary label convention (what the CNN actually trains on).
BINARY_DOWN, BINARY_UP = 0, 1

_RAW_TO_BINARY = {RAW_DOWN: BINARY_DOWN, RAW_UP: BINARY_UP}  # RAW_FLAT intentionally absent


# ─────────────────────────────────────────────────────────────────────────────
# Canonical sample pairing: (global_heatmap_index, binary_label)
# ─────────────────────────────────────────────────────────────────────────────

class Sample(NamedTuple):
    """One training example. global_idx is the heatmap's ORIGINAL filename
    index (hm_{global_idx:06d}.npy); label is already binary (0=DOWN, 1=UP).
    Keeping these paired in one tuple makes misalignment structurally
    impossible — there is no second array whose position could be confused
    with global_idx."""
    global_idx: int
    label: int


def build_binary_samples(raw_labels: np.ndarray) -> List[Sample]:
    """
    Convert a full, file-order-aligned array of 3-class raw labels into a
    list of (global_heatmap_index, binary_label) pairs, dropping FLAT rows.

    This is the ONLY place FLAT filtering happens, and it is the ONLY place
    that is allowed to read `raw_labels` by position — because at this point
    position == global heatmap index (raw_labels has not been filtered yet).
    Every function downstream of this one only ever sees Sample tuples, never
    raw_labels itself, so there is nothing left to misalign.

    Original order is preserved (no sorting/shuffling), which is required
    for the temporal split to remain valid.
    """
    samples: List[Sample] = []
    for global_idx, raw_label in enumerate(raw_labels):
        raw_label = int(raw_label)
        if raw_label == RAW_FLAT:
            continue
        samples.append(Sample(global_idx=global_idx, label=_RAW_TO_BINARY[raw_label]))

    log.info(
        "Built %d binary samples from %d raw labels (%d FLAT dropped) — "
        "DOWN: %d  UP: %d",
        len(samples), len(raw_labels), len(raw_labels) - len(samples),
        sum(1 for s in samples if s.label == BINARY_DOWN),
        sum(1 for s in samples if s.label == BINARY_UP),
    )
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class HeatmapDataset(Dataset):
    """
    Loads order-book heatmap images paired with binary direction labels.

    Parameters
    ----------
    samples : Sequence[Sample]
        The (global_heatmap_index, binary_label) pairs belonging to this
        split, in temporal order. This dataset NEVER looks anything up in a
        separate labels array — the label for each item is carried right
        next to the global index that names its heatmap file, so the two
        cannot drift apart.
    heatmap_dir : Path | None
        Directory containing per-sample hm_{global_idx:06d}.npy files.
        Mutually exclusive with ``heatmap_array``.
    heatmap_array : np.ndarray | None
        Pre-loaded array of shape (N_total, H, W), indexed by global index
        (same numbering as the .npy filenames would use).
    mean : float | None
        Per-channel mean for normalisation. Computed lazily if None.
    std : float | None
        Per-channel standard deviation for normalisation.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        heatmap_dir: Optional[Path] = None,
        heatmap_array: Optional[np.ndarray] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ) -> None:
        if heatmap_dir is None and heatmap_array is None:
            raise ValueError("Supply either heatmap_dir or heatmap_array.")
        if heatmap_dir is not None and heatmap_array is not None:
            raise ValueError("Supply only one of heatmap_dir or heatmap_array.")

        self.samples       = list(samples)
        self.heatmap_dir   = heatmap_dir
        self.heatmap_array = heatmap_array
        self.mean          = mean
        self.std           = std

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def _load_raw(self, global_idx: int) -> np.ndarray:
        """Return a single heatmap as a float32 array of shape (H, W)."""
        if self.heatmap_array is not None:
            img = self.heatmap_array[global_idx]
        else:
            path = self.heatmap_dir / f"hm_{global_idx:06d}.npy"
            try:
                # allow_pickle=False prevents loading of malicious pickled objects
                img = np.load(str(path), allow_pickle=False)
            except FileNotFoundError:
                log.error("Missing heatmap file: %s", path)
                raise
            except Exception as exc:
                log.error("Failed to load %s: %s", path, exc)
                raise

        # Squeeze a leading channel dim if stored as (1, H, W)
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]

        if img.ndim != 2:
            raise ValueError(
                f"Unexpected heatmap shape {img.shape} at global_idx {global_idx}. "
                "Expected (H, W) or (1, H, W)."
            )

        return img.astype(np.float32)

    # ------------------------------------------------------------------
    def __getitem__(self, local_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[local_idx]   # (global_idx, binary_label) — paired, never looked up

        # ── Load the heatmap that matches THIS sample's global_idx ─────
        img = self._load_raw(sample.global_idx)          # (H, W)  float32

        # ── Normalise ─────────────────────────────────────────────────
        if self.mean is not None and self.std is not None:
            img = (img - self.mean) / (self.std + 1e-8)

        # ── Add channel dim → (1, H, W) ───────────────────────────────
        img_tensor = torch.from_numpy(img).unsqueeze(0)   # (1, 64, 100)

        # Label comes straight from the sample tuple — no array lookup,
        # so it is mathematically impossible for it to point at the wrong row.
        label_tensor = torch.tensor(sample.label, dtype=torch.long)

        return img_tensor, label_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_mean_std(
    dataset: HeatmapDataset,
) -> Tuple[float, float]:
    """
    Compute exact per-channel mean and std from the FULL training split.

    Using the entire training set gives exact statistics without any sampling
    noise. For a few tens of thousands of heatmaps this is fast enough on a
    laptop.
    """
    pixels: list[np.ndarray] = []
    for i in range(len(dataset)):
        img, _ = dataset[i]
        pixels.append(img.numpy().ravel())   # (H*W,)

    flat  = np.concatenate(pixels)
    mean  = float(flat.mean())
    std   = float(flat.std())
    log.info("Computed normalisation — mean: %.5f  std: %.5f", mean, std)
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Temporal split
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(
    samples: Sequence[Sample] | int,
    train_ratio: float = cfg.TRAIN_RATIO,
    val_ratio:   float = cfg.VAL_RATIO,
) -> Tuple[List[Sample] | List[int], List[Sample] | List[int], List[Sample] | List[int]]:
    """
    Split samples chronologically into train / val / test lists.

    The function accepts either a sequence of samples or a simple integer
    number of rows. The integer form is handy for preprocessing utilities
    that only need index ranges and do not want to build a full sample list.

    Splits on POSITION IN THE SAMPLE LIST (which is already in original
    temporal order, per build_binary_samples), not on global_idx values —
    global_idx values are not contiguous once FLAT rows are dropped, so
    slicing by global_idx would silently drop or duplicate samples near the
    split boundaries. Slicing the samples list itself is always correct
    because each entry already carries its own global_idx.
    """
    n_samples = len(samples) if not isinstance(samples, int) else samples
    n_train = int(n_samples * train_ratio)
    n_val   = int(n_samples * val_ratio)

    if isinstance(samples, int):
        train_samples = list(range(0, n_train))
        val_samples   = list(range(n_train, n_train + n_val))
        test_samples  = list(range(n_train + n_val, n_samples))
    else:
        train_samples = list(samples[0:n_train])
        val_samples   = list(samples[n_train:n_train + n_val])
        test_samples  = list(samples[n_train + n_val:n_samples])

    log.info(
        "Temporal split — train: %d  val: %d  test: %d  (total: %d)",
        len(train_samples), len(val_samples), len(test_samples), n_samples,
    )
    return train_samples, val_samples, test_samples


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    raw_labels: np.ndarray,
    heatmap_dir: Optional[Path] = None,
    heatmap_array: Optional[np.ndarray] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, float, float]:
    """
    Build train / val / test DataLoaders with proper normalisation.

    Parameters
    ----------
    raw_labels : np.ndarray
        FULL, file-order-aligned 3-class label array (0=DOWN, 1=FLAT, 2=UP),
        i.e. raw_labels[i] is the label for hm_{i:06d}.npy / heatmap_array[i].
        This is intentionally the raw array, not a pre-filtered binary one —
        filtering happens exactly once, here, inside build_binary_samples.

    Returns
    -------
    train_loader, val_loader, test_loader, mean, std
    """
    # ── Sanity-check heatmap availability against the raw label array ────
    if heatmap_dir is not None:
        n_heatmaps = len(list(heatmap_dir.glob("hm_*.npy")))
        if n_heatmaps == 0:
            raise FileNotFoundError(f"No hm_*.npy files found in {heatmap_dir}")
    else:
        n_heatmaps = len(heatmap_array)

    n_labels = len(raw_labels)
    if n_heatmaps != n_labels:
        log.warning(
            "Mismatch: %d raw labels vs %d heatmaps. Only global indices "
            "< %d will be used, since raw_labels must stay aligned with "
            "heatmap file indices by position.",
            n_labels, n_heatmaps, min(n_labels, n_heatmaps),
        )
        n_usable = min(n_labels, n_heatmaps)
        raw_labels = raw_labels[:n_usable]

    # ── Build (global_idx, binary_label) pairs ONCE, from raw 3-class labels,
    #    while raw_labels is still in untouched file order ─────────────────
    samples = build_binary_samples(raw_labels)

    train_samples, val_samples, test_samples = temporal_split(samples)

    # ── Build a temporary train dataset to compute normalisation ──────
    train_ds_raw = HeatmapDataset(
        samples=train_samples,
        heatmap_dir=heatmap_dir, heatmap_array=heatmap_array,
    )

    if cfg.NORM_MEAN is not None and cfg.NORM_STD is not None:
        mean, std = cfg.NORM_MEAN, cfg.NORM_STD
        log.info("Using pre-configured normalisation — mean: %.5f  std: %.5f", mean, std)
    else:
        mean, std = compute_mean_std(train_ds_raw)

    # ── Rebuild all splits with normalisation ─────────────────────────
    kwargs = dict(
        heatmap_dir=heatmap_dir,
        heatmap_array=heatmap_array,
        mean=mean,
        std=std,
    )
    train_ds = HeatmapDataset(samples=train_samples, **kwargs)
    val_ds   = HeatmapDataset(samples=val_samples,   **kwargs)
    test_ds  = HeatmapDataset(samples=test_samples,  **kwargs)

    # ── DataLoader settings optimised for RTX 2050 laptop ────────────
    common = dict(
        batch_size         = cfg.BATCH_SIZE,
        num_workers        = cfg.NUM_WORKERS,
        pin_memory         = cfg.PIN_MEMORY,
        persistent_workers = cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS > 0,
        prefetch_factor    = cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None,
    )

    train_loader = DataLoader(train_ds, shuffle=False, **common)  # temporal order kept
    val_loader   = DataLoader(val_ds,   shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common)

    log.info(
        "DataLoaders — train batches: %d  val batches: %d  test batches: %d",
        len(train_loader), len(val_loader), len(test_loader),
    )
    return train_loader, val_loader, test_loader, mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Resolve label file (expects RAW 3-class labels: 0=DOWN, 1=FLAT, 2=UP)
    lbl_path = cfg.LABELS_PATH if cfg.LABELS_PATH.exists() else cfg.LABELS_ALT_PATH
    if not lbl_path.exists():
        log.error("No label file found at %s or %s", cfg.LABELS_PATH, cfg.LABELS_ALT_PATH)
        raise SystemExit(1)

    raw_labels = np.load(str(lbl_path))
    log.info("Raw labels loaded: shape=%s  dtype=%s", raw_labels.shape, raw_labels.dtype)
    log.info(
        "Raw class distribution — DOWN: %d  FLAT: %d  UP: %d",
        (raw_labels == RAW_DOWN).sum(),
        (raw_labels == RAW_FLAT).sum(),
        (raw_labels == RAW_UP).sum(),
    )

    if not cfg.HEATMAP_DIR.exists():
        log.warning(
            "Heatmap directory %s not found — creating a synthetic dataset for testing.",
            cfg.HEATMAP_DIR,
        )
        # Synthetic in-memory array: N × H × W, indexed by global_idx
        heatmap_array = np.random.rand(len(raw_labels), cfg.IMG_HEIGHT, cfg.IMG_WIDTH).astype(np.float32)
        train_loader, val_loader, test_loader, mean, std = build_dataloaders(
            raw_labels, heatmap_array=heatmap_array
        )
    else:
        train_loader, val_loader, test_loader, mean, std = build_dataloaders(
            raw_labels, heatmap_dir=cfg.HEATMAP_DIR
        )

    imgs, lbls = next(iter(train_loader))
    log.info("Sample batch — images: %s  labels: %s", imgs.shape, lbls.shape)
    log.info("Image range after normalisation — min: %.3f  max: %.3f", imgs.min(), imgs.max())
    log.info("dataset.py self-test passed ✓")