#!/usr/bin/env python3
"""
Complete audit of CV_HEATmap pipeline to find validation-test accuracy gap.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cnn import config as cfg
from cnn.dataset import build_binary_samples, build_dataloaders, temporal_split

print("=" * 80)
print("PHASE 1: DATASET INTEGRITY CHECK")
print("=" * 80)

# Load raw labels
labels_path = cfg.LABELS_PATH if cfg.LABELS_PATH.exists() else cfg.LABELS_ALT_PATH
raw_labels = np.load(str(labels_path))
print(f"✓ Raw labels loaded: {labels_path}")
print(f"  Shape: {raw_labels.shape}")
print(f"  Dtype: {raw_labels.dtype}")
print(f"  Range: [{raw_labels.min()}, {raw_labels.max()}]")
print(f"  Distribution: 0(DOWN)={np.sum(raw_labels==0)}, 1(FLAT)={np.sum(raw_labels==1)}, 2(UP)={np.sum(raw_labels==2)}")

# Check heatmap count
heatmap_files = sorted(cfg.HEATMAP_DIR.glob("hm_*.npy"))
n_heatmaps = len(heatmap_files)
n_labels = len(raw_labels)
print(f"\n✓ Heatmaps found: {n_heatmaps}")
print(f"  Mismatch: {n_heatmaps - n_labels} heatmaps extra")

# Build binary samples
samples = build_binary_samples(raw_labels)
print(f"\n✓ Binary samples created: {len(samples)}")
print(f"  DOWN: {sum(1 for s in samples if s.label==0)}")
print(f"  UP: {sum(1 for s in samples if s.label==1)}")

# Check temporal split correctness
train_samples, val_samples, test_samples = temporal_split(samples)
print(f"\n✓ Temporal split:")
print(f"  Train: {len(train_samples)} samples")
print(f"  Val:   {len(val_samples)} samples")
print(f"  Test:  {len(test_samples)} samples")
print(f"  Total: {len(train_samples) + len(val_samples) + len(test_samples)}")

# Check for ordering
train_global_idx = [s.global_idx for s in train_samples]
val_global_idx = [s.global_idx for s in val_samples]
test_global_idx = [s.global_idx for s in test_samples]

print(f"\n✓ Global index ranges:")
print(f"  Train: [{min(train_global_idx)}, {max(train_global_idx)}]")
print(f"  Val:   [{min(val_global_idx)}, {max(val_global_idx)}]")
print(f"  Test:  [{min(test_global_idx)}, {max(test_global_idx)}]")

# Check temporal ordering
is_train_temporal = train_global_idx == sorted(train_global_idx)
is_val_temporal = val_global_idx == sorted(val_global_idx)
is_test_temporal = test_global_idx == sorted(test_global_idx)
print(f"\n✓ Temporal ordering:")
print(f"  Train in order: {is_train_temporal}")
print(f"  Val in order:   {is_val_temporal}")
print(f"  Test in order:  {is_test_temporal}")

# Check for overlaps
all_train_set = set(train_global_idx)
all_val_set = set(val_global_idx)
all_test_set = set(test_global_idx)

train_val_overlap = all_train_set & all_val_set
train_test_overlap = all_train_set & all_test_set
val_test_overlap = all_val_set & all_test_set

print(f"\n✓ Overlap check:")
print(f"  Train ∩ Val:  {len(train_val_overlap)} samples")
print(f"  Train ∩ Test: {len(train_test_overlap)} samples")
print(f"  Val ∩ Test:   {len(val_test_overlap)} samples")

print("\n" + "=" * 80)
print("PHASE 2: BUILD DATALOADERS")
print("=" * 80)

try:
    train_loader, val_loader, test_loader, mean, std = build_dataloaders(
        raw_labels, heatmap_dir=cfg.HEATMAP_DIR
    )
    print(f"✓ DataLoaders built successfully")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std:  {std:.6f}")
    
    # Check dataset sizes
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"\n✓ Dataset sizes:")
    print(f"  Train: {n_train} samples")
    print(f"  Val:   {n_val} samples")
    print(f"  Test:  {n_test} samples")
    
    # Verify samples are correct
    test_global_from_loader = [s.global_idx for s in test_loader.dataset.samples]
    print(f"\n✓ Test loader global indices:")
    print(f"  Range: [{min(test_global_from_loader)}, {max(test_global_from_loader)}]")
    print(f"  First 5: {test_global_from_loader[:5]}")
    print(f"  Last 5:  {test_global_from_loader[-5:]}")
    
except Exception as e:
    print(f"✗ Error building dataloaders: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("PHASE 3: LABEL LEAKAGE CHECK")
print("=" * 80)

# Check if test labels are peeking at training data
train_labels = np.array([s.label for s in train_loader.dataset.samples])
val_labels = np.array([s.label for s in val_loader.dataset.samples])
test_labels = np.array([s.label for s in test_loader.dataset.samples])

print(f"✓ Label distributions:")
print(f"  Train: 0={np.sum(train_labels==0)}, 1={np.sum(train_labels==1)}")
print(f"  Val:   0={np.sum(val_labels==0)}, 1={np.sum(val_labels==1)}")
print(f"  Test:  0={np.sum(test_labels==0)}, 1={np.sum(test_labels==1)}")

# Check class imbalance
print(f"\n✓ Class balance (ratio UP/DOWN):")
print(f"  Train: {np.sum(train_labels==1) / np.sum(train_labels==0):.3f}")
print(f"  Val:   {np.sum(val_labels==1) / np.sum(val_labels==0):.3f}")
print(f"  Test:  {np.sum(test_labels==1) / np.sum(test_labels==0):.3f}")

print("\n" + "=" * 80)
print("PHASE 4: SAMPLE BATCH CHECK")
print("=" * 80)

# Get one sample from each split (single sample, no multiprocessing)
train_sample_img, train_sample_lbl = train_loader.dataset[0]
val_sample_img, val_sample_lbl = val_loader.dataset[0]
test_sample_img, test_sample_lbl = test_loader.dataset[0]

import torch
print(f"✓ Train sample: image shape={train_sample_img.shape}, label={train_sample_lbl.item()}")
print(f"  Image range: [{train_sample_img.min():.4f}, {train_sample_img.max():.4f}]")

print(f"\n✓ Val sample: image shape={val_sample_img.shape}, label={val_sample_lbl.item()}")
print(f"  Image range: [{val_sample_img.min():.4f}, {val_sample_img.max():.4f}]")

print(f"\n✓ Test sample: image shape={test_sample_img.shape}, label={test_sample_lbl.item()}")
print(f"  Image range: [{test_sample_img.min():.4f}, {test_sample_img.max():.4f}]")

print("\n" + "=" * 80)
print("✓ AUDIT COMPLETE")
print("=" * 80)
