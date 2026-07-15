# src/heatmap/build_dataset.py

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from heatmap.generate_heatmap import load_chunks, generate_heatmap, TIME_STEPS

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("data/raw_orderbook1")

NPY_DIR = Path("data/heatmaps_npy")
PNG_DIR = Path("data/heatmaps_png")

NPY_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)

STEP_SIZE = 10

# --------------------------------------------------
# Load data
# --------------------------------------------------

files = sorted(DATA_DIR.glob("chunk_*.parquet"))

if not files:
    print("❌ No parquet files found.")
    exit()

print(f"📂 Found {len(files)} parquet files")

df = load_chunks(files)

print(f"✅ Loaded {len(df)} snapshots")

# --------------------------------------------------
# Generate heatmaps
# --------------------------------------------------

num_saved = 0
metadata = []

max_start = len(df) - TIME_STEPS

for start_idx in range(0, max_start, STEP_SIZE):

    end_idx = start_idx + TIME_STEPS

    window = df.iloc[start_idx:end_idx]

    heatmap, mid = generate_heatmap(window)

    # --------------------------
    # Store metadata
    # --------------------------
    metadata.append({
        "heatmap_file": f"hm_{num_saved:06d}.npy",
        "window_start": start_idx,
        "window_end": end_idx - 1,
        "reference_mid": float(mid)
    })

    # --------------------------
    # Save NPY
    # --------------------------
    npy_file = NPY_DIR / f"hm_{num_saved:06d}.npy"
    np.save(npy_file, heatmap.astype(np.float32))

    # --------------------------
    # Save PNG
    # --------------------------
    png_file = PNG_DIR / f"hm_{num_saved:06d}.png"

    plt.imsave(
        png_file,
        heatmap,
        cmap="hot",
        origin="lower"
    )

    num_saved += 1

    if num_saved % 100 == 0:
        print(f"Saved {num_saved} heatmaps...")

# --------------------------------------------------
# Save metadata
# --------------------------------------------------

metadata_df = pd.DataFrame(metadata)

metadata_path = Path("data/heatmap_metadata.csv")
metadata_df.to_csv(metadata_path, index=False)

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n✅ Dataset generation complete")
print(f"📦 Heatmaps saved: {num_saved}")
print(f"📁 NPY: {NPY_DIR}")
print(f"📁 PNG: {PNG_DIR}")
print(f"📄 Metadata: {metadata_path}")