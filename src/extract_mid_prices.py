"""
Script to compute rolling median mid-prices from order book chunks
and save them as a NumPy array.
"""

from pathlib import Path
import numpy as np

from heatmap.generate_heatmap import (
    load_chunks,
    compute_mid_price,
    TIME_STEPS
)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DATA_DIR = Path("data/raw_orderbook1")
STEP_SIZE = 10                       # stride between consecutive windows

# ------------------------------------------------------------
# 1. Load all chunk files
# ------------------------------------------------------------
print("Discovering parquet chunks...")
files = sorted(DATA_DIR.glob("chunk_*.parquet"))
print(f"Found {len(files)} chunk(s). Loading data...")

df = load_chunks(files)
print(f"Loaded {len(df)} rows of order book data.")

# ------------------------------------------------------------
# 2. Compute median mid-price in sliding windows
# ------------------------------------------------------------
mid_prices = []
max_start = len(df) - TIME_STEPS
num_windows = len(range(0, max_start, STEP_SIZE))

print(f"Processing windows (step={STEP_SIZE}, width={TIME_STEPS})...")
for idx, start_idx in enumerate(range(0, max_start, STEP_SIZE)):
    window = df.iloc[start_idx:start_idx + TIME_STEPS]
    mids = window.apply(compute_mid_price, axis=1)
    mid = mids.median()
    mid_prices.append(mid)

    # Log progress every 10% of windows
    if (idx + 1) % max(1, num_windows // 10) == 0:
        print(f"  Progress: {idx+1}/{num_windows} windows processed.")

# ------------------------------------------------------------
# 3. Save result
# ------------------------------------------------------------
mid_prices = np.array(mid_prices, dtype=np.float32)
out_path = Path("data/mid_prices.npy")
np.save(out_path, mid_prices)

print(f"\nSaved {len(mid_prices)} mid-prices to {out_path}")
print("Done.")