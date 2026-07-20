import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Data directory (relative to src/)
DATA_DIR = Path("data/raw_orderbook1")

files = sorted(DATA_DIR.glob("chunk_*.parquet"))

if not files:
    print(f"❌ No parquet files found in: {DATA_DIR.resolve()}")
    sys.exit(1)

spreads = []

for file in files:
    df = pd.read_parquet(file)

    # Verify expected columns exist
    required = {"ask_p_0", "bid_p_0"}
    if not required.issubset(df.columns):
        print(f"\n❌ Missing expected columns in {file.name}")
        print("Available columns:")
        print(df.columns.tolist())
        sys.exit(1)

    spread = df["ask_p_0"] - df["bid_p_0"]
    spreads.extend(spread.to_numpy())

spreads = np.array(spreads)

print("\nSpread statistics")
print("-----------------")
print(f"Samples : {len(spreads)}")
print(f"Mean    : {spreads.mean():.8f}")
print(f"Median  : {np.median(spreads):.8f}")
print(f"Min     : {spreads.min():.8f}")
print(f"Max     : {spreads.max():.8f}")
print(f"95%     : {np.percentile(spreads, 95):.8f}")