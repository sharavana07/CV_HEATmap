# plot_heatmap.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from heatmap.generate_heatmap import load_chunks, generate_heatmap
import glob
from pathlib import Path
import matplotlib.pyplot as plt
# Find all parquet files
data_dir = Path("data/raw_orderbook1")
files = sorted(data_dir.glob("chunk_*.parquet"))

if not files:
    print("❌ No data found. Run ws_capture.py first.")
    exit()

# Load everything and take the last 100 rows
df = load_chunks(files)
window = df.tail(100)
print(f"✅ Loaded {len(df)} snapshots, using last {len(window)} for heatmap")

# Generate heatmap
heatmap, mid = generate_heatmap(window)
print(f"Heatmap shape: {heatmap.shape}, mid price: {mid:.2f}")

# Plot with a diverging colormap (shows bid vs ask sign)
plt.figure(figsize=(10, 6))
plt.imshow(heatmap, cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Normalised volume (bids +, asks -)")
plt.xlabel("Time steps (100 ms each)")
plt.ylabel("Price bins (64)")
plt.title(f"BTCUSDT Order Book Heatmap (mid ≈ {mid:.2f})")
plt.tight_layout()
plt.show()