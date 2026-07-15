import numpy as np
from pathlib import Path

heatmaps = sorted(Path("data/heatmaps_npy").glob("*.npy"))

occupied = []

mins = []
maxs = []

for f in heatmaps[:1000]:
    hm = np.load(f)

    occupied.append(np.count_nonzero(np.abs(hm).sum(axis=1)))

    mins.append(hm.min())
    maxs.append(hm.max())

print("Heatmaps:", len(heatmaps))
print("Occupied bins")
print(" Min :", min(occupied))
print(" Mean:", np.mean(occupied))
print(" Max :", max(occupied))

print()
print("Global minimum:", min(mins))
print("Global maximum:", max(maxs))