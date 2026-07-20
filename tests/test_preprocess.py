import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnn.preprocess import compute_and_save_norm_stats


def test_compute_and_save_norm_stats_uses_temporal_split_inputs(tmp_path):
    labels = np.array([0, 1, 2, 0, 2, 1, 0, 2], dtype=np.int64)
    heatmap_dir = tmp_path

    for i in range(len(labels)):
        np.save(heatmap_dir / f"{i}.npy", np.ones((64, 100), dtype=np.float32))

    mean, std = compute_and_save_norm_stats(labels, heatmap_dir=heatmap_dir, sample_limit=4, out_path=tmp_path / "norm.json")

    assert np.isfinite(mean)
    assert np.isfinite(std)
    assert (tmp_path / "norm.json").exists()
