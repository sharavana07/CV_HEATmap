# src/heatmap/generate_heatmap.py
import numpy as np
import pandas as pd

PRICE_BINS = 64       # vertical resolution
TIME_STEPS = 100       # horizontal resolution (window length)
PRICE_RANGE_PCT = 0.00005  # Half-range: ±0.005% around mid-price

def load_chunks(chunk_paths):
    dfs = [pd.read_parquet(p) for p in chunk_paths]
    df = pd.concat(dfs).sort_values("ts").reset_index(drop=True)
    return df

def compute_mid_price(row):
    return (row["bid_p_0"] + row["ask_p_0"]) / 2

def snapshot_to_column(row, mid, bin_edges):
    """Returns a PRICE_BINS-length vector of aggregated volume per price bin."""
    col = np.zeros(PRICE_BINS)
    for i in range(20):
        for side, pcol, qcol in [("bid", f"bid_p_{i}", f"bid_q_{i}"),
                                   ("ask", f"ask_p_{i}", f"ask_q_{i}")]:
            p, q = row.get(pcol), row.get(qcol)
            if p is None or pd.isna(p):
                continue
            idx = np.digitize(p, bin_edges) - 1
            if 0 <= idx < PRICE_BINS:
                if side == "bid":
                    col[idx] += q
                else:
                    col[idx] -= q
    return col

def generate_heatmap(df_window):
    mids = df_window.apply(compute_mid_price, axis=1)
    mid = mids.median()
    lo, hi = mid * (1 - PRICE_RANGE_PCT), mid * (1 + PRICE_RANGE_PCT)
    bin_edges = np.linspace(lo, hi, PRICE_BINS + 1)

    heatmap = np.zeros((PRICE_BINS, TIME_STEPS))
    for t, (_, row) in enumerate(df_window.iterrows()):
        if t >= TIME_STEPS:
            break
        heatmap[:, t] = snapshot_to_column(row, mid, bin_edges)

    # log-scale + normalize for visual contrast
    heatmap = np.sign(heatmap) * np.log1p(np.abs(heatmap))

    return heatmap, mid