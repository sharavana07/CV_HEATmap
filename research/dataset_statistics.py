from pathlib import Path
import pandas as pd
import numpy as np

DATASET = Path("src/data/dataset.parquet")   # change if needed

print("="*60)
print("CV_HEATmap Dataset Statistics")
print("="*60)

df = pd.read_parquet(DATASET)

print(f"Total Samples           : {len(df):,}")

if "label" in df.columns:
    print("\nLabel Distribution")
    print(df["label"].value_counts())
    print()
    print((df["label"].value_counts(normalize=True)*100).round(2))

print("\nColumns")
for c in df.columns:
    print("-", c)

print("\nMemory Usage (MB)")
print(df.memory_usage(deep=True).sum()/1024/1024)

stats = []

stats.append(("Total Samples", len(df)))
stats.append(("Features", len(df.columns)))

if "label" in df.columns:
    stats.append(("Classes", df["label"].nunique()))

table = pd.DataFrame(stats, columns=["Metric","Value"])

print("\n")
print(table)

table.to_csv("dataset_statistics.csv", index=False)

print("\nSaved dataset_statistics.csv")