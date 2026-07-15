import numpy as np

labels = np.load("data/labels_final.npy")

unique, counts = np.unique(labels, return_counts=True)

print(dict(zip(unique, counts)))