import numpy as np

labels = np.load("data/labels_final.npy")

print(labels.shape)
print(np.unique(labels, return_counts=True))