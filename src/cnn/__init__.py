"""
cnn — CV_HEATmap convolutional neural network package.

Sub-modules
───────────
  config      Central hyperparameters and paths
  dataset     HeatmapDataset, DataLoader factory, temporal split
  model       OrderBookCNN architecture
  train       Training loop (AMP, early stopping, checkpointing)
  evaluate    Metrics, plots, predictions CSV
  inference   Live / batch inference on new heatmaps
  checkpoint  Save / resume full training state
  visualize   Grad-CAM, feature maps, filter visualisation
  preprocess  Raw data validation and .npy conversion pipeline
  logger      Structured file + console logging, experiment tracking
  utils       Shared helpers (EarlyStopping, set_seed, get_device)
"""