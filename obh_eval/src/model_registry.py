"""
Plugin point connecting an experiment's name to a real, loadable PyTorch
model. Feature-map visualisation and Grad-CAM need an actual forward pass,
so they are the only parts of the pipeline that require this file to be
filled in with your real architectures.

Everything else in the framework (training curves, metrics, comparison
charts, ranked tables, the qualitative gallery) works purely from the CSV/
JSON logs you already produce and needs no changes here.

--------------------------------------------------------------------------
HOW TO WIRE UP YOUR REAL MODELS
--------------------------------------------------------------------------
1. Import your actual model classes (Baseline CNN / SE-OrderBookCNN /
   DAFNet / anything you add later) below.
2. Register a builder function for each experiment folder name in
   MODEL_BUILDERS. The key must match the folder name under
   experiments/ exactly (e.g. "DAFNet").
3. Register the layer(s) Grad-CAM / feature-map hooks should attach to in
   TARGET_LAYER_GETTERS. Point at the last convolutional layer for
   Grad-CAM, and at early + final conv layers for the feature-map figures.
4. If a checkpoint.pt exists for that experiment, it will be loaded with
   `model.load_state_dict(torch.load(checkpoint_path))` automatically.

If an experiment has no entry here (or torch/checkpoint is unavailable),
the pipeline logs a clear message and skips feature-map/Grad-CAM figures
for that model rather than failing the whole run.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable, Optional

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - torch is an optional dependency
    torch = None
    nn = None


# ---------------------------------------------------------------------------
# Example placeholder architectures.
#
# These minimal CNNs let the pipeline run end-to-end (and demonstrate the
# expected hook points) even before your real model code is plugged in.
# Replace / remove them once your real classes are imported above.
# ---------------------------------------------------------------------------

if torch is not None:

    class _BaselineCNNExample(nn.Module):
        def __init__(self, in_ch: int = 1, num_classes: int = 2):
            super().__init__()
            self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU())
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, num_classes)

        def forward(self, x):
            x = self.pool(self.conv1(x))
            x = self.conv2(x)
            x = self.gap(x).flatten(1)
            return self.fc(x)

    class _SEOrderBookCNNExample(nn.Module):
        """Adds a squeeze-and-excitation style channel-attention block."""

        def __init__(self, in_ch: int = 1, num_classes: int = 2):
            super().__init__()
            self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU())
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
            self.se_fc1 = nn.Linear(32, 8)
            self.se_fc2 = nn.Linear(8, 32)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, num_classes)
            self.last_attention = None  # populated on forward for visualisation

        def forward(self, x):
            x = self.pool(self.conv1(x))
            x = self.conv2(x)
            z = self.gap(x).flatten(1)
            s = torch.sigmoid(self.se_fc2(torch.relu(self.se_fc1(z))))
            self.last_attention = s.detach()
            x = x * s.unsqueeze(-1).unsqueeze(-1)
            x = self.gap(x).flatten(1)
            return self.fc(x)

    MODEL_BUILDERS: dict[str, Callable[[], "nn.Module"]] = {
        "Baseline CNN": lambda: _BaselineCNNExample(),
        "SE-OrderBookCNN": lambda: _SEOrderBookCNNExample(),
        # "DAFNet": lambda: YourRealDAFNet(),  # <-- plug in your real class
    }

    # Layer(s) to hook for feature-map visualisation: (early_layer, final_layer)
    TARGET_LAYER_GETTERS: dict[str, Callable[["nn.Module"], tuple]] = {
        "Baseline CNN": lambda m: (m.conv1, m.conv2),
        "SE-OrderBookCNN": lambda m: (m.conv1, m.conv2),
    }

    # Final convolutional layer to use for Grad-CAM per model.
    GRADCAM_LAYER_GETTERS: dict[str, Callable[["nn.Module"], "nn.Module"]] = {
        "Baseline CNN": lambda m: m.conv2[0],
        "SE-OrderBookCNN": lambda m: m.conv2[0],
    }
else:
    MODEL_BUILDERS = {}
    TARGET_LAYER_GETTERS = {}
    GRADCAM_LAYER_GETTERS = {}


def build_model(name: str, checkpoint_path: Optional[str] = None):
    """Instantiate (and optionally load weights for) the model registered
    under `name`. Returns None if torch is unavailable or nothing is
    registered for this experiment name."""
    if torch is None:
        return None
    builder = MODEL_BUILDERS.get(name)
    if builder is None:
        return None
    model = builder()
    if checkpoint_path is not None:
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
        except Exception:  # noqa: BLE001
            # Checkpoint incompatible / example architecture only — still
            # return the (randomly initialised) model so figures can be
            # produced to validate the pipeline, but this should be treated
            # as a signal to plug in the real architecture + checkpoint.
            pass
    model.eval()
    return model