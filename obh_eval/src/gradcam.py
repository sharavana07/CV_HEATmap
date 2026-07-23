"""Grad-CAM explainability overlays for correct and incorrect predictions.

Generic implementation that works with any registered CNN (see
model_registry.py) — attaches forward + backward hooks to the layer
returned by GRADCAM_LAYER_GETTERS, computes the class-discriminative
localisation map, and overlays it on the original heatmap.
"""

from __future__ import annotations

import logging

import numpy as np

from . import config, style
from .io_utils import Experiment, load_sample_array
from . import model_registry as reg

log = logging.getLogger("obh_eval")

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def _to_tensor(arr: np.ndarray):
    arr = arr.astype("float32")
    if arr.max() > 1.5:
        arr = arr / 255.0
    if arr.ndim == 2:
        arr = arr[None, None, :, :]
    elif arr.ndim == 3:
        arr = arr[None, :, :, :]
    t = torch.from_numpy(arr)
    t.requires_grad_(True)
    return t


def _gradcam(model, layer, input_tensor, class_idx: int) -> np.ndarray:
    activations = {}
    gradients = {}

    def fwd_hook(_m, _i, out):
        activations["value"] = out

    def bwd_hook(_m, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    logits = model(input_tensor)
    score = logits[0, class_idx]
    score.backward()

    h1.remove()
    h2.remove()

    acts = activations["value"][0].detach()          # (C, H, W)
    grads = gradients["value"][0].detach()            # (C, H, W)
    weights = grads.mean(dim=(1, 2))                   # (C,)
    cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
    cam = cam / (cam.max() + 1e-8)
    return cam.numpy()


def _overlay_and_save(arr: np.ndarray, cam: np.ndarray, title: str, out_path) -> None:
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    if cam.shape != arr.shape[:2]:
        zoom_factors = (arr.shape[0] / cam.shape[0], arr.shape[1] / cam.shape[1])
        cam = zoom(cam, zoom_factors, order=1)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    axes[0].imshow(arr, cmap="gray")
    axes[0].set_title("Input Heatmap", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(arr, cmap="gray")
    axes[1].imshow(cam, cmap="jet", alpha=0.45)
    axes[1].set_title("Grad-CAM Overlay", fontsize=10)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    style.savefig(fig, out_path)


def generate(experiments: list[Experiment]) -> None:
    if torch is None:
        log.warning("torch is not installed — skipping all Grad-CAM visualisations. "
                    "Install torch and re-run to enable this section.")
        return

    style.apply_style()
    base_out = config.OUTPUT_SUBDIRS["gradcam"]

    for exp in experiments:
        model = reg.build_model(exp.name, exp.checkpoint_path)
        if model is None:
            log.info("[%s] no registered model / checkpoint — skipping Grad-CAM. "
                      "Register this model in src/model_registry.py to enable.", exp.name)
            continue
        if exp.samples_dir is None or not exp.has_predictions:
            log.info("[%s] no sample heatmaps available — skipping Grad-CAM.", exp.name)
            continue
        layer_getter = reg.GRADCAM_LAYER_GETTERS.get(exp.name)
        if layer_getter is None:
            log.info("[%s] no Grad-CAM target layer registered — skipping.", exp.name)
            continue
        target_layer = layer_getter(model)

        preds = exp.predictions
        correct = preds[preds["true_label"] == preds["predicted_label"]]
        incorrect = preds[preds["true_label"] != preds["predicted_label"]]

        out_dir = base_out / exp.name
        for label, subset in (("correct", correct), ("incorrect", incorrect)):
            n_done = 0
            for _, row in subset.iterrows():
                if n_done >= 4:
                    break
                arr = load_sample_array(exp, int(row["sample_index"]))
                if arr is None:
                    continue
                tensor = _to_tensor(arr)
                cam = _gradcam(model, target_layer, tensor, class_idx=int(row["predicted_label"]))
                title = (f"{exp.name} — sample {int(row['sample_index'])} "
                         f"(GT={int(row['true_label'])}, Pred={int(row['predicted_label'])})")
                _overlay_and_save(arr, cam, title, out_dir / f"{label}_sample_{int(row['sample_index'])}.png")
                n_done += 1
            if n_done == 0:
                log.info("[%s] no %s-prediction samples with saved images found — nothing to plot.",
                          exp.name, label)