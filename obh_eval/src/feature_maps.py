"""Feature-map (and attention) visualisation for CNN-based models.

Requires: torch installed, a model registered in model_registry.py for the
experiment's name, and at least one saved sample heatmap. Any of these
being missing is not treated as an error — the figure is skipped with a
log message explaining why, so the rest of the pipeline still completes.
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
except ImportError:
    torch = None


def _to_tensor(arr: np.ndarray):
    arr = arr.astype("float32")
    if arr.max() > 1.5:
        arr = arr / 255.0
    if arr.ndim == 2:
        arr = arr[None, None, :, :]
    elif arr.ndim == 3:
        arr = arr[None, :, :, :]
    return torch.from_numpy(arr)


def _plot_maps(maps: np.ndarray, title: str, out_path) -> None:
    import matplotlib.pyplot as plt

    n = min(maps.shape[0], 8)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i in range(n):
        axes[i].imshow(maps[i], cmap="viridis")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(f"ch {i}", fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    style.savefig(fig, out_path)


def generate(experiments: list[Experiment]) -> None:
    if torch is None:
        log.warning("torch is not installed — skipping all feature-map visualisations. "
                    "Install torch and re-run to enable this section.")
        return

    style.apply_style()
    base_out = config.OUTPUT_SUBDIRS["feature_maps"]

    for exp in experiments:
        model = reg.build_model(exp.name, exp.checkpoint_path)
        if model is None:
            log.info("[%s] no registered model / no torch checkpoint — skipping feature maps. "
                      "Register this model in src/model_registry.py to enable.", exp.name)
            continue
        if exp.samples_dir is None or not exp.has_predictions:
            log.info("[%s] no sample heatmaps available — skipping feature maps.", exp.name)
            continue

        sample_index = int(exp.predictions.iloc[0]["sample_index"])
        arr = load_sample_array(exp, sample_index)
        if arr is None:
            log.warning("[%s] could not load sample #%d — skipping feature maps.", exp.name, sample_index)
            continue

        getter = reg.TARGET_LAYER_GETTERS.get(exp.name)
        if getter is None:
            log.info("[%s] no target layers registered — skipping feature maps.", exp.name)
            continue
        early_layer, final_layer = getter(model)

        activations = {}

        def hook(name):
            def _fn(_module, _inp, out):
                activations[name] = out.detach().numpy()[0]
            return _fn

        h1 = early_layer.register_forward_hook(hook("early"))
        h2 = final_layer.register_forward_hook(hook("final"))
        with torch.no_grad():
            model(_to_tensor(arr))
        h1.remove()
        h2.remove()

        out_dir = base_out / exp.name
        if "early" in activations:
            _plot_maps(activations["early"], f"Early Convolution Feature Maps — {exp.name}",
                       out_dir / "early_feature_maps.png")
        if "final" in activations:
            _plot_maps(activations["final"], f"Final Convolution Feature Maps — {exp.name}",
                       out_dir / "final_feature_maps.png")

        attn = getattr(model, "last_attention", None)
        if attn is not None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5.5, 2.2))
            vals = attn.numpy().flatten()
            ax.bar(range(len(vals)), vals, color=config.PALETTE[2])
            ax.set_xlabel("Channel")
            ax.set_ylabel("Attention Weight")
            ax.set_title(f"Channel Attention Response — {exp.name}")
            style.savefig(fig, out_dir / "attention_response.png")
        else:
            log.info("[%s] model has no attention module (or none exposed as "
                      "`last_attention`) — skipping attention figure.", exp.name)