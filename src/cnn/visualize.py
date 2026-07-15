"""
visualize.py — CNN interpretability and training visualisation for CV_HEATmap.

Implements
──────────
  GradCAM          : Class Activation Map via gradient-weighted feature maps.
                     Shows WHICH regions of the order-book the CNN attends to.
  FeatureMapViewer : Intermediate feature map grids per conv block.
  FilterVisualizer : Learned conv filter weights as a grid image.
  plot_sample_grid : Overlay predictions on raw heatmap images.

Why Grad-CAM for order-book heatmaps?
──────────────────────────────────────
  Grad-CAM localises the price levels and time depths that most influenced
  the prediction.  For a quantitative researcher this answers: "Is the model
  looking at the bid-ask spread region, large limit orders, or noise?"  This
  is critical for trusting the model and explaining it in a research paper.

  Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from
  Deep Networks via Gradient-based Localization." ICCV.

All plots saved to outputs/plots/.

Usage
─────
  python src/cnn/visualize.py                    # runs all visualisations
  python src/cnn/visualize.py --n-samples 8      # CAM on 8 test samples
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn import config as cfg
from cnn.model import OrderBookCNN
from cnn.utils import get_device

log = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi":          120,
    "font.family":         "DejaVu Sans",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
})
LABEL_NAMES = {0: "DOWN", 1: "UP"}
COLORS = {"train": "#2196F3", "val": "#FF5722", "test": "#4CAF50", "cam": "jet"}


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for OrderBookCNN.

    Hooks into the last convolutional layer of `model.features` to capture
    both the forward activation and the gradient of the target class score
    with respect to that activation.

    The CAM is upsampled to the input resolution (64×100) so it can be
    overlaid directly on the heatmap image.

    Parameters
    ----------
    model      : Trained OrderBookCNN.
    target_layer : The nn.Module to hook into.
                   Defaults to the last ConvBlock in model.features.
    """

    def __init__(
        self,
        model:        nn.Module,
        target_layer: Optional[nn.Module] = None,
    ) -> None:
        self.model  = model
        self.model.eval()

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Default: last sub-block inside the last ConvBlock's .block Sequential
        if target_layer is None:
            # model.features[-1] is a ConvBlock; .block[-1] is MaxPool2d
            # We want the ReLU output, i.e. .block[2]
            target_layer = list(model.features.children())[-1].block[2]

        self._layer = target_layer
        self._hooks: list = []
        self._register_hooks()

    # ------------------------------------------------------------------
    def _register_hooks(self) -> None:
        def _fwd_hook(module, inp, out):
            self._activations = out.detach().clone()

        def _bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach().clone()

        self._hooks.append(self._layer.register_forward_hook(_fwd_hook))
        self._hooks.append(self._layer.register_full_backward_hook(_bwd_hook))

    # ------------------------------------------------------------------
    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    @torch.enable_grad()
    def __call__(
        self,
        img_tensor: torch.Tensor,
        class_idx:  Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM for a single image tensor.

        Parameters
        ----------
        img_tensor : (1, 1, H, W) float32 tensor (normalised).
        class_idx  : Target class (0=DOWN, 1=UP). Defaults to predicted class.

        Returns
        -------
        cam : (H, W) float32 numpy array in [0, 1], upsampled to input size.
        """
        assert img_tensor.ndim == 4 and img_tensor.shape[0] == 1

        device = next(self.model.parameters()).device
        img    = img_tensor.to(device).requires_grad_(True)

        # Forward
        logits = self.model(img)          # (1, 2)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward for target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Grad-CAM formula
        # weights = global-average-pooled gradients  (C,)
        acts = self._activations[0]      # (C, H', W')
        grds = self._gradients[0]        # (C, H', W')
        weights = grds.mean(dim=(1, 2))  # (C,)

        # Weighted sum of activations
        cam = (weights[:, None, None] * acts).sum(dim=0)  # (H', W')
        cam = F.relu(cam)                                  # keep positives

        # Upsample to input resolution
        cam = cam.unsqueeze(0).unsqueeze(0)                # (1,1,H',W')
        cam = F.interpolate(
            cam,
            size=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_gradcam_grid(
    model:      nn.Module,
    images:     torch.Tensor,         # (N, 1, H, W)
    true_labels: np.ndarray,          # (N,)
    pred_labels: np.ndarray,          # (N,)
    prob_up:    np.ndarray,           # (N,)
    n_cols:     int  = 4,
    out_path:   Path = cfg.PLOT_DIR / "gradcam_grid.png",
) -> None:
    """
    Plot a grid of heatmap images with Grad-CAM overlays.

    Each cell shows: raw heatmap (left) | CAM overlay (right).
    Title: true label / predicted label / P(UP).
    """
    grad_cam = GradCAM(model)
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols * 2,
        figsize=(n_cols * 5, n_rows * 2.8),
        gridspec_kw={"wspace": 0.05, "hspace": 0.45},
    )
    axes = np.array(axes).reshape(n_rows, n_cols * 2)

    for i in range(n):
        row, col = divmod(i, n_cols)
        ax_raw = axes[row, col * 2]
        ax_cam = axes[row, col * 2 + 1]

        img_np = images[i, 0].cpu().numpy()           # (H, W)
        cam    = grad_cam(images[i:i+1], class_idx=1) # always visualise UP class

        true_str = LABEL_NAMES[int(true_labels[i])]
        pred_str = LABEL_NAMES[int(pred_labels[i])]
        correct  = true_labels[i] == pred_labels[i]
        colour   = "green" if correct else "red"

        ax_raw.imshow(img_np, aspect="auto", origin="lower", cmap="viridis")
        ax_raw.set_title(f"True: {true_str}", fontsize=7)
        ax_raw.axis("off")

        ax_cam.imshow(img_np, aspect="auto", origin="lower", cmap="viridis")
        ax_cam.imshow(cam,    aspect="auto", origin="lower",
                      cmap=COLORS["cam"], alpha=0.45)
        ax_cam.set_title(
            f"Pred: {pred_str} | P↑={prob_up[i]:.2f}",
            fontsize=7, color=colour,
        )
        ax_cam.axis("off")

    # Hide empty cells
    for j in range(n, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row, col * 2].axis("off")
        axes[row, col * 2 + 1].axis("off")

    fig.suptitle("Grad-CAM — Order Book Heatmap Attention (class: UP)",
                 fontsize=12, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    grad_cam.remove_hooks()
    log.info("Grad-CAM grid saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Feature map grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_maps(
    model:     nn.Module,
    img_tensor: torch.Tensor,          # (1, 1, H, W)
    max_channels: int = 16,
    out_path: Path = cfg.PLOT_DIR / "feature_maps.png",
) -> None:
    """
    Visualise the output feature maps of each ConvBlock.

    For each ConvBlock (3 blocks → 3 rows), displays the first
    ``max_channels`` feature maps as a grid of greyscale images.
    This reveals how the CNN progressively abstracts order-book patterns.
    """
    device = next(model.parameters()).device
    img    = img_tensor.to(device)

    # Register forward hooks on every ConvBlock
    block_outputs: list[torch.Tensor] = []

    def _hook(module, inp, out):
        block_outputs.append(out.detach().cpu())

    hooks = []
    for m in model.features.children():
        hooks.append(m.register_forward_hook(_hook))

    with torch.no_grad():
        model.eval()
        model(img)

    for h in hooks:
        h.remove()

    n_blocks = len(block_outputs)
    fig, axes = plt.subplots(
        n_blocks, max_channels,
        figsize=(max_channels * 1.2, n_blocks * 2.0),
        gridspec_kw={"hspace": 0.4, "wspace": 0.1},
    )
    if n_blocks == 1:
        axes = axes[np.newaxis, :]

    for b_idx, fmaps in enumerate(block_outputs):
        n_ch = min(fmaps.shape[1], max_channels)
        for c in range(max_channels):
            ax = axes[b_idx, c]
            if c < n_ch:
                fm = fmaps[0, c].numpy()
                ax.imshow(fm, aspect="auto", origin="lower", cmap="viridis")
            ax.axis("off")
        axes[b_idx, 0].set_ylabel(
            f"Block {b_idx + 1}\n({fmaps.shape[1]} ch)",
            rotation=0, ha="right", va="center", fontsize=8,
        )

    fig.suptitle("Feature Maps per ConvBlock (first 16 channels)",
                 fontsize=12, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Feature maps saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Filter visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_conv_filters(
    model:    nn.Module,
    layer_idx: int  = 0,
    max_filters: int = 32,
    out_path: Path = cfg.PLOT_DIR / "conv_filters.png",
) -> None:
    """
    Display learned convolutional filter weights as a grid.

    Filters from the first ConvBlock are visualised — these are the
    lowest-level features the CNN has learned to detect in the order-book
    heatmap (e.g. price-level edges, order-quantity gradients).

    Parameters
    ----------
    layer_idx : Index into model.features (0 = first ConvBlock).
    """
    conv_blocks = list(model.features.children())
    if layer_idx >= len(conv_blocks):
        log.warning("layer_idx %d out of range (%d blocks)", layer_idx, len(conv_blocks))
        return

    # First child of the ConvBlock's Sequential is nn.Conv2d
    conv = list(conv_blocks[layer_idx].block.children())[0]
    if not isinstance(conv, nn.Conv2d):
        log.warning("Expected Conv2d at block %d, got %s", layer_idx, type(conv))
        return

    weights = conv.weight.detach().cpu().numpy()   # (out_ch, in_ch, kH, kW)
    n_filters = min(weights.shape[0], max_filters)

    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.3, n_rows * 1.3))
    axes = np.array(axes).ravel()

    for i in range(n_filters):
        flt = weights[i, 0]   # (kH, kW) — single input channel
        # Normalise each filter independently for visibility
        flt = (flt - flt.min()) / (flt.max() - flt.min() + 1e-8)
        axes[i].imshow(flt, cmap="RdBu_r", vmin=0, vmax=1)
        axes[i].axis("off")

    for i in range(n_filters, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        f"Conv Filters — Block {layer_idx} "
        f"({weights.shape[0]} filters, {weights.shape[-1]}×{weights.shape[-2]} kernel)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Conv filters saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Sample prediction grid (no CAM)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_grid(
    images:       torch.Tensor,       # (N, 1, H, W)
    true_labels:  np.ndarray,
    pred_labels:  np.ndarray,
    prob_up:      np.ndarray,
    n_cols:       int  = 4,
    out_path:     Path = cfg.PLOT_DIR / "sample_predictions.png",
) -> None:
    """
    Grid of order-book heatmaps annotated with ground-truth and prediction.

    Correct predictions have a green border; wrong ones have red.
    """
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.5, n_rows * 2.2),
        gridspec_kw={"hspace": 0.5, "wspace": 0.1},
    )
    axes = np.array(axes).ravel()

    for i in range(n):
        ax     = axes[i]
        img_np = images[i, 0].cpu().numpy()
        ax.imshow(img_np, aspect="auto", origin="lower", cmap="viridis")

        true_str = LABEL_NAMES[int(true_labels[i])]
        pred_str = LABEL_NAMES[int(pred_labels[i])]
        correct  = true_labels[i] == pred_labels[i]

        for spine in ax.spines.values():
            spine.set_edgecolor("green" if correct else "red")
            spine.set_linewidth(2)
            spine.set_visible(True)

        ax.set_title(
            f"T:{true_str} P:{pred_str}\nP↑={prob_up[i]:.2f}",
            fontsize=7,
            color="green" if correct else "red",
        )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Sample Predictions (green=correct, red=wrong)",
                 fontsize=12, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Sample grid saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Activation statistics
# ─────────────────────────────────────────────────────────────────────────────

def plot_activation_stats(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    n_batches:  int  = 10,
    out_path:   Path = cfg.PLOT_DIR / "activation_stats.png",
) -> None:
    """
    Plot the mean and std of activations after each ConvBlock over N batches.

    Healthy activations should have non-zero mean and reasonable std.
    Dead activations (mean ≈ 0, std ≈ 0) indicate dying ReLU or bad init.
    """
    device  = next(model.parameters()).device
    model.eval()

    block_means: List[List[float]] = [[] for _ in model.features.children()]
    block_stds:  List[List[float]] = [[] for _ in model.features.children()]

    block_list  = list(model.features.children())
    act_buffer: List[Optional[torch.Tensor]] = [None] * len(block_list)

    hooks = []
    for b_idx, blk in enumerate(block_list):
        def _hook(module, inp, out, idx=b_idx):
            act_buffer[idx] = out.detach()
        hooks.append(blk.register_forward_hook(_hook))

    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(loader):
            if batch_idx >= n_batches:
                break
            imgs = imgs.to(device)
            model(imgs)
            for b_idx in range(len(block_list)):
                if act_buffer[b_idx] is not None:
                    t = act_buffer[b_idx]
                    block_means[b_idx].append(t.mean().item())
                    block_stds[b_idx].append(t.std().item())

    for h in hooks:
        h.remove()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(1, len(block_list) + 1)

    means = [np.mean(m) for m in block_means]
    stds  = [np.mean(s) for s in block_stds]

    ax1.bar(x, means, color=cfg.CONV_BLOCKS[i][0] and "#2196F3")
    ax1.set_xlabel("ConvBlock")
    ax1.set_ylabel("Mean activation")
    ax1.set_title("Mean Activation per Block", fontweight="bold")
    ax1.set_xticks(x)

    ax2.bar(x, stds, color="#FF5722")
    ax2.set_xlabel("ConvBlock")
    ax2.set_ylabel("Std activation")
    ax2.set_title("Std Activation per Block", fontweight="bold")
    ax2.set_xticks(x)

    fig.suptitle("Activation Statistics", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Activation stats saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CV_HEATmap visualisation tools")
    p.add_argument("--n-samples", type=int, default=8,
                   help="Number of test samples for Grad-CAM and sample grid")
    p.add_argument("--no-gradcam", action="store_true",
                   help="Skip Grad-CAM (faster, no backward pass needed)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _parse_args()

    device = get_device()
    model  = OrderBookCNN().to(device)

    if not cfg.BEST_MODEL_PATH.exists():
        log.error("No checkpoint at %s — run train.py first.", cfg.BEST_MODEL_PATH)
        sys.exit(1)

    model.load_state_dict(torch.load(str(cfg.BEST_MODEL_PATH), map_location=device))
    model.eval()
    log.info("Loaded model from %s", cfg.BEST_MODEL_PATH)

    # ── Load a few test samples ───────────────────────────────────────
    import numpy as np
    from cnn.dataset import build_dataloaders, temporal_split

    lbl_path = cfg.LABELS_PATH if cfg.LABELS_PATH.exists() else cfg.LABELS_ALT_PATH
    labels   = np.load(str(lbl_path))
    n        = len(labels)
    _, _, test_idx = temporal_split(n)

    if cfg.HEATMAP_DIR.exists():
        _, _, test_loader, mean, std = build_dataloaders(labels, heatmap_dir=cfg.HEATMAP_DIR)
    else:
        log.warning("Heatmap dir not found — using synthetic data.")
        arr = np.random.rand(n, cfg.IMG_HEIGHT, cfg.IMG_WIDTH).astype(np.float32)
        _, _, test_loader, mean, std = build_dataloaders(labels, heatmap_array=arr)

    # Collect N samples
    batch_imgs, batch_true = next(iter(test_loader))
    batch_imgs  = batch_imgs[:args.n_samples]
    batch_true  = batch_true[:args.n_samples].numpy()

    with torch.no_grad():
        logits   = model(batch_imgs.to(device))
        probs    = F.softmax(logits, dim=1).cpu().numpy()
        pred_lbl = logits.argmax(dim=1).cpu().numpy()
    prob_up = probs[:, 1]

    # ── Plots ─────────────────────────────────────────────────────────
    plot_conv_filters(model, layer_idx=0)
    plot_feature_maps(model, batch_imgs[:1])
    plot_sample_grid(batch_imgs, batch_true, pred_lbl, prob_up)

    if not args.no_gradcam:
        plot_gradcam_grid(model, batch_imgs, batch_true, pred_lbl, prob_up)

    plot_activation_stats(model, test_loader)

    log.info("visualize.py complete ✓  →  %s", cfg.PLOT_DIR)