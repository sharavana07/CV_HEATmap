"""
model_se.py — SE-ResNet-lite OrderBookCNN for heatmap-based price direction prediction.

This is a drop-in replacement for model.py. The public class name, constructor
signature, and forward/predict_proba API are unchanged, so train.py, dataset.py,
and any inference/backtest code that does `from cnn.model import OrderBookCNN`
can instead do `from cnn.model_se import OrderBookCNN` (or be re-pointed at this
module) with zero other changes required.

Why SE-ResNet-lite (recap of the architectural decision)
──────────────────────────────────────────────────────────
Two additions on top of the plain 3-block CNN baseline:

  1. Residual (identity) shortcuts around each conv block.
     - Order-book heatmaps have a lot of near-flat, low-information regions
       (empty price levels away from the mid-price). A residual connection
       lets the network default to "pass the input through" and only learn a
       *correction* on top of that, rather than having to relearn the identity
       mapping from scratch in every block. This generally eases optimization
       and — more importantly for this project — acts as an implicit
       regularizer: the effective hypothesis space includes shallow,
       near-identity solutions, so the model is less prone to fitting the
       ~8.5k-sample training noise floor imposed by bid/ask microstructure
       jitter.
     - Cost: a single 1x1 conv per block when channel counts change (used
       here to match the projection needed anyway for the SE-scaled path),
       negligible parameter/FLOP overhead versus the plain CNN.

  2. Squeeze-and-Excitation (SE) channel attention after each residual block.
     - SE learns a per-channel gate: global-average-pool each feature map to
       a scalar ("squeeze"), pass the C-length vector through a tiny
       bottleneck MLP ("excitation"), and rescale each channel by its learned
       sigmoid weight. This lets the network reweight *which learned filters
       matter for this particular heatmap* — e.g. down-weighting a
       far-from-mid-price channel that mostly encodes empty order-book depth,
       and up-weighting channels that fire on bid/ask imbalance near the
       touch — without adding any spatial parameters and without increasing
       the receptive field (so it doesn't increase capacity to overfit
       spatial arrangement, only channel importance).
     - Cost: two tiny 1x1-equivalent Linear layers per block
       (C -> C/r -> C, r=16 by default), a few thousand parameters, no extra
       conv FLOPs to speak of. This is the cheapest attention mechanism
       available and is the reason it was chosen over CBAM / spatial
       attention / a transformer head for a ~1.5M-heatmap, 4GB-VRAM setup.

Net effect vs the plain CNN baseline
─────────────────────────────────────
  - Same 3 stages, same 32→64→128 channel progression, same GAP head, same
    dropout(0.5) classifier -> parameter count increases by only ~1-2%.
  - Expected to close some of the val/test accuracy gap (72.2% / 61.8% in the
    baseline) by improving what the network attends to per-sample rather than
    by adding raw capacity, which is the stated goal (reduce overfitting,
    don't just make the model bigger).
  - Still comfortably trains within the existing ~300-500MB VRAM / 5-15s per
    epoch budget on an RTX 2050 at batch_size=64.

Nothing else in the pipeline changes: input remains (B, 1, 64, 100), output
remains (B, 2) raw logits for [DOWN, UP], and train.py's label
ownership/index-alignment contract with dataset.py is untouched.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Squeeze-and-Excitation block
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention (Hu et al., 2018).

    Squeeze : global average pool each channel to a scalar -> (B, C)
    Excite  : C -> C/r -> C bottleneck MLP with ReLU + Sigmoid -> per-channel
              gate in (0, 1)
    Scale   : multiply the original feature map by its channel gate

    Parameters are tiny (2 * C * C/r) and there is no added spatial receptive
    field, so this is pure "which channels matter" attention rather than
    "where in the image" attention.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        gate = self.avg_pool(x).view(b, c)      # squeeze -> (B, C)
        gate = self.fc(gate).view(b, c, 1, 1)   # excite  -> (B, C, 1, 1)
        return x * gate                          # scale


# ─────────────────────────────────────────────────────────────────────────────
# Residual + SE building block
# ─────────────────────────────────────────────────────────────────────────────

class ResidualSEBlock(nn.Module):
    """
    Conv -> BN -> ReLU -> Conv -> BN -> SE -> (+ shortcut) -> ReLU -> MaxPool

    The shortcut is:
      - identity, if in_channels == out_channels
      - a 1x1 Conv + BN projection, if channel counts differ (true for every
        block in this network, since channels grow 1->32->64->128)

    Pooling is applied *after* the residual add + activation, matching the
    spatial downsampling schedule of the original plain-CNN baseline
    (pool_size=2 per block, same 64x100 -> 8x12 final map).
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,
        pool_size:    int = 2,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=kernel_size // 2, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=kernel_size, padding=kernel_size // 2, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction=se_reduction)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)          # channel-reweight before residual add
        out = out + identity
        out = self.relu(out)
        out = self.pool(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class OrderBookCNN(nn.Module):
    """
    SE-ResNet-lite CNN for binary classification of order-book heatmaps.

    Input  : (B, 1, 64, 100)  — single-channel grayscale heatmap
    Output : (B, 2)            — logits for [DOWN, UP]

    Architecture
    ────────────
      ResidualSEBlock(1  → 32,  k=3, pool=2)   →  (B, 32,  32, 50)
      ResidualSEBlock(32 → 64,  k=3, pool=2)   →  (B, 64,  16, 25)
      ResidualSEBlock(64 → 128, k=3, pool=2)   →  (B, 128,  8, 12)
      GlobalAveragePooling                       →  (B, 128)
      Linear(128 → 256) + ReLU
      Dropout(0.5)
      Linear(256 → 2)

    This mirrors the original plain-CNN block layout exactly (same channel
    progression, same spatial downsampling), so it is a like-for-like
    architectural swap: every accuracy delta observed against model.py's
    OrderBookCNN comes from the residual + SE additions, not from a change in
    depth, width, or input handling.
    """

    def __init__(
        self,
        in_channels:   int = cfg.IN_CHANNELS,
        num_classes:   int = cfg.NUM_CLASSES,
        conv_blocks:   list = cfg.CONV_BLOCKS,
        fc_hidden:     int = cfg.FC_HIDDEN,
        dropout_rate:  float = cfg.DROPOUT_RATE,
        se_reduction:  int = None,
    ) -> None:
        super().__init__()

        # Allow an optional SE_REDUCTION entry in config.py without requiring
        # it to exist there; falls back to the standard SE-Net default of 16.
        if se_reduction is None:
            se_reduction = getattr(cfg, "SE_REDUCTION", 16)

        # ── Residual + SE conv stem ──────────────────────────────────
        layers: list[nn.Module] = []
        ch = in_channels
        for out_ch, k, pool in conv_blocks:
            layers.append(
                ResidualSEBlock(
                    ch, out_ch,
                    kernel_size=k, pool_size=pool,
                    se_reduction=se_reduction,
                )
            )
            ch = out_ch
        self.features = nn.Sequential(*layers)

        # ── Global Average Pooling ───────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)   # (B, C, H, W) → (B, C, 1, 1)

        # ── Classifier head (unchanged from baseline) ────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                          # (B, C)
            nn.Linear(ch, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden, num_classes),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)    # residual + SE conv blocks
        x = self.gap(x)         # (B, C, 1, 1)
        x = self.classifier(x)  # (B, num_classes)
        return x

    # ------------------------------------------------------------------
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities — for inference / backtest export."""
        return F.softmax(self.forward(x), dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Model summary utility (identical interface to model.py)
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model: nn.Module) -> None:
    total, trainable = count_parameters(model)
    device = next(model.parameters()).device

    print("=" * 60)
    print("  OrderBookCNN (SE-ResNet-lite) — Architecture Summary")
    print("=" * 60)
    print(model)
    print("-" * 60)
    print(f"  Device              : {device}")
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Non-trainable params: {total - trainable:,}")
    vram_mb = (total * 4) / (1024 ** 2)
    print(f"  Param memory (fp32) : {vram_mb:.1f} MB  (activations extra)")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    model = OrderBookCNN().to(device)
    print_model_summary(model)

    dummy = torch.randn(cfg.BATCH_SIZE, cfg.IN_CHANNELS, cfg.IMG_HEIGHT, cfg.IMG_WIDTH).to(device)
    with torch.no_grad():
        logits = model(dummy)
        probs  = model.predict_proba(dummy)

    print(f"\nForward pass — input: {dummy.shape}  →  logits: {logits.shape}")
    print(f"Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
    assert logits.shape == (cfg.BATCH_SIZE, cfg.NUM_CLASSES)
    assert torch.allclose(probs.sum(dim=1), torch.ones(cfg.BATCH_SIZE, device=device), atol=1e-4)
    print("model_se.py self-test passed ✓")