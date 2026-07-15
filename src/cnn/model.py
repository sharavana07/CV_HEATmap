"""
model.py — OrderBookCNN architecture for heatmap-based price direction prediction.

Architecture rationale
──────────────────────
Order-book heatmaps are 2-D spatial representations where:
  * The vertical axis encodes price levels (local structure = support/resistance).
  * The horizontal axis encodes time depth or order quantity layers (temporal/depth structure).

CNNs are the natural choice because:
  1. Local receptive fields capture bid/ask clustering patterns at nearby price levels.
  2. Translation invariance helps generalise patterns that shift in price space.
  3. Hierarchical feature extraction — shallow layers detect raw order imbalances,
     deeper layers compose them into higher-level market microstructure features.
  4. Parameter efficiency: far fewer parameters than an equivalent MLP on the raw
     64×100 pixel input, reducing overfitting risk.
  5. Proven track record in limit order book research (DeepLOB and derivatives).

Expected performance on RTX 2050
──────────────────────────────────
  VRAM usage   : ~300–500 MB at batch_size=64  (4 GB card → comfortable)
  Training time: ~5–15 s / epoch  →  50 epochs ≈ 5–12 min total

Overfitting risks & mitigations
─────────────────────────────────
  Risk: ~22 k samples is small for a CV model.
  Mitigations applied:
    • BatchNorm after every conv (stabilises activations, slight regularisation).
    • Dropout(0.5) before the output head.
    • Weight decay (L2) in the optimiser (set in config).
    • Global Average Pooling instead of Flatten removes ~75 % of FC parameters.
    • Early stopping + LR decay (configured in train.py).
  Future options: data augmentation (horizontal flip, small jitter), label smoothing.

Future architecture upgrades
─────────────────────────────
  1. CNN + LSTM       : Replace GAP with a sequence of conv feature maps fed into
                        an LSTM to model temporal order-flow dynamics across frames.
  2. CNN + Transformer: Use the spatial feature map as a token sequence for a
                        Transformer encoder — captures long-range dependencies
                        across price levels.
  3. ResNet backbone  : Replace the plain conv stack with residual blocks to allow
                        much deeper networks without vanishing gradients.
  4. Vision Transformer (ViT): Patch the heatmap into 8×8 tokens and apply pure
                        self-attention — state of the art on image benchmarks.
  5. Temporal attention: Stack multiple consecutive heatmaps into a T×H×W tensor
                         and apply 3-D convolutions + attention for multi-frame
                         prediction.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Building block
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2D → BatchNorm → ReLU → MaxPool."""

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,
        pool_size:    int = 2,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,   # 'same' spatial size before pooling
                bias=False,                  # bias redundant when BN follows
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class OrderBookCNN(nn.Module):
    """
    Lightweight CNN for binary classification of order-book heatmaps.

    Input  : (B, 1, 64, 100)  — single-channel grayscale heatmap
    Output : (B, 2)            — logits for [DOWN, UP]

    Architecture
    ────────────
      ConvBlock(1  → 32,  k=3, pool=2)   →  (B, 32,  32, 50)
      ConvBlock(32 → 64,  k=3, pool=2)   →  (B, 64,  16, 25)
      ConvBlock(64 → 128, k=3, pool=2)   →  (B, 128,  8, 12)
      GlobalAveragePooling                →  (B, 128)
      Linear(128 → 256) + ReLU
      Dropout(0.5)
      Linear(256 → 2)

    Global Average Pooling instead of Flatten
    ─────────────────────────────────────────
    GAP averages each feature map to a scalar, producing a (B, C) tensor
    regardless of spatial resolution.  Benefits:
      • Eliminates FC layers that depend on fixed spatial size → robust to
        minor input dimension changes.
      • Dramatically reduces parameter count vs Flatten (128×8×12 → 128).
      • Acts as an implicit regulariser (no spatial weights to overfit).
    """

    def __init__(
        self,
        in_channels:   int = cfg.IN_CHANNELS,
        num_classes:   int = cfg.NUM_CLASSES,
        conv_blocks:   list = cfg.CONV_BLOCKS,
        fc_hidden:     int = cfg.FC_HIDDEN,
        dropout_rate:  float = cfg.DROPOUT_RATE,
    ) -> None:
        super().__init__()

        # ── Conv stem ────────────────────────────────────────────────
        layers: list[nn.Module] = []
        ch = in_channels
        for out_ch, k, pool in conv_blocks:
            layers.append(ConvBlock(ch, out_ch, kernel_size=k, pool_size=pool))
            ch = out_ch
        self.features = nn.Sequential(*layers)

        # ── Global Average Pooling ───────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)   # (B, C, H, W) → (B, C, 1, 1)

        # ── Classifier head ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                          # (B, C)
            nn.Linear(ch, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden, num_classes),
        )

        # Weight initialisation
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
        x = self.features(x)   # conv blocks
        x = self.gap(x)        # (B, C, 1, 1)
        x = self.classifier(x) # (B, num_classes)
        return x

    # ------------------------------------------------------------------
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities — for inference / backtest export."""
        return F.softmax(self.forward(x), dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Model summary utility
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
    print("  OrderBookCNN — Architecture Summary")
    print("=" * 60)
    print(model)
    print("-" * 60)
    print(f"  Device              : {device}")
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Non-trainable params: {total - trainable:,}")
    # Rough VRAM estimate: 4 bytes per float32 param + activations
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

    # Forward pass check
    dummy = torch.randn(cfg.BATCH_SIZE, cfg.IN_CHANNELS, cfg.IMG_HEIGHT, cfg.IMG_WIDTH).to(device)
    with torch.no_grad():
        logits = model(dummy)
        probs  = model.predict_proba(dummy)

    print(f"\nForward pass — input: {dummy.shape}  →  logits: {logits.shape}")
    print(f"Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
    print("model.py self-test passed ✓")