"""Training behaviour figures: loss, accuracy, learning-rate schedule.

One figure per metric, all models overlaid, so convergence/generalisation
can be compared directly across architectures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from . import config, style
from .io_utils import Experiment

log = logging.getLogger("obh_eval")


def _plot_train_val(experiments: list[Experiment], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=config.FIGSIZE_WIDE)
    any_plotted = False
    for i, exp in enumerate(experiments):
        if not exp.has_training_curves or "train_loss" not in exp.history.columns:
            continue
        h = exp.history
        color = config.PALETTE[i % len(config.PALETTE)]
        ax.plot(h["epoch"] if "epoch" in h.columns else h.index, h["train_loss"],
                color=color, linestyle="-", label=f"{exp.name} (train)")
        if "val_loss" in h.columns:
            ax.plot(h["epoch"] if "epoch" in h.columns else h.index, h["val_loss"],
                     color=color, linestyle="--", label=f"{exp.name} (val)")
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        log.warning("No train/val loss data available — skipping loss curve figure.")
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs. Validation Loss")
    ax.legend(ncol=2, loc="best")
    style.savefig(fig, out_dir / "loss_curves.png")


def _plot_accuracy(experiments: list[Experiment], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=config.FIGSIZE_WIDE)
    any_plotted = False
    for i, exp in enumerate(experiments):
        if not exp.has_training_curves or "train_acc" not in exp.history.columns:
            continue
        h = exp.history
        color = config.PALETTE[i % len(config.PALETTE)]
        ax.plot(h["epoch"] if "epoch" in h.columns else h.index, h["train_acc"],
                color=color, linestyle="-", label=f"{exp.name} (train)")
        if "val_acc" in h.columns:
            ax.plot(h["epoch"] if "epoch" in h.columns else h.index, h["val_acc"],
                     color=color, linestyle="--", label=f"{exp.name} (val)")
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        log.warning("No train/val accuracy data available — skipping accuracy curve figure.")
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs. Validation Accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=2, loc="best")
    style.savefig(fig, out_dir / "accuracy_curves.png")


def _plot_lr_schedule(experiments: list[Experiment], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=config.FIGSIZE_WIDE)
    any_plotted = False
    for i, exp in enumerate(experiments):
        if not exp.has_training_curves or "lr" not in exp.history.columns:
            continue
        h = exp.history
        ax.plot(h["epoch"] if "epoch" in h.columns else h.index, h["lr"],
                color=config.PALETTE[i % len(config.PALETTE)], label=exp.name)
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        log.warning("No learning-rate column found — skipping LR schedule figure.")
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.set_title("Learning Rate Schedule")
    ax.legend(loc="best")
    style.savefig(fig, out_dir / "lr_schedule.png")


def generate(experiments: list[Experiment]) -> None:
    style.apply_style()
    out_dir = config.OUTPUT_SUBDIRS["training"]
    _plot_train_val(experiments, out_dir)
    _plot_accuracy(experiments, out_dir)
    _plot_lr_schedule(experiments, out_dir)