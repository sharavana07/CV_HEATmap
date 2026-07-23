"""
Consistent, publication-quality matplotlib styling shared by every figure
in the pipeline. Import `apply_style()` once at the top of any plotting
module and use `savefig(fig, path)` to write out figures with the correct
DPI, tight layout, and transparent-safe backgrounds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / script-safe backend
import matplotlib.pyplot as plt

from . import config


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": config.DPI,
            "font.size": config.FONT_SIZE_BASE,
            "font.family": "serif",
            "axes.titlesize": config.FONT_SIZE_BASE + 2,
            "axes.labelsize": config.FONT_SIZE_BASE + 1,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "legend.fontsize": config.FONT_SIZE_BASE - 1,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "xtick.labelsize": config.FONT_SIZE_BASE - 1,
            "ytick.labelsize": config.FONT_SIZE_BASE - 1,
            "lines.linewidth": 1.8,
            "axes.prop_cycle": plt.cycler(color=config.PALETTE),
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def savefig(fig, path: Path, dpi: int | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi or config.DPI)
    plt.close(fig)