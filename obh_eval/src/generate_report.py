"""
Main entry point.

Usage:
    python -m src.generate_report
    python -m src.generate_report --experiments-dir path/to/experiments --output-dir path/to/outputs

Automatically:
  1. Detects every trained model under the experiments directory.
  2. Reads training history, predictions, metrics, params, classification
     reports, and (optionally) checkpoints + sample heatmaps for each.
  3. Generates training-behaviour, classification-performance, model
     comparison, heatmap-gallery, feature-map, and Grad-CAM figures.
  4. Exports the ranked comparison table and the publication-ready research
     summary table (CSV / Excel / LaTeX / Markdown).
  5. Writes an index.md summarising everything produced, for quick review
     before a viva / paper submission.

Every stage is independent: if a given model is missing some inputs (no
checkpoint, no saved sample images, no attention module, etc.) only the
figures that depend on that input are skipped, with a clear log line
explaining why. The run never aborts because one model/figure is
incomplete.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import config
from .io_utils import discover_experiments
from . import training
from . import classification
from . import comparison
from . import heatmaps
from . import feature_maps
from . import gradcam
from . import tables


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _ensure_output_dirs() -> None:
    for d in config.OUTPUT_SUBDIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def _write_index(experiments, ranked_df) -> None:
    out_path = config.OUTPUT_SUBDIRS["report"] / "index.md"
    lines = [
        "# Evaluation Report — Order Book Heatmap Vision Models",
        "",
        f"Models evaluated: **{len(experiments)}** "
        f"({', '.join(e.name for e in experiments) if experiments else 'none found'})",
        "",
        "## Contents",
        "",
        "| Section | Location |",
        "|---|---|",
        "| Training behaviour (loss / accuracy / LR) | `evaluation/training/` |",
        "| Per-model classification performance | `evaluation/classification/<model>/` |",
        "| Cross-model comparison charts | `evaluation/comparison/` |",
        "| Qualitative heatmap gallery | `evaluation/heatmaps/<model>/` |",
        "| Feature-map / attention visualisation | `evaluation/feature_maps/<model>/` |",
        "| Grad-CAM explainability | `evaluation/gradcam/<model>/` |",
        "| Research summary table (csv/xlsx/tex/md) | `evaluation/tables/` |",
        "",
    ]
    if ranked_df is not None and not ranked_df.empty:
        lines.append("## Ranked Model Comparison")
        lines.append("")
        lines.append(ranked_df.to_markdown(index=False))
        lines.append("")
    out_path.write_text("\n".join(lines))
    logging.getLogger("obh_eval").info("Wrote summary index to %s", out_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate the full model evaluation report.")
    parser.add_argument("--experiments-dir", type=Path, default=config.EXPERIMENTS_DIR,
                         help="Directory containing one sub-folder per trained model.")
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR,
                         help="Directory to write all figures/tables into.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    config.EXPERIMENTS_DIR = args.experiments_dir
    config.OUTPUT_DIR = args.output_dir
    config.OUTPUT_SUBDIRS.update({k: args.output_dir / "evaluation" / k for k in [
        "training", "classification", "comparison", "heatmaps", "feature_maps", "gradcam", "tables", "report"
    ]})

    _setup_logging(args.verbose)
    log = logging.getLogger("obh_eval")
    _ensure_output_dirs()

    log.info("Scanning %s for experiments...", config.EXPERIMENTS_DIR)
    experiments = discover_experiments(config.EXPERIMENTS_DIR)
    if not experiments:
        log.error("No experiments found under %s. Nothing to do.", config.EXPERIMENTS_DIR)
        return 1
    log.info("Found %d experiment(s): %s", len(experiments), ", ".join(e.name for e in experiments))

    log.info("== Training behaviour figures ==")
    training.generate(experiments)

    log.info("== Classification performance figures ==")
    classification.generate(experiments)

    log.info("== Model comparison figures + ranked table ==")
    ranked_df = comparison.generate(experiments)

    log.info("== Qualitative heatmap gallery ==")
    heatmaps.generate(experiments)

    log.info("== Feature map / attention visualisation ==")
    feature_maps.generate(experiments)

    log.info("== Grad-CAM explainability ==")
    gradcam.generate(experiments)

    log.info("== Research summary table ==")
    summary_df = tables.build_summary_table(experiments)
    tables.export_all(summary_df)

    _write_index(experiments, ranked_df)

    log.info("Done. All outputs written under: %s", config.OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())