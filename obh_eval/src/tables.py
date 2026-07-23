"""Builds the final publication-ready research summary table and exports it
as CSV, Excel, LaTeX, and Markdown.

Columns: Model | Accuracy | Precision | Recall | F1 | AUC | Parameters | Training Time
"""

from __future__ import annotations

import logging

import pandas as pd

from . import config
from .io_utils import Experiment

log = logging.getLogger("obh_eval")

COLUMN_ORDER = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "Parameters", "Training Time (s)"]


def _fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float)):
        return round(v, decimals)
    return v


def build_summary_table(experiments: list[Experiment]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        rows.append(
            {
                "Model": exp.name,
                "Accuracy": _fmt(exp.metrics.get("accuracy")),
                "Precision": _fmt(exp.metrics.get("precision")),
                "Recall": _fmt(exp.metrics.get("recall")),
                "F1": _fmt(exp.metrics.get("f1")),
                "AUC": _fmt(exp.metrics.get("auc")),
                "Parameters": exp.params.get("num_parameters"),
                "Training Time (s)": _fmt(exp.params.get("train_time_sec"), 1),
            }
        )
    df = pd.DataFrame(rows, columns=COLUMN_ORDER)
    if "F1" in df.columns and df["F1"].notna().any():
        df = df.sort_values("F1", ascending=False).reset_index(drop=True)
    return df


def export_all(df: pd.DataFrame) -> None:
    out_dir = config.OUTPUT_SUBDIRS["tables"]
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "research_summary_table.csv"
    df.to_csv(csv_path, index=False)

    try:
        xlsx_path = out_dir / "research_summary_table.xlsx"
        df.to_excel(xlsx_path, index=False)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not export .xlsx (missing openpyxl?): %s", e)

    tex_path = out_dir / "research_summary_table.tex"
    with open(tex_path, "w") as f:
        f.write(df.to_latex(index=False, na_rep="--", float_format="%.4f",
                             caption="Model comparison summary.", label="tab:model_comparison"))

    md_path = out_dir / "research_summary_table.md"
    with open(md_path, "w") as f:
        f.write(df.to_markdown(index=False))

    log.info("Research summary table exported to %s (.csv/.xlsx/.tex/.md)", out_dir)