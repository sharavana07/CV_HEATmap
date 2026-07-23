"""
Auto-discovery and loading of experiment artifacts.

An "experiment" is any sub-directory of config.EXPERIMENTS_DIR. The folder
name is used as the model's display name (e.g. "DAFNet"). Adding a new
model to the comparison is as simple as dropping a new folder here with the
same file layout as the others — nothing in the code needs to change.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger("obh_eval")


@dataclass
class Experiment:
    name: str
    root: Path
    history: Optional[pd.DataFrame] = None
    predictions: Optional[pd.DataFrame] = None
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    classification_report_text: Optional[str] = None
    checkpoint_path: Optional[Path] = None
    samples_dir: Optional[Path] = None

    @property
    def has_training_curves(self) -> bool:
        return self.history is not None and not self.history.empty

    @property
    def has_predictions(self) -> bool:
        return self.predictions is not None and not self.predictions.empty


def _read_metrics_file(path: Path) -> dict:
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        # Accept either a single-row table or a key/value table
        if {"metric", "value"}.issubset(set(c.lower() for c in df.columns)):
            cols = {c.lower(): c for c in df.columns}
            return dict(zip(df[cols["metric"]], df[cols["value"]]))
        return df.iloc[0].to_dict()
    return {}


def _derive_metrics_from_predictions(preds: pd.DataFrame) -> dict:
    """Fallback: compute accuracy/precision/recall/f1/auc directly from
    predictions.csv if a dedicated metrics file wasn't provided."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = preds["true_label"].to_numpy()
    y_pred = preds["predicted_label"].to_numpy()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if "probability_up" in preds.columns and len(set(y_true)) > 1:
        try:
            metrics["auc"] = roc_auc_score(y_true, preds["probability_up"].to_numpy())
        except ValueError:
            pass
    return metrics


def discover_experiments(experiments_dir: Path = config.EXPERIMENTS_DIR) -> list[Experiment]:
    """Scan `experiments_dir` and load every model folder found."""
    experiments_dir = Path(experiments_dir)
    if not experiments_dir.exists():
        log.warning("Experiments directory %s does not exist.", experiments_dir)
        return []

    experiments: list[Experiment] = []
    for child in sorted(p for p in experiments_dir.iterdir() if p.is_dir()):
        exp = Experiment(name=child.name, root=child)

        hist_path = child / config.FILE_HISTORY
        if hist_path.exists():
            try:
                if hist_path.suffix == ".json":
                    with open(hist_path) as f:
                        data = json.load(f)

                    # Some files store everything under a "history" key.
                    if "history" in data:
                        data = data["history"]

                    exp.history = pd.DataFrame(data)
                else:
                    exp.history = pd.read_csv(hist_path)
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] failed to read %s: %s", exp.name, hist_path, e)

        preds_path = child / config.FILE_PREDICTIONS
        if preds_path.exists():
            try:
                exp.predictions = pd.read_csv(preds_path)
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] failed to read %s: %s", exp.name, preds_path, e)

        metrics_path_json = child / config.FILE_METRICS
        metrics_path_csv = child / "metrics.csv"
        if metrics_path_json.exists():
            exp.metrics = _read_metrics_file(metrics_path_json)
        elif metrics_path_csv.exists():
            exp.metrics = _read_metrics_file(metrics_path_csv)
        elif exp.has_predictions:
            log.info("[%s] no metrics file found — deriving metrics from predictions.csv", exp.name)
            exp.metrics = _derive_metrics_from_predictions(exp.predictions)

        params_path = child / config.FILE_PARAMS
        if params_path.exists():    
            try:
                with open(params_path) as f:
                    exp.params = json.load(f)
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] failed to read %s: %s", exp.name, params_path, e)

        report_path = child / config.FILE_CLASSIFICATION_REPORT
        if report_path.exists():
            exp.classification_report_text = report_path.read_text()

        ckpt_path = child / config.FILE_CHECKPOINT
        if ckpt_path.exists():
            exp.checkpoint_path = ckpt_path

        samples_dir = child / config.SAMPLES_DIR
        if samples_dir.exists():
            exp.samples_dir = samples_dir

        if not (exp.has_training_curves or exp.has_predictions):
            log.warning("[%s] no history.json or predictions.csv found — skipping folder.", exp.name)
            continue

        experiments.append(exp)
        log.info(
            "Loaded experiment '%s' (training_curves=%s, predictions=%s, metrics=%s, checkpoint=%s)",
            exp.name,
            exp.has_training_curves,
            exp.has_predictions,
            bool(exp.metrics),
            exp.checkpoint_path is not None,
        )

    return experiments


def load_sample_array(exp: Experiment, sample_index: int) -> Optional[np.ndarray]:
    """Load a single heatmap sample (as a 2D/3D numpy array) by index, trying
    each known filename pattern in turn. Returns None if not found."""
    if exp.samples_dir is None:
        return None
    for pattern in config.SAMPLE_FILENAME_PATTERNS:
        candidate = exp.samples_dir / pattern.format(idx=sample_index)
        if candidate.exists():
            if candidate.suffix == ".npy":
                return np.load(candidate)
            if candidate.suffix in (".png", ".jpg", ".jpeg"):
                from PIL import Image

                return np.array(Image.open(candidate).convert("L"))
    return None