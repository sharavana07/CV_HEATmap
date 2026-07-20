"""Simple logistic-regression baseline for the heatmap classification task."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn import config as cfg
from cnn.dataset import build_dataloaders


def _collect_features(loader):
    features, labels = [], []
    for imgs, lbls in loader:
        features.append(imgs.numpy().reshape(imgs.shape[0], -1))
        labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)


def main() -> None:
    lbl_path = cfg.LABELS_PATH if cfg.LABELS_PATH.exists() else cfg.LABELS_ALT_PATH
    raw_labels = np.load(lbl_path)

    train_loader, _, test_loader, _, _ = build_dataloaders(raw_labels, heatmap_dir=cfg.HEATMAP_DIR)

    X_train, y_train = _collect_features(train_loader)
    X_test, y_test = _collect_features(test_loader)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.SEED)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("=" * 50)
    print(" Logistic Regression Baseline")
    print("=" * 50)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"], zero_division=0))
    print("\nConfusion Matrix\n")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
