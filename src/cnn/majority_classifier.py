import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from cnn import config as cfg
from cnn.dataset import build_dataloaders


def main():
    # Load the raw labels
    lbl_path = cfg.LABELS_PATH if cfg.LABELS_PATH.exists() else cfg.LABELS_ALT_PATH
    raw_labels = np.load(lbl_path)

    # Build the same temporal split used by the CNN
    train_loader, _, test_loader, _, _ = build_dataloaders(
        raw_labels,
        heatmap_dir=cfg.HEATMAP_DIR
    )

    # ----------------------------
    # Collect test labels
    # ----------------------------
    y_true = []

    for _, labels in test_loader:
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)

    # ----------------------------
    # Collect training labels
    # ----------------------------
    train_labels = []

    for _, labels in train_loader:
        train_labels.extend(labels.numpy())

    train_labels = np.array(train_labels)

    # ----------------------------
    # Majority class from training set
    # ----------------------------
    majority_class = np.bincount(train_labels).argmax()

    print(f"Majority class (train set): {majority_class}")

    # Predict majority class for every test sample
    y_pred = np.full_like(y_true, majority_class)

    # ----------------------------
    # Evaluation
    # ----------------------------
    print("=" * 50)
    print(" Majority Classifier")
    print("=" * 50)

    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred, zero_division=0):.4f}")

    print("\nClassification Report\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["DOWN", "UP"],
            zero_division=0,
        )
    )

    print("Confusion Matrix\n")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()