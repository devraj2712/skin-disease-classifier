"""Classification metric helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(y_true: list[int], y_pred: list[int], labels: list[int]) -> dict[str, float]:
    """Compute core multiclass metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def classification_report_text(y_true: list[int], y_pred: list[int], class_names: list[str]) -> str:
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


def save_metrics_json(metrics: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_classification_report(y_true: list[int], y_pred: list[int], class_names: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(classification_report_text(y_true, y_pred, class_names), encoding="utf-8")


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    path: str | Path,
    normalize: bool = False,
) -> None:
    """Save confusion matrix plot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    values = cm
    fmt = "d"
    title = "Confusion Matrix"
    if normalize:
        values = cm.astype(float)
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"

    plt.figure(figsize=(12, 10))
    sns.heatmap(values, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=60, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_predictions_csv(
    image_paths: list[str],
    y_true: list[int],
    y_pred: list[int],
    probabilities: list[float],
    idx_to_class: dict[int, str],
    path: str | Path,
) -> None:
    """Save per-image predictions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path, true_idx, pred_idx, prob in zip(image_paths, y_true, y_pred, probabilities):
        rows.append({
            "image_path": image_path,
            "true_idx": int(true_idx),
            "true_class": idx_to_class[int(true_idx)],
            "pred_idx": int(pred_idx),
            "pred_class": idx_to_class[int(pred_idx)],
            "pred_confidence": float(prob),
            "correct": int(true_idx) == int(pred_idx),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
