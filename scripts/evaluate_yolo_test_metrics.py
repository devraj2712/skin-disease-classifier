from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from ultralytics import YOLO


PROJECT_ROOT = Path("/backup/Intern/combine_skindiseaseclassifier_devraj")
MODEL_PATH = PROJECT_ROOT / "training_outputs/yolo/yolov8n_cls_fixed_12ep_20260701_121336/weights/best.pt"
DATA_DIR = PROJECT_ROOT / "data/yolo_cls_balanced"
OUTPUT_DIR = PROJECT_ROOT / "reports/yolo_training/full_split_metrics"
IMAGE_SIZE = 224
BATCH_SIZE = 128

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(test_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for class_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.exists():
                paths.append(path)
    return paths


def save_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    class_names: list[str],
    output_path: Path,
    normalize: bool = False,
) -> None:
    if normalize:
        matrix = confusion_matrix(y_true, y_pred, labels=class_names, normalize="true")
        fmt = ".2f"
        title = "YOLOv8n-cls Normalized Test Confusion Matrix"
    else:
        matrix = confusion_matrix(y_true, y_pred, labels=class_names)
        fmt = "d"
        title = "YOLOv8n-cls Test Confusion Matrix"

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def evaluate_split(model: YOLO, split: str, device: int | str) -> dict[str, object]:
    split_dir = DATA_DIR / split
    split_output_dir = OUTPUT_DIR / split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(split_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {split_dir}")

    class_names = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    rows = []

    for start in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=f"YOLO {split} prediction"):
        batch_paths = image_paths[start : start + BATCH_SIZE]
        results = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=IMAGE_SIZE,
            device=device,
            batch=BATCH_SIZE,
            verbose=False,
            save=False,
            stream=False,
        )

        for path, result in zip(batch_paths, results):
            true_label = path.parent.name
            pred_index = int(result.probs.top1)
            pred_label = result.names[pred_index]
            confidence = float(result.probs.top1conf)

            top5_indices = [int(i) for i in result.probs.top5]
            top5_labels = [result.names[i] for i in top5_indices]
            top5_confidences = [float(result.probs.data[i]) for i in top5_indices]

            rows.append(
                {
                    "image_path": str(path),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": confidence,
                    "correct_top1": true_label == pred_label,
                    "correct_top5": true_label in top5_labels,
                    "top5_labels": "|".join(top5_labels),
                    "top5_confidences": "|".join(f"{v:.6f}" for v in top5_confidences),
                }
            )

    pred_df = pd.DataFrame(rows)
    y_true = pred_df["true_label"].tolist()
    y_pred = pred_df["pred_label"].tolist()

    metrics = {
        "model": "yolov8n-cls",
        "model_path": str(MODEL_PATH),
        "split": split,
        "split_dir": str(split_dir),
        "images": len(pred_df),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=class_names, average="weighted", zero_division=0),
        "top5_accuracy": float(pred_df["correct_top5"].mean()),
        "device": str(device),
        "class_names": class_names,
    }

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        target_names=class_names,
        zero_division=0,
    )

    pred_df.to_csv(split_output_dir / f"yolo_{split}_predictions.csv", index=False)
    (split_output_dir / f"yolo_{split}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (split_output_dir / f"yolo_{split}_classification_report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    (split_output_dir / f"yolo_{split}_classification_report.txt").write_text(report_text, encoding="utf-8")

    save_confusion_matrix(y_true, y_pred, class_names, split_output_dir / f"yolo_{split}_confusion_matrix.png", normalize=False)
    save_confusion_matrix(y_true, y_pred, class_names, split_output_dir / f"yolo_{split}_confusion_matrix_normalized.png", normalize=True)

    print(f"\nYOLO {split} evaluation complete.")
    print(f"Images evaluated : {metrics['images']}")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Precision macro  : {metrics['precision_macro']:.4f}")
    print(f"Recall macro     : {metrics['recall_macro']:.4f}")
    print(f"F1 macro         : {metrics['f1_macro']:.4f}")
    print(f"F1 weighted      : {metrics['f1_weighted']:.4f}")
    print(f"Top-5 accuracy   : {metrics['top5_accuracy']:.4f}")
    print(f"Saved under      : {split_output_dir}")

    return metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    device = 0 if torch.cuda.is_available() else "cpu"

    summary = {}
    for split in ["train", "val", "test"]:
        summary[split] = evaluate_split(model, split, device)

    (OUTPUT_DIR / "yolo_train_val_test_metrics_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved combined summary: {OUTPUT_DIR / 'yolo_train_val_test_metrics_summary.json'}")


if __name__ == "__main__":
    main()
