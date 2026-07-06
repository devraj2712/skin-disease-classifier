from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torchvision import transforms
from tqdm.auto import tqdm


PROJECT_ROOT = Path("/backup/Intern/combine_skindiseaseclassifier_devraj")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pretrained_cnns import build_pretrained_cnn  # noqa: E402
from src.models.psa_eca_resnet import build_psa_eca_resnet  # noqa: E402
from src.models.vit import build_vit_b16  # noqa: E402
from src.training.data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class CheckpointSpec:
    display_name: str
    model_type: str
    checkpoint_path: Path
    status: str = "completed"


CHECKPOINTS = [
    CheckpointSpec(
        "PSA + ECA ResNet",
        "psa_eca_resnet",
        PROJECT_ROOT / "training_outputs/psa_eca_resnet/psa_eca_resnet_fixed_12ep_20260630_042833/best_model.pth",
    ),
    CheckpointSpec(
        "EfficientNet-B0",
        "efficientnet_b0",
        PROJECT_ROOT
        / "training_outputs/pretrained_cnns/efficientnet_b0/efficientnet_b0_fixed_12ep_20260630_091338/best_model.pth",
    ),
    CheckpointSpec(
        "ConvNeXt-Tiny",
        "convnext_tiny",
        PROJECT_ROOT
        / "training_outputs/pretrained_cnns/convnext_tiny/convnext_tiny_fixed_12ep_20260701_070637/best_model.pth",
    ),
    CheckpointSpec(
        "YOLOv8n-cls",
        "yolo",
        PROJECT_ROOT / "training_outputs/yolo/yolov8n_cls_fixed_12ep_20260701_121336/weights/best.pt",
    ),
    CheckpointSpec(
        "ViT-B/16",
        "vit_b_16",
        PROJECT_ROOT / "training_outputs/vit/vit_b_16/vit_b_16_fixed_12ep_20260702_070652/best_model.pth",
        status="in_progress_checkpoint",
    ),
]


def eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def collect_sample_images(split_dir: Path, samples_per_class: int) -> list[Path]:
    paths: list[Path] = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        class_images = [
            p
            for p in sorted(class_dir.iterdir())
            if p.suffix.lower() in IMAGE_EXTENSIONS and p.exists()
        ]
        paths.extend(class_images[:samples_per_class])
    return paths


def collect_all_images(split_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        paths.extend(
            p
            for p in sorted(class_dir.iterdir())
            if p.suffix.lower() in IMAGE_EXTENSIONS and p.exists()
        )
    return paths


def load_torch_model(spec: CheckpointSpec, device: torch.device) -> tuple[nn.Module, dict[str, int]]:
    checkpoint = torch.load(spec.checkpoint_path, map_location="cpu")
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    if spec.model_type == "psa_eca_resnet":
        model = build_psa_eca_resnet(num_classes=num_classes)
    elif spec.model_type in {"efficientnet_b0", "convnext_tiny"}:
        model = build_pretrained_cnn(spec.model_type, num_classes=num_classes, pretrained=False)
    elif spec.model_type == "vit_b_16":
        model = build_vit_b16(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unsupported torch model type: {spec.model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_to_idx


@torch.no_grad()
def predict_torch_checkpoint(
    spec: CheckpointSpec,
    image_paths: list[Path],
    device: torch.device,
    batch_size: int,
    image_size: int,
) -> list[dict[str, object]]:
    model, class_to_idx = load_torch_model(spec, device)
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    transform = eval_transform(image_size)
    rows: list[dict[str, object]] = []

    for start in tqdm(range(0, len(image_paths), batch_size), desc=spec.display_name, leave=False):
        batch_paths = image_paths[start : start + batch_size]
        tensors = []
        for path in batch_paths:
            image = Image.open(path).convert("RGB")
            tensors.append(transform(image))
        batch = torch.stack(tensors).to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

        for path, pred_idx, confidence in zip(batch_paths, preds.cpu().tolist(), confs.cpu().tolist()):
            true_label = path.parent.name
            pred_label = idx_to_class[int(pred_idx)]
            rows.append(
                {
                    "model": spec.display_name,
                    "model_type": spec.model_type,
                    "checkpoint_path": str(spec.checkpoint_path),
                    "checkpoint_status": spec.status,
                    "image_path": str(path),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": float(confidence),
                    "correct": true_label == pred_label,
                }
            )
    return rows


def predict_yolo_checkpoint(
    spec: CheckpointSpec,
    image_paths: list[Path],
    device: torch.device,
    batch_size: int,
    image_size: int,
) -> list[dict[str, object]]:
    from ultralytics import YOLO

    model = YOLO(str(spec.checkpoint_path))
    yolo_device = 0 if device.type == "cuda" else "cpu"
    rows: list[dict[str, object]] = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc=spec.display_name, leave=False):
        batch_paths = image_paths[start : start + batch_size]
        results = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=image_size,
            batch=batch_size,
            device=yolo_device,
            verbose=False,
            save=False,
            stream=False,
        )
        for path, result in zip(batch_paths, results):
            true_label = path.parent.name
            pred_idx = int(result.probs.top1)
            pred_label = result.names[pred_idx]
            confidence = float(result.probs.top1conf)
            rows.append(
                {
                    "model": spec.display_name,
                    "model_type": spec.model_type,
                    "checkpoint_path": str(spec.checkpoint_path),
                    "checkpoint_status": spec.status,
                    "image_path": str(path),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": confidence,
                    "correct": true_label == pred_label,
                }
            )
    return rows


def metric_summary(rows: list[dict[str, object]]) -> pd.DataFrame:
    summaries = []
    for model_name, group in pd.DataFrame(rows).groupby("model"):
        y_true = group["true_label"].tolist()
        y_pred = group["pred_label"].tolist()
        labels = sorted(set(y_true) | set(y_pred))
        summaries.append(
            {
                "model": model_name,
                "images_tested": len(group),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_macro": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                "recall_macro": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                "mean_confidence": group["confidence"].mean(),
                "checkpoint_path": group["checkpoint_path"].iloc[0],
                "checkpoint_status": group["checkpoint_status"].iloc[0],
            }
        )
    return pd.DataFrame(summaries).sort_values("f1_macro", ascending=False)


def save_confusion_matrices(rows: list[dict[str, object]], output_dir: Path) -> None:
    df = pd.DataFrame(rows)
    for model_name, group in df.groupby("model"):
        labels = sorted(set(group["true_label"]) | set(group["pred_label"]))
        cm = confusion_matrix(group["true_label"], group["pred_label"], labels=labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(f"{model_name} Checkpoint Sample Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=60, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        safe = model_name.lower().replace(" ", "_").replace("+", "plus").replace("/", "_")
        plt.savefig(output_dir / f"{safe}_sample_confusion_matrix.png", dpi=180)
        plt.close()


def save_sample_prediction_grid(rows: list[dict[str, object]], output_dir: Path, max_images: int = 28) -> Path:
    df = pd.DataFrame(rows)
    sample_df = df.groupby("true_label", group_keys=False).head(2).head(max_images)
    cols = 4
    rows_count = (len(sample_df) + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(16, rows_count * 4))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, sample_df.iterrows()):
        image = Image.open(row["image_path"]).convert("RGB")
        image = ImageOps.contain(image, (280, 220))
        ax.imshow(image)
        status = "OK" if row["correct"] else "WRONG"
        ax.set_title(
            f"{row['model']}\nTrue: {row['true_label']}\nPred: {row['pred_label']}\n{status} | {row['confidence']:.1%}",
            fontsize=8,
        )
        ax.axis("off")
    for ax in axes[len(sample_df) :]:
        ax.axis("off")
    fig.tight_layout()
    path = output_dir / "sample_prediction_grid.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_markdown(summary_df: pd.DataFrame, output_dir: Path, report_path: Path, sample_mode: str, split: str) -> None:
    table = summary_df.copy()
    for col in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "mean_confidence"]:
        table[col] = table[col].map(lambda x: f"{x * 100:.2f}%")

    display_cols = [
        "model",
        "images_tested",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "mean_confidence",
        "checkpoint_status",
    ]
    header = "| " + " | ".join(display_cols) + " |"
    separator = "| " + " | ".join(["---"] * len(display_cols)) + " |"
    body = []
    for _, row in table[display_cols].iterrows():
        body.append("| " + " | ".join(str(row[col]) for col in display_cols) + " |")
    markdown_table = "\n".join([header, separator, *body])

    content = f"""# Checkpoint Testing Notebook Summary

Split used: **{split}**

Mode: **{sample_mode}**

Output folder: `{output_dir.relative_to(PROJECT_ROOT)}`

## Why This Check Was Done

This check loads every saved model checkpoint and runs prediction on the same test images. It verifies that:

- checkpoint files are present,
- model architectures load correctly,
- class mappings are correct,
- each model can produce predictions,
- sample metrics can be calculated from checkpoint predictions,
- mentor can visually inspect sample predictions.

This is not training. It is a checkpoint health check and inference test.

## Sample Metric Summary

{markdown_table}

## Important Notes

- These metrics are calculated on the selected notebook sample unless the notebook is run in full-test mode.
- For final reporting, use the full test metrics from `overall_model_comparison_summary.md`.
- ViT-B/16 may be listed as an in-progress checkpoint if final training/test evaluation has not finished yet.

## Saved Files

- Predictions CSV: `{(output_dir / "checkpoint_sample_predictions.csv").relative_to(PROJECT_ROOT)}`
- Metrics CSV: `{(output_dir / "checkpoint_sample_metrics.csv").relative_to(PROJECT_ROOT)}`
- Sample prediction grid: `{(output_dir / "sample_prediction_grid.png").relative_to(PROJECT_ROOT)}`
- Confusion matrix images: `{output_dir.relative_to(PROJECT_ROOT)}/*_sample_confusion_matrix.png`

![Sample prediction grid](../{(output_dir / "sample_prediction_grid.png").relative_to(PROJECT_ROOT)})
"""
    report_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load every checkpoint and test predictions on sample images.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--full-split", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--include-vit", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    split_dir = PROJECT_ROOT / "data/splits" / args.split
    output_dir = PROJECT_ROOT / "reports/checkpoint_testing"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / "markdown_reports/checkpoint_testing_summary.md"

    image_paths = collect_all_images(split_dir) if args.full_split else collect_sample_images(split_dir, args.samples_per_class)
    sample_mode = "full split" if args.full_split else f"{args.samples_per_class} images per class"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"Split        : {args.split}")
    print(f"Sample mode  : {sample_mode}")
    print(f"Images       : {len(image_paths)}")

    all_rows: list[dict[str, object]] = []
    for spec in CHECKPOINTS:
        if spec.model_type == "vit_b_16" and not args.include_vit:
            continue
        if not spec.checkpoint_path.exists():
            print(f"Skipping missing checkpoint: {spec.display_name} -> {spec.checkpoint_path}")
            continue
        print(f"\nChecking checkpoint: {spec.display_name}")
        if spec.model_type == "yolo":
            rows = predict_yolo_checkpoint(spec, image_paths, device, args.batch_size, args.image_size)
        else:
            rows = predict_torch_checkpoint(spec, image_paths, device, args.batch_size, args.image_size)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No checkpoint predictions were generated.")

    predictions_df = pd.DataFrame(all_rows)
    summary_df = metric_summary(all_rows)

    predictions_df.to_csv(output_dir / "checkpoint_sample_predictions.csv", index=False)
    summary_df.to_csv(output_dir / "checkpoint_sample_metrics.csv", index=False)
    (output_dir / "checkpoint_sample_metrics.json").write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    save_confusion_matrices(all_rows, output_dir)
    save_sample_prediction_grid(all_rows, output_dir)
    write_markdown(summary_df, output_dir, report_path, sample_mode, args.split)

    print("\nCheckpoint test complete.")
    print(summary_df[["model", "images_tested", "accuracy", "precision_macro", "recall_macro", "f1_macro"]])
    print(f"\nPredictions: {output_dir / 'checkpoint_sample_predictions.csv'}")
    print(f"Metrics    : {output_dir / 'checkpoint_sample_metrics.csv'}")
    print(f"Markdown   : {report_path}")


if __name__ == "__main__":
    main()
