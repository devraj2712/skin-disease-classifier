from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path("/backup/Intern/combine_skindiseaseclassifier_devraj")
MARKDOWN_DIR = PROJECT_ROOT / "markdown_reports"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_comparison"
OUTPUT_MD = MARKDOWN_DIR / "overall_model_comparison_summary.md"
OUTPUT_CSV = REPORT_DIR / "model_comparison.csv"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def md_rel(path: Path) -> str:
    return "../" + rel(path)


def pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def num(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.4f}"


def best_row_from_log(log_path: Path) -> dict[str, Any]:
    df = pd.read_csv(log_path)
    if "val_f1_macro" in df.columns:
        idx = df["val_f1_macro"].idxmax()
    elif "metrics/accuracy_top1" in df.columns:
        idx = df["metrics/accuracy_top1"].idxmax()
    else:
        idx = len(df) - 1
    return df.loc[idx].to_dict()


def last_row_from_log(log_path: Path) -> dict[str, Any]:
    return pd.read_csv(log_path).iloc[-1].to_dict()


def add_pytorch_run(rows: list[dict[str, Any]], metrics_path: Path, display_name: str) -> None:
    run_dir = metrics_path.parent
    payload = load_json(metrics_path)
    log_path = run_dir / "training_log.csv"
    best_row = best_row_from_log(log_path)
    last_row = last_row_from_log(log_path)
    test = payload.get("test_metrics", {})

    train_f1 = best_row.get("train_f1_macro")
    val_f1 = best_row.get("val_f1_macro")
    test_f1 = test.get("f1_macro")

    rows.append(
        {
            "model": display_name,
            "model_family": "PyTorch CNN",
            "run_name": payload.get("run_name", run_dir.name),
            "best_epoch": int(payload.get("best_epoch", best_row.get("epoch", 0))),
            "last_epoch": int(last_row.get("epoch", payload.get("best_epoch", 0))),
            "train_accuracy": best_row.get("train_accuracy"),
            "train_precision_macro": best_row.get("train_precision_macro"),
            "train_recall_macro": best_row.get("train_recall_macro"),
            "train_f1_macro": train_f1,
            "train_loss": best_row.get("train_loss"),
            "val_accuracy": best_row.get("val_accuracy"),
            "val_precision_macro": best_row.get("val_precision_macro"),
            "val_recall_macro": best_row.get("val_recall_macro"),
            "val_f1_macro": val_f1,
            "val_loss": best_row.get("val_loss"),
            "test_accuracy": test.get("accuracy"),
            "test_precision_macro": test.get("precision_macro"),
            "test_recall_macro": test.get("recall_macro"),
            "test_f1_macro": test_f1,
            "test_f1_weighted": test.get("f1_weighted"),
            "test_loss": test.get("loss"),
            "overfit_gap_train_val_f1": None if train_f1 is None or val_f1 is None else train_f1 - val_f1,
            "val_test_gap_f1": None if val_f1 is None or test_f1 is None else val_f1 - test_f1,
            "confusion_matrix": rel(run_dir / "confusion_matrix.png"),
            "confusion_matrix_normalized": rel(run_dir / "confusion_matrix_normalized.png"),
            "classification_report": rel(run_dir / "classification_report.txt"),
            "run_dir": rel(run_dir),
            "notes": "Full train/val/test precision, recall, F1, loss available.",
        }
    )


def add_yolo_run(rows: list[dict[str, Any]], metrics_path: Path) -> None:
    payload = load_json(metrics_path)
    split_summary_path = PROJECT_ROOT / "reports/yolo_training/full_split_metrics/yolo_train_val_test_metrics_summary.json"
    full_metrics_path = PROJECT_ROOT / "reports/yolo_training/full_test_metrics/yolo_test_metrics.json"
    split_summary = load_json(split_summary_path) if split_summary_path.exists() else {}
    full_metrics = load_json(full_metrics_path) if full_metrics_path.exists() else {}
    train_metrics = split_summary.get("train", {})
    val_metrics = split_summary.get("val", {})
    test_metrics = split_summary.get("test", full_metrics)
    run_dir = Path(payload["run_dir"])
    results_path = run_dir / "results.csv"
    best_row = best_row_from_log(results_path)
    last_row = last_row_from_log(results_path)

    test_dir = Path(str(run_dir) + "_test")
    raw_cm = test_dir / "confusion_matrix.png"
    norm_cm = test_dir / "confusion_matrix_normalized.png"
    if not raw_cm.exists():
        raw_cm = run_dir / "confusion_matrix.png"
    if not norm_cm.exists():
        norm_cm = run_dir / "confusion_matrix_normalized.png"

    rows.append(
        {
            "model": "YOLOv8n-cls",
            "model_family": "YOLO classification",
            "run_name": payload.get("run_name", run_dir.name),
            "best_epoch": int(best_row.get("epoch", last_row.get("epoch", 0))),
            "last_epoch": int(last_row.get("epoch", 0)),
            "train_accuracy": train_metrics.get("accuracy"),
            "train_precision_macro": train_metrics.get("precision_macro"),
            "train_recall_macro": train_metrics.get("recall_macro"),
            "train_f1_macro": train_metrics.get("f1_macro"),
            "train_loss": best_row.get("train/loss"),
            "val_accuracy": val_metrics.get("accuracy", best_row.get("metrics/accuracy_top1")),
            "val_precision_macro": val_metrics.get("precision_macro"),
            "val_recall_macro": val_metrics.get("recall_macro"),
            "val_f1_macro": val_metrics.get("f1_macro"),
            "val_loss": best_row.get("val/loss"),
            "test_accuracy": test_metrics.get("accuracy", payload.get("test_top1_acc")),
            "test_precision_macro": test_metrics.get("precision_macro"),
            "test_recall_macro": test_metrics.get("recall_macro"),
            "test_f1_macro": test_metrics.get("f1_macro"),
            "test_f1_weighted": test_metrics.get("f1_weighted"),
            "test_loss": None,
            "test_top5_accuracy": test_metrics.get("top5_accuracy", payload.get("test_top5_acc")),
            "overfit_gap_train_val_f1": None
            if train_metrics.get("f1_macro") is None or val_metrics.get("f1_macro") is None
            else train_metrics.get("f1_macro") - val_metrics.get("f1_macro"),
            "val_test_gap_f1": None
            if val_metrics.get("f1_macro") is None or test_metrics.get("f1_macro") is None
            else val_metrics.get("f1_macro") - test_metrics.get("f1_macro"),
            "confusion_matrix": rel(PROJECT_ROOT / "reports/yolo_training/full_split_metrics/test/yolo_test_confusion_matrix.png")
            if (PROJECT_ROOT / "reports/yolo_training/full_split_metrics/test/yolo_test_confusion_matrix.png").exists()
            else rel(PROJECT_ROOT / "reports/yolo_training/full_test_metrics/yolo_test_confusion_matrix.png")
            if (PROJECT_ROOT / "reports/yolo_training/full_test_metrics/yolo_test_confusion_matrix.png").exists()
            else rel(raw_cm)
            if raw_cm.exists()
            else "",
            "confusion_matrix_normalized": rel(PROJECT_ROOT / "reports/yolo_training/full_split_metrics/test/yolo_test_confusion_matrix_normalized.png")
            if (PROJECT_ROOT / "reports/yolo_training/full_split_metrics/test/yolo_test_confusion_matrix_normalized.png").exists()
            else rel(PROJECT_ROOT / "reports/yolo_training/full_test_metrics/yolo_test_confusion_matrix_normalized.png")
            if (PROJECT_ROOT / "reports/yolo_training/full_test_metrics/yolo_test_confusion_matrix_normalized.png").exists()
            else rel(norm_cm)
            if norm_cm.exists()
            else "",
            "classification_report": rel(PROJECT_ROOT / "reports/yolo_training/full_split_metrics/test/yolo_test_classification_report.txt")
            if (PROJECT_ROOT / "reports/yolo_training/full_split_metrics/test/yolo_test_classification_report.txt").exists()
            else rel(PROJECT_ROOT / "reports/yolo_training/full_test_metrics/yolo_test_classification_report.txt")
            if (PROJECT_ROOT / "reports/yolo_training/full_test_metrics/yolo_test_classification_report.txt").exists()
            else "",
            "run_dir": rel(run_dir),
            "notes": "YOLO train/val/test metrics were computed from exported predictions. Train split is balanced with symlinks, so train metrics are useful for overfitting check but not a natural-data score.",
        }
    )


def make_bar_plot(df: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    plot_df = df.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if plot_df.empty:
        return
    plt.figure(figsize=(10, 5))
    bars = plt.bar(plot_df["model"], plot_df[metric] * 100)
    plt.ylabel(metric.replace("_", " ").title() + " (%)")
    plt.title(title)
    plt.ylim(0, 100)
    plt.xticks(rotation=25, ha="right")
    for bar in bars:
        value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_grouped_f1_plot(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.dropna(subset=["train_f1_macro", "val_f1_macro", "test_f1_macro"]).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("test_f1_macro", ascending=False)
    x = range(len(plot_df))
    width = 0.25
    plt.figure(figsize=(11, 5))
    plt.bar([i - width for i in x], plot_df["train_f1_macro"] * 100, width=width, label="Train macro-F1")
    plt.bar(list(x), plot_df["val_f1_macro"] * 100, width=width, label="Val macro-F1")
    plt.bar([i + width for i in x], plot_df["test_f1_macro"] * 100, width=width, label="Test macro-F1")
    plt.xticks(list(x), plot_df["model"], rotation=25, ha="right")
    plt.ylabel("Macro-F1 (%)")
    plt.title("Train vs Validation vs Test Macro-F1")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_gap_plot(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.dropna(subset=["overfit_gap_train_val_f1"]).sort_values("overfit_gap_train_val_f1", ascending=False)
    if plot_df.empty:
        return
    plt.figure(figsize=(10, 5))
    values = plot_df["overfit_gap_train_val_f1"] * 100
    bars = plt.bar(plot_df["model"], values)
    plt.axhline(10, color="red", linestyle="--", linewidth=1, label="10 point gap reference")
    plt.ylabel("Train macro-F1 - validation macro-F1 (percentage points)")
    plt.title("Overfitting Gap Check")
    plt.xticks(rotation=25, ha="right")
    plt.legend()
    for bar in bars:
        value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_markdown(df: pd.DataFrame) -> None:
    sorted_by_f1 = df.dropna(subset=["test_f1_macro"]).sort_values("test_f1_macro", ascending=False)
    sorted_by_acc = df.dropna(subset=["test_accuracy"]).sort_values("test_accuracy", ascending=False)
    best_f1 = sorted_by_f1.iloc[0] if not sorted_by_f1.empty else None

    rows = []
    for _, row in df.sort_values("test_accuracy", ascending=False).iterrows():
        rows.append(
            [
                row["model"],
                str(row["best_epoch"]),
                pct(row["train_accuracy"]),
                pct(row["train_f1_macro"]),
                pct(row["val_accuracy"]),
                pct(row["val_f1_macro"]),
                pct(row["test_accuracy"]),
                pct(row["test_precision_macro"]),
                pct(row["test_recall_macro"]),
                pct(row["test_f1_macro"]),
                pct(row["test_f1_weighted"]),
                pct(row["overfit_gap_train_val_f1"]),
            ]
        )

    rank_f1_rows = []
    for _, row in sorted_by_f1.iterrows():
        rank_f1_rows.append([row["model"], pct(row["test_f1_macro"]), pct(row["test_recall_macro"]), pct(row["test_accuracy"])])

    rank_acc_rows = []
    for _, row in sorted_by_acc.iterrows():
        rank_acc_rows.append([row["model"], pct(row["test_accuracy"]), pct(row["test_f1_macro"]), row["notes"]])

    cm_sections = []
    for _, row in df.iterrows():
        raw = row["confusion_matrix"]
        norm = row["confusion_matrix_normalized"]
        report = row["classification_report"]
        cm_sections.append(f"### {row['model']}\n")
        if raw:
            cm_sections.append(f"![{row['model']} confusion matrix]({md_rel(PROJECT_ROOT / raw)})\n")
        if norm:
            cm_sections.append(f"![{row['model']} normalized confusion matrix]({md_rel(PROJECT_ROOT / norm)})\n")
        if report:
            cm_sections.append(f"Per-class classification report: `{report}`\n")
        cm_sections.append(f"Run folder: `{row['run_dir']}`\n")

    best_text = "N/A"
    if best_f1 is not None:
        best_text = (
            f"**{best_f1['model']}** is currently the best overall model because it has the highest "
            f"test macro-F1 ({pct(best_f1['test_f1_macro'])}) and also the highest test accuracy among the "
            f"models with full precision/recall/F1 metrics ({pct(best_f1['test_accuracy'])})."
        )

    content = f"""# Overall Model Comparison Summary

Generated on: 2026-07-02

## Dataset Used

- Project folder: `/backup/Intern/combine_skindiseaseclassifier_devraj`
- Dataset folder: `data/selected_images`
- Split folder: `data/splits`
- Total images: **36,528**
- Total classes: **14**
- Split rule: **70% train / 15% validation / 15% test**
- Train images: **25,576**
- Validation images: **5,478**
- Test images: **5,474**
- Leakage prevention: perceptual-hash group IDs were kept in only one split.

## Preprocessing And Balancing

- Corrupt/bad files were removed before training.
- Exact duplicate and near-duplicate handling was done before splitting.
- Perceptual-hash grouping was used so similar images do not leak across train, validation, and test.
- PyTorch models used `WeightedRandomSampler` on the training split only.
- PyTorch train-time augmentation used random crop, horizontal flip, small rotation, mild color jitter, and ImageNet normalization.
- PyTorch validation/test used deterministic resize, center crop, and ImageNet normalization.
- YOLO used a separate classification folder at `data/yolo_cls_balanced`; only its train split was balanced using symlinks, while validation and test stayed natural.

## Main Comparison Table

{markdown_table(rows, ["Model", "Best Epoch", "Train Acc", "Train Macro-F1", "Val Acc", "Val Macro-F1", "Test Acc", "Test Precision", "Test Recall", "Test Macro-F1", "Test Weighted-F1", "Train-Val F1 Gap"])}

## Best Model Finding

{best_text}

For this skin disease classification task, the most important comparison metric is **macro-F1**, followed by **macro recall** and the **normalized confusion matrix**. Accuracy is useful, but it can hide weak performance on smaller disease classes.

## Ranking By Test Macro-F1

{markdown_table(rank_f1_rows, ["Ranked Model", "Test Macro-F1", "Test Macro Recall", "Test Accuracy"]) if rank_f1_rows else "No full macro-F1 values were available."}

## Ranking By Test Accuracy

{markdown_table(rank_acc_rows, ["Ranked Model", "Test Accuracy", "Test Macro-F1", "Note"]) if rank_acc_rows else "No test accuracy values were available."}

## Graph Summary

![Test accuracy comparison]({md_rel(REPORT_DIR / "test_accuracy_comparison.png")})

![Test macro-F1 comparison]({md_rel(REPORT_DIR / "test_macro_f1_comparison.png")})

![Train validation test macro-F1 comparison]({md_rel(REPORT_DIR / "train_val_test_macro_f1.png")})

![Overfitting gap comparison]({md_rel(REPORT_DIR / "overfitting_gap_comparison.png")})

## Confusion Matrix Comparison

The normal confusion matrix shows the number of images predicted in each class. The normalized confusion matrix shows percentages, which is easier for checking which disease classes are being confused.

{"".join(cm_sections)}

## Overfitting Interpretation

- A large gap between train macro-F1 and validation macro-F1 means the model is learning the training images much better than unseen validation images.
- ConvNeXt-Tiny has the highest performance, but its train macro-F1 is much higher than validation/test macro-F1, so it should be watched for overfitting.
- EfficientNet-B0 is also strong and slightly lighter than ConvNeXt-Tiny.
- PSA+ECA ResNet is weaker in this run and may need better tuning, stronger regularization, or more epochs with a staged training strategy.
- YOLOv8n-cls now has full test precision, recall, macro-F1, weighted-F1, top-1 accuracy, top-5 accuracy, and confusion matrices, so it can be compared more fairly with the PyTorch models.

## Metrics To Prioritize For Skin Disease Classification

1. **Macro-F1**: treats every disease class equally, even when some classes have fewer images.
2. **Macro recall**: important because missing a disease class can be more serious than a simple wrong prediction.
3. **Per-class recall/F1**: shows which exact diseases are weak.
4. **Normalized confusion matrix**: shows which diseases are confused with each other.
5. **Test accuracy**: useful as a general score, but not enough alone.
6. **Train vs validation gap**: helps detect overfitting.

## Recommended Next Improvements

- Train ViT-B/16 as the transformer baseline and compare it using the same test metrics.
- Try ConvNeXt-Tiny final run with careful regularization and monitor the train-validation gap.
- Add Grad-CAM or attention visualization so the mentor can see which skin region the model is using.
- Review confused class pairs from the confusion matrix and clean labels where diseases look visually similar.
- Try focal loss or class-balanced loss if smaller classes still have weak recall.
- Use grouped k-fold validation later for a stronger research-style evaluation.
- Validate on an external dataset if available, because external testing is the best check for real generalization.

## Saved Comparison Files

- CSV metrics table: `reports/model_comparison/model_comparison.csv`
- Test accuracy graph: `reports/model_comparison/test_accuracy_comparison.png`
- Test macro-F1 graph: `reports/model_comparison/test_macro_f1_comparison.png`
- Train/validation/test macro-F1 graph: `reports/model_comparison/train_val_test_macro_f1.png`
- Overfitting gap graph: `reports/model_comparison/overfitting_gap_comparison.png`
"""

    OUTPUT_MD.write_text(content, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    add_pytorch_run(
        rows,
        PROJECT_ROOT / "training_outputs/psa_eca_resnet/psa_eca_resnet_fixed_12ep_20260630_042833/metrics.json",
        "PSA + ECA ResNet",
    )
    add_pytorch_run(
        rows,
        PROJECT_ROOT / "training_outputs/pretrained_cnns/efficientnet_b0/efficientnet_b0_fixed_12ep_20260630_091338/metrics.json",
        "EfficientNet-B0",
    )
    add_pytorch_run(
        rows,
        PROJECT_ROOT / "training_outputs/pretrained_cnns/convnext_tiny/convnext_tiny_fixed_12ep_20260701_070637/metrics.json",
        "ConvNeXt-Tiny",
    )
    vit_metrics = sorted((PROJECT_ROOT / "training_outputs/vit/vit_b_16").glob("*/metrics.json"))
    if vit_metrics:
        add_pytorch_run(rows, vit_metrics[-1], "ViT-B/16")

    yolo_metrics = PROJECT_ROOT / "reports/yolo_training/yolov8n_cls_fixed_12ep_20260701_121336_test_metrics.json"
    if yolo_metrics.exists():
        add_yolo_run(rows, yolo_metrics)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    make_bar_plot(df, "test_accuracy", "Test Accuracy Comparison", REPORT_DIR / "test_accuracy_comparison.png")
    make_bar_plot(df, "test_f1_macro", "Test Macro-F1 Comparison", REPORT_DIR / "test_macro_f1_comparison.png")
    make_grouped_f1_plot(df, REPORT_DIR / "train_val_test_macro_f1.png")
    make_gap_plot(df, REPORT_DIR / "overfitting_gap_comparison.png")
    write_markdown(df)

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote plots under {REPORT_DIR}")


if __name__ == "__main__":
    main()
