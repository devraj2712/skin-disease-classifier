from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


PROJECT_ROOT = Path("/backup/Intern/combine_skindiseaseclassifier_devraj")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vit import build_vit_b16, count_parameters, set_vit_backbone_trainable  # noqa: E402
from src.training.checkpoint import save_checkpoint  # noqa: E402
from src.training.data import build_dataloaders  # noqa: E402
from src.training.engine import evaluate, train_one_epoch  # noqa: E402
from src.training.metrics import (  # noqa: E402
    save_classification_report,
    save_confusion_matrix,
    save_metrics_json,
    save_predictions_csv,
)
from src.utils.seed import set_seed  # noqa: E402


def metric_row(epoch: int, phase: str, lr: float, trainable_params: int, train_metrics: dict, val_metrics: dict, improved: bool) -> dict:
    return {
        "epoch": epoch,
        "phase": phase,
        "lr": lr,
        "trainable_params": trainable_params,
        "train_accuracy": train_metrics["accuracy"],
        "train_precision_macro": train_metrics["precision_macro"],
        "train_recall_macro": train_metrics["recall_macro"],
        "train_f1_macro": train_metrics["f1_macro"],
        "train_f1_weighted": train_metrics["f1_weighted"],
        "train_loss": train_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision_macro": val_metrics["precision_macro"],
        "val_recall_macro": val_metrics["recall_macro"],
        "val_f1_macro": val_metrics["f1_macro"],
        "val_f1_weighted": val_metrics["f1_weighted"],
        "val_loss": val_metrics["loss"],
        "improved": improved,
    }


def append_training_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    params = [param for param in model.parameters() if param.requires_grad]
    return AdamW(params, lr=lr, weight_decay=weight_decay)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = PROJECT_ROOT / "data/splits"
    output_root = PROJECT_ROOT / "training_outputs/vit/vit_b_16"
    if args.resume_run_dir:
        run_dir = Path(args.resume_run_dir)
        run_name = run_dir.name
    else:
        run_name = args.run_name or f"vit_b_16_fixed_12ep_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ViT-B/16 TRAINING")
    print("=" * 70)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data dir     : {data_dir}")
    print(f"Run dir      : {run_dir}")
    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    bundle = build_dataloaders(
        data_dir=data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=True,
    )
    class_names = [bundle.idx_to_class[idx] for idx in range(len(bundle.idx_to_class))]
    metric_labels = list(range(len(class_names)))

    model = build_vit_b16(num_classes=len(class_names), pretrained=args.pretrained).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Classes      : {len(class_names)}")
    print(f"Train images : {len(bundle.train_dataset)}")
    print(f"Val images   : {len(bundle.val_dataset)}")
    print(f"Test images  : {len(bundle.test_dataset)}")
    print(f"Total params : {total_params:,}")
    print(f"Trainable    : {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler(enabled=device.type == "cuda")
    best_val_f1 = -1.0
    best_epoch = 0
    start_epoch = 1
    history_rows: list[dict] = []

    log_path = run_dir / "training_log.csv"
    config = {
        "model": "vit_b_16",
        "run_name": run_name,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "freeze_epochs": args.freeze_epochs,
        "phase1_lr": args.phase1_lr,
        "phase2_lr": args.phase2_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "use_weighted_sampler": True,
        "pretrained": args.pretrained,
        "early_stopping": False,
        "best_metric": "val_f1_macro",
    }
    save_metrics_json(config, run_dir / "config.json")

    if args.resume_run_dir:
        last_checkpoint = run_dir / "last_model.pth"
        if not last_checkpoint.exists():
            raise FileNotFoundError(f"Cannot resume because last_model.pth was not found: {last_checkpoint}")
        checkpoint = torch.load(last_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if log_path.exists():
            previous_log = pd.read_csv(log_path)
            if not previous_log.empty:
                best_val_f1 = float(previous_log["val_f1_macro"].max())
                best_epoch = int(previous_log.loc[previous_log["val_f1_macro"].idxmax(), "epoch"])
        print(f"Resuming from: {last_checkpoint}")
        print(f"Next epoch   : {start_epoch}")
        print(f"Best val F1  : {best_val_f1:.4f} at epoch {best_epoch}")

    for phase_name, epoch_start, epoch_end, lr in [
        ("head_only", 1, args.freeze_epochs, args.phase1_lr),
        ("full_finetune", args.freeze_epochs + 1, args.epochs, args.phase2_lr),
    ]:
        if epoch_start > epoch_end:
            continue
        set_vit_backbone_trainable(model, trainable=(phase_name == "full_finetune"))
        _, trainable_params = count_parameters(model)
        optimizer = build_optimizer(model, lr=lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epoch_end - epoch_start + 1, 1))

        print("\n" + "-" * 70)
        print(f"Phase: {phase_name} | epochs {epoch_start}-{epoch_end} | lr={lr} | trainable={trainable_params:,}")
        print("-" * 70)

        actual_epoch_start = max(epoch_start, start_epoch)
        if actual_epoch_start > epoch_end:
            print(f"Skipping phase {phase_name}; epochs {epoch_start}-{epoch_end} already completed.")
            continue

        for epoch in range(actual_epoch_start, epoch_end + 1):
            train_metrics = train_one_epoch(
                model,
                bundle.train_loader,
                criterion,
                optimizer,
                device,
                scaler=scaler,
                metric_labels=metric_labels,
            )
            val_metrics = evaluate(
                model,
                bundle.val_loader,
                criterion,
                device,
                return_predictions=False,
                metric_labels=metric_labels,
            )
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            improved = float(val_metrics["f1_macro"]) > best_val_f1
            if improved:
                best_val_f1 = float(val_metrics["f1_macro"])
                best_epoch = epoch
                save_checkpoint(
                    run_dir / "best_model.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    {"train": train_metrics, "val": val_metrics},
                    bundle.class_to_idx,
                    extra=config,
                )

            save_checkpoint(
                run_dir / "last_model.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                {"train": train_metrics, "val": val_metrics},
                bundle.class_to_idx,
                extra=config,
            )
            row = metric_row(epoch, phase_name, current_lr, trainable_params, train_metrics, val_metrics, improved)
            append_training_log(log_path, row)
            history_rows.append(row)

            print(
                f"Epoch {epoch:02d}/{args.epochs} | {phase_name} | "
                f"train_f1={train_metrics['f1_macro']:.4f} | "
                f"val_f1={val_metrics['f1_macro']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"improved={improved}"
            )

    print("\n" + "=" * 70)
    print("TEST EVALUATION USING BEST CHECKPOINT")
    print("=" * 70)
    checkpoint = torch.load(run_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model,
        bundle.test_loader,
        criterion,
        device,
        return_predictions=True,
        metric_labels=metric_labels,
    )
    y_true = test_metrics.pop("y_true")
    y_pred = test_metrics.pop("y_pred")
    y_prob = test_metrics.pop("y_prob")

    image_paths = [sample[0] for sample in bundle.test_dataset.samples]
    save_predictions_csv(image_paths, y_true, y_pred, y_prob, bundle.idx_to_class, run_dir / "predictions_test.csv")
    save_classification_report(y_true, y_pred, class_names, run_dir / "classification_report.txt")
    save_confusion_matrix(y_true, y_pred, class_names, run_dir / "confusion_matrix.png", normalize=False)
    save_confusion_matrix(y_true, y_pred, class_names, run_dir / "confusion_matrix_normalized.png", normalize=True)

    payload = {
        "model": "vit_b_16",
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_checkpoint": str(run_dir / "best_model.pth"),
        "test_metrics": test_metrics,
        "config": config,
    }
    save_metrics_json(payload, run_dir / "metrics.json")
    pd.DataFrame(history_rows).to_csv(run_dir / "training_history_snapshot.csv", index=False)

    print(f"Best epoch       : {best_epoch}")
    print(f"Test accuracy    : {test_metrics['accuracy']:.4f}")
    print(f"Test precision   : {test_metrics['precision_macro']:.4f}")
    print(f"Test recall      : {test_metrics['recall_macro']:.4f}")
    print(f"Test macro-F1    : {test_metrics['f1_macro']:.4f}")
    print(f"Test weighted-F1 : {test_metrics['f1_weighted']:.4f}")
    print(f"Run saved at     : {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ViT-B/16 baseline for skin disease classification.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--phase1-lr", type=float, default=1e-3)
    parser.add_argument("--phase2-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--resume-run-dir", type=str, default="")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
