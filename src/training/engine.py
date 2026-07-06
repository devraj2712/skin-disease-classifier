"""Reusable PyTorch train/evaluate loops."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from src.training.metrics import compute_classification_metrics


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
    metric_labels: list[int] | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    use_amp = scaler is not None and device.type == "cuda"
    progress = tqdm(loader, desc="train", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        progress.set_postfix(loss=total_loss / max(total_samples, 1))

    labels_range = metric_labels if metric_labels is not None else sorted(set(y_true) | set(y_pred))
    metrics = compute_classification_metrics(y_true, y_pred, labels=labels_range)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    return_predictions: bool = False,
    metric_labels: list[int] | None = None,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []

    progress = tqdm(loader, desc="eval", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_prob.extend(confs.detach().cpu().tolist())
        progress.set_postfix(loss=total_loss / max(total_samples, 1))

    labels_range = metric_labels if metric_labels is not None else sorted(set(y_true) | set(y_pred))
    metrics: dict[str, object] = compute_classification_metrics(y_true, y_pred, labels=labels_range)
    metrics["loss"] = total_loss / max(total_samples, 1)
    if return_predictions:
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred
        metrics["y_prob"] = y_prob
    return metrics
