"""Dataset, transforms, and sampler helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class DataBundle:
    train_dataset: datasets.ImageFolder
    val_dataset: datasets.ImageFolder
    test_dataset: datasets.ImageFolder
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]
    class_counts: torch.Tensor
    class_weights: torch.Tensor


def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    """Return train and eval transforms.

    Train transform uses mild augmentation. Eval transform is deterministic.
    """
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(image_size, scale=(0.80, 1.00), ratio=(0.90, 1.10)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, eval_transform


def compute_class_weights(targets: list[int] | torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute inverse-frequency class weights from train targets."""
    targets_tensor = torch.as_tensor(targets, dtype=torch.long)
    class_counts = torch.bincount(targets_tensor, minlength=num_classes).float()
    if torch.any(class_counts == 0):
        missing = torch.where(class_counts == 0)[0].tolist()
        raise ValueError(f"Classes with zero train samples: {missing}")
    class_weights = len(targets_tensor) / (num_classes * class_counts)
    return class_counts, class_weights


def build_weighted_sampler(targets: list[int] | torch.Tensor, class_weights: torch.Tensor) -> WeightedRandomSampler:
    """Create a train-only WeightedRandomSampler."""
    targets_tensor = torch.as_tensor(targets, dtype=torch.long)
    sample_weights = class_weights[targets_tensor].double()
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def build_dataloaders(
    data_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    use_weighted_sampler: bool = True,
    pin_memory: bool | None = None,
) -> DataBundle:
    """Build ImageFolder datasets and loaders for data/splits.

    num_workers defaults to 0 because this server showed /dev/shm DataLoader
    errors with multiple workers. Increase later only if the server allows it.
    """
    data_dir = Path(data_dir)
    train_transform, eval_transform = build_transforms(image_size=image_size)

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=eval_transform)

    if train_dataset.class_to_idx != val_dataset.class_to_idx or train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Train/val/test class_to_idx mappings do not match.")

    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_counts, class_weights = compute_class_weights(train_dataset.targets, len(class_to_idx))

    sampler = build_weighted_sampler(train_dataset.targets, class_weights) if use_weighted_sampler else None
    shuffle_train = sampler is None
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return DataBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        class_counts=class_counts,
        class_weights=class_weights,
    )


def class_weight_table(idx_to_class: dict[int, str], class_counts: torch.Tensor, class_weights: torch.Tensor) -> pd.DataFrame:
    """Return a readable class count/weight table."""
    rows = []
    for class_idx in range(len(class_counts)):
        rows.append({
            "class_idx": class_idx,
            "class_name": idx_to_class[class_idx],
            "train_count": int(class_counts[class_idx].item()),
            "class_weight": float(class_weights[class_idx].item()),
        })
    return pd.DataFrame(rows)
