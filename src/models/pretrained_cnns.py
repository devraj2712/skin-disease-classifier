"""Pretrained CNN model factory for skin disease classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
)


SUPPORTED_MODELS = ("efficientnet_b0", "densenet121", "convnext_tiny")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    phase1_lr: float
    phase2_lr: float
    batch_size: int = 32


DEFAULT_MODEL_SPECS = {
    "efficientnet_b0": ModelSpec("efficientnet_b0", phase1_lr=1e-3, phase2_lr=1e-4),
    "densenet121": ModelSpec("densenet121", phase1_lr=1e-3, phase2_lr=1e-4),
    "convnext_tiny": ModelSpec("convnext_tiny", phase1_lr=1e-3, phase2_lr=5e-5),
}


def build_pretrained_cnn(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a supported pretrained CNN and replace the classifier head."""
    model_name = model_name.lower()
    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "densenet121":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    if model_name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model_name={model_name!r}. Supported: {SUPPORTED_MODELS}")


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    """Freeze/unfreeze the backbone while keeping the classifier head trainable."""
    for param in model.parameters():
        param.requires_grad = trainable

    model_name = model_name.lower()
    if model_name == "efficientnet_b0":
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == "densenet121":
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == "convnext_tiny":
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported model_name={model_name!r}.")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


def _shape_test() -> None:
    torch.manual_seed(42)
    for model_name in SUPPORTED_MODELS:
        model = build_pretrained_cnn(model_name, num_classes=14, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            y = model(x)
        total, trainable = count_parameters(model)
        print(f"{model_name}: output={tuple(y.shape)} total={total:,} trainable={trainable:,}")
        if y.shape != torch.Size([2, 14]):
            raise RuntimeError(f"Unexpected output shape for {model_name}: {tuple(y.shape)}")
    print("Pretrained CNN shape tests passed.")


if __name__ == "__main__":
    _shape_test()
