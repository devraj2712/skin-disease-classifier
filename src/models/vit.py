"""Vision Transformer model factory for skin disease classification."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights


SUPPORTED_VIT_MODELS = ("vit_b_16",)


def build_vit_b16(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build ViT-B/16 and replace the ImageNet classifier head."""
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def set_vit_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze ViT backbone while keeping classifier head trainable."""
    for param in model.parameters():
        param.requires_grad = trainable
    for param in model.heads.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def _shape_test() -> None:
    torch.manual_seed(42)
    model = build_vit_b16(num_classes=14, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    total, trainable = count_parameters(model)
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Total params: {total:,}")
    print(f"Trainable   : {trainable:,}")
    if y.shape != torch.Size([2, 14]):
        raise RuntimeError(f"Unexpected output shape: {tuple(y.shape)}")
    print("ViT-B/16 shape test passed.")


if __name__ == "__main__":
    _shape_test()
