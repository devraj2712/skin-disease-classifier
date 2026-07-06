"""ResNet-style skin disease classifier with Improved PSA + ECA.

This implements the custom research model used for comparison with pretrained
CNNs, ViT, and YOLO classification models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ECAModule(nn.Module):
    """Efficient Channel Attention without channel reduction.

    Input:  B x C x H x W
    Output: B x C x H x W
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be odd.")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.avg_pool(x)              # B x C x 1 x 1
        weights = weights.squeeze(-1).transpose(-1, -2)  # B x 1 x C
        weights = self.conv(weights)
        weights = self.sigmoid(weights).transpose(-1, -2).unsqueeze(-1)  # B x C x 1 x 1
        return x * weights


class ImprovedPSA(nn.Module):
    """Improved Pyramid Split Attention with ECA and depthwise convolutions.

    The input channels are split into four groups. Each group passes through a
    depthwise convolution with a different kernel size, then through ECA. The
    scale dimension is softmax-weighted before the groups are concatenated.
    """

    def __init__(self, channels: int, scales: int = 4, kernels: tuple[int, ...] = (3, 5, 7, 9)) -> None:
        super().__init__()
        if scales != len(kernels):
            raise ValueError("Number of scales must match number of kernel sizes.")
        if channels % scales != 0:
            raise ValueError(f"channels={channels} must be divisible by scales={scales}.")

        self.channels = channels
        self.scales = scales
        self.group_channels = channels // scales

        self.branches = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for kernel_size in kernels:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.group_channels,
                        self.group_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        groups=self.group_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.group_channels),
                    nn.ReLU(inplace=True),
                )
            )
            self.attentions.append(ECAModule(self.group_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"ImprovedPSA expects 4D input, got shape {tuple(x.shape)}")
        if x.size(1) != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.size(1)}")

        chunks = torch.chunk(x, self.scales, dim=1)
        scale_features = []
        scale_attention_features = []

        for chunk, branch, attention in zip(chunks, self.branches, self.attentions):
            feature = branch(chunk)
            attended = attention(feature)
            scale_features.append(feature)
            scale_attention_features.append(attended)

        features = torch.stack(scale_features, dim=1)          # B x S x Cg x H x W
        attentions = torch.stack(scale_attention_features, dim=1)
        scale_weights = torch.softmax(attentions, dim=1)
        weighted = features * scale_weights
        return torch.cat([weighted[:, idx] for idx in range(self.scales)], dim=1)


class Bottleneck(nn.Module):
    """Standard ResNet bottleneck block."""

    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        outplanes = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inplanes != outplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class PsaEcaBottleneck(nn.Module):
    """Inverted residual bottleneck with Improved PSA + ECA."""

    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, expand_ratio: int = 2) -> None:
        super().__init__()
        hidden_dim = planes * expand_ratio
        if hidden_dim % 4 != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by 4 for PSA.")

        outplanes = planes * self.expansion
        self.conv_expand = nn.Conv2d(inplanes, hidden_dim, kernel_size=1, stride=stride, bias=False)
        self.bn_expand = nn.BatchNorm2d(hidden_dim)
        self.psa = ImprovedPSA(hidden_dim, scales=4)
        self.conv_reduce = nn.Conv2d(hidden_dim, outplanes, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        if stride != 1 or inplanes != outplanes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn_expand(self.conv_expand(x)))
        out = self.psa(out)
        out = self.bn_reduce(self.conv_reduce(out))
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out = self.relu(out + identity)
        return out


@dataclass(frozen=True)
class PsaEcaResNetConfig:
    num_classes: int = 14
    layers: tuple[int, int, int, int] = (3, 4, 6, 3)
    dropout: float = 0.2


class PsaEcaResNet(nn.Module):
    """ResNet-50 style classifier using PSA+ECA blocks in the final stage."""

    def __init__(self, config: PsaEcaResNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or PsaEcaResNetConfig()
        self.inplanes = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(Bottleneck, planes=64, blocks=self.config.layers[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck, planes=128, blocks=self.config.layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, planes=256, blocks=self.config.layers[2], stride=2)
        self.layer4 = self._make_layer(PsaEcaBottleneck, planes=512, blocks=self.config.layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(self.config.dropout) if self.config.dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * Bottleneck.expansion, self.config.num_classes)

        self._init_weights()

    def _make_layer(self, block: type[nn.Module], planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [block(self.inplanes, planes, stride=stride)]
        self.inplanes = planes * getattr(block, "expansion")
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def build_psa_eca_resnet(num_classes: int = 14, dropout: float = 0.2) -> PsaEcaResNet:
    return PsaEcaResNet(PsaEcaResNetConfig(num_classes=num_classes, dropout=dropout))


def _shape_test() -> None:
    torch.manual_seed(42)
    model = build_psa_eca_resnet(num_classes=14)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    total, trainable = count_parameters(model)
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total params: {total:,}")
    print(f"Trainable   : {trainable:,}")
    expected = torch.Size([2, 14])
    if y.shape != expected:
        raise RuntimeError(f"Expected output shape {expected}, got {y.shape}")
    print("Shape test passed.")


if __name__ == "__main__":
    _shape_test()
