from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path("/backup/Intern/combine_skindiseaseclassifier_devraj")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pretrained_cnns import build_pretrained_cnn  # noqa: E402
from src.models.psa_eca_resnet import build_psa_eca_resnet  # noqa: E402
from src.training.data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


MODEL_RUNS = {
    "convnext_tiny": PROJECT_ROOT
    / "training_outputs/pretrained_cnns/convnext_tiny/convnext_tiny_fixed_12ep_20260701_070637/best_model.pth",
    "efficientnet_b0": PROJECT_ROOT
    / "training_outputs/pretrained_cnns/efficientnet_b0/efficientnet_b0_fixed_12ep_20260630_091338/best_model.pth",
    "psa_eca_resnet": PROJECT_ROOT
    / "training_outputs/psa_eca_resnet/psa_eca_resnet_fixed_12ep_20260630_042833/best_model.pth",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        _module: torch.nn.Module,
        _grad_input: tuple[torch.Tensor],
        grad_output: tuple[torch.Tensor],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, target_class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        score = logits[:, target_class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        if self.activations.dim() != 4 or self.gradients.dim() != 4:
            raise RuntimeError(
                "Target layer must return a 4D tensor shaped B x C x H x W. "
                f"Got activation shape {tuple(self.activations.shape)}."
            )

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = cam - cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam = cam / max_value
        return cam

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def load_model(model_name: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, int]]:
    checkpoint_path = MODEL_RUNS[model_name]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    if model_name in {"convnext_tiny", "efficientnet_b0"}:
        model = build_pretrained_cnn(model_name, num_classes=num_classes, pretrained=False)
    elif model_name == "psa_eca_resnet":
        model = build_psa_eca_resnet(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model_name={model_name!r}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_to_idx


def target_layer_for(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    if model_name == "convnext_tiny":
        return model.features[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]
    if model_name == "psa_eca_resnet":
        return model.layer4[-1]
    raise ValueError(f"Unsupported model_name={model_name!r}")


def collect_samples(split_dir: Path, samples_per_class: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    selected: list[Path] = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        images = [
            p
            for p in sorted(class_dir.iterdir())
            if p.suffix.lower() in IMAGE_EXTENSIONS and p.exists()
        ]
        rng.shuffle(images)
        selected.extend(images[:samples_per_class])
    return selected


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    display_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
    ])
    tensor_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return display_transform, tensor_transform


def overlay_heatmap(rgb_image: np.ndarray, cam: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8((1 - alpha) * rgb_image + alpha * heatmap)
    return overlay


def save_visualization(
    output_path: Path,
    rgb_image: np.ndarray,
    cam: np.ndarray,
    overlay: np.ndarray,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    axes[0].imshow(rgb_image)
    axes[0].set_title("Input")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown_report(model_name: str, rows: list[dict[str, object]], output_dir: Path, report_path: Path) -> None:
    correct = sum(1 for row in rows if row["true_label"] == row["pred_label"])
    total = len(rows)
    rel_output = output_dir.relative_to(PROJECT_ROOT)

    examples = []
    for row in rows[:12]:
        image_rel = Path(row["gradcam_path"]).relative_to(PROJECT_ROOT)
        examples.append(
            f"### {row['true_label']} | predicted: {row['pred_label']} | confidence: {float(row['confidence']):.2%}\n\n"
            f"![Grad-CAM](../{image_rel})\n"
        )

    content = f"""# Grad-CAM Visualization Report

Model: **{model_name}**

Split used: **test**

Samples generated: **{total}**

Correct predictions in sampled images: **{correct}/{total}**

Output folder: `{rel_output}`

## What This Shows

Grad-CAM highlights image regions that most influenced the model's predicted class.

Red/yellow areas mean stronger influence. Blue areas mean weaker influence.

For this project, Grad-CAM helps check whether the model is looking at the skin lesion or disease-affected region instead of unrelated background, ruler marks, image borders, hair, lighting, or text artifacts.

## Important Limitation

Grad-CAM is an explanation tool, not proof that the model is medically correct. It should be used together with metrics, confusion matrices, and mentor/clinical review.

## Sample Visualizations

{"".join(examples)}

## Saved Files

- Metadata CSV: `{rel_output / "gradcam_metadata.csv"}`
- Individual Grad-CAM images: `{rel_output}`
"""
    report_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for trained skin disease CNNs.")
    parser.add_argument("--model", choices=sorted(MODEL_RUNS), default="convnext_tiny")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_dir = PROJECT_ROOT / "data/splits" / args.split
    output_dir = PROJECT_ROOT / "reports/gradcam" / args.model / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    model, class_to_idx = load_model(args.model, device)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    target_layer = target_layer_for(model, args.model)
    gradcam = GradCAM(model, target_layer)
    display_transform, tensor_transform = build_transforms(args.image_size)

    image_paths = collect_samples(split_dir, args.samples_per_class, args.seed)
    rows: list[dict[str, object]] = []

    try:
        for image_path in image_paths:
            true_label = image_path.parent.name
            pil_image = Image.open(image_path).convert("RGB")
            display_image = display_transform(pil_image)
            rgb_image = np.array(display_image)
            input_tensor = tensor_transform(pil_image).unsqueeze(0).to(device)

            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            pred_idx = int(probabilities.argmax().item())
            pred_label = idx_to_class[pred_idx]
            confidence = float(probabilities[pred_idx].item())

            cam = gradcam(input_tensor, pred_idx)
            overlay = overlay_heatmap(rgb_image, cam)

            safe_name = f"{true_label}__pred_{pred_label}__{image_path.stem[:40]}.png".replace("/", "_")
            gradcam_path = output_dir / safe_name
            title = f"True: {true_label} | Pred: {pred_label} | Confidence: {confidence:.2%}"
            save_visualization(gradcam_path, rgb_image, cam, overlay, title)

            rows.append(
                {
                    "image_path": str(image_path),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": confidence,
                    "correct": true_label == pred_label,
                    "gradcam_path": str(gradcam_path),
                }
            )
    finally:
        gradcam.close()

    metadata_path = output_dir / "gradcam_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report_path = PROJECT_ROOT / "markdown_reports" / f"gradcam_{args.model}_{args.split}_summary.md"
    write_markdown_report(args.model, rows, output_dir, report_path)

    print(f"Generated {len(rows)} Grad-CAM visualizations.")
    print(f"Output folder : {output_dir}")
    print(f"Metadata CSV  : {metadata_path}")
    print(f"Markdown      : {report_path}")


if __name__ == "__main__":
    main()
