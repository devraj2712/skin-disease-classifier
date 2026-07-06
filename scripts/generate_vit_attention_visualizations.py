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

from src.models.vit import build_vit_b16  # noqa: E402
from src.training.data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


CHECKPOINT_PATH = PROJECT_ROOT / "training_outputs/vit/vit_b_16/vit_b_16_fixed_12ep_20260702_070652/best_model.pth"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(device: torch.device) -> tuple[torch.nn.Module, dict[str, int]]:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    class_to_idx = checkpoint["class_to_idx"]
    model = build_vit_b16(num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_to_idx


def collect_samples(split_dir: Path, samples_per_class: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    selected: list[Path] = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        images = [p for p in sorted(class_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS and p.exists()]
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


@torch.no_grad()
def forward_with_attention_rollout(model: torch.nn.Module, input_tensor: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    x = model._process_input(input_tensor)
    batch_size = x.shape[0]
    class_token = model.class_token.expand(batch_size, -1, -1)
    x = torch.cat([class_token, x], dim=1)

    rollout = None
    for layer in model.encoder.layers:
        y = layer.ln_1(x)
        attn_output, attn_weights = layer.self_attention(
            y,
            y,
            y,
            need_weights=True,
            average_attn_weights=False,
        )
        attn_output = layer.dropout(attn_output)
        x = x + attn_output
        z = layer.ln_2(x)
        z = layer.mlp(z)
        x = x + z

        attention = attn_weights.mean(dim=1)[0]
        identity = torch.eye(attention.size(0), device=attention.device)
        attention = attention + identity
        attention = attention / attention.sum(dim=-1, keepdim=True)
        rollout = attention if rollout is None else attention @ rollout

    x = model.encoder.ln(x)
    logits = model.heads(x[:, 0])

    patch_attention = rollout[0, 1:]
    grid_size = int(np.sqrt(patch_attention.numel()))
    attention_map = patch_attention.reshape(grid_size, grid_size).detach().cpu().numpy()
    attention_map = attention_map - attention_map.min()
    if attention_map.max() > 0:
        attention_map = attention_map / attention_map.max()
    return logits, attention_map


def overlay_attention(rgb_image: np.ndarray, attention_map: np.ndarray, alpha: float = 0.42) -> tuple[np.ndarray, np.ndarray]:
    heat = cv2.resize(attention_map, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0, 1)
    heatmap = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8((1 - alpha) * rgb_image + alpha * heatmap)
    return heat, overlay


def save_visualization(output_path: Path, rgb_image: np.ndarray, heat: np.ndarray, overlay: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    axes[0].imshow(rgb_image)
    axes[0].set_title("Input")
    axes[1].imshow(heat, cmap="jet")
    axes[1].set_title("ViT Attention")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown(rows: list[dict[str, object]], output_dir: Path, report_path: Path) -> None:
    correct = sum(1 for row in rows if row["true_label"] == row["pred_label"])
    total = len(rows)
    examples = []
    for row in rows[:12]:
        image_rel = Path(row["attention_path"]).relative_to(PROJECT_ROOT)
        examples.append(
            f"### {row['true_label']} | predicted: {row['pred_label']} | confidence: {float(row['confidence']):.2%}\n\n"
            f"![ViT attention](../{image_rel})\n"
        )

    content = f"""# ViT-B/16 Attention Visualization Report

Model: **ViT-B/16**

Checkpoint: `{CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}`

Split used: **test**

Samples generated: **{total}**

Correct predictions in sampled images: **{correct}/{total}**

Output folder: `{output_dir.relative_to(PROJECT_ROOT)}`

## What This Shows

This is **attention rollout**, not standard CNN Grad-CAM.

ViT-B/16 splits each image into 16x16 patches. Attention rollout follows how the class token attends to image patches across the transformer layers.

Red/yellow areas mean patches that had stronger influence on the final image-level prediction. Blue areas mean weaker influence.

## Why This Is Useful

It helps check whether ViT is focusing on disease-affected skin regions instead of background, borders, lighting, text artifacts, or unrelated objects.

## Important Limitation

Attention visualization is an explanation aid, not proof of medical correctness. Use it together with metrics, confusion matrices, and mentor review.

## Sample Visualizations

{"".join(examples)}

## Saved Files

- Metadata CSV: `{(output_dir / "vit_attention_metadata.csv").relative_to(PROJECT_ROOT)}`
- Individual attention images: `{output_dir.relative_to(PROJECT_ROOT)}`
"""
    report_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ViT-B/16 attention rollout visualizations.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_dir = PROJECT_ROOT / "data/splits" / args.split
    output_dir = PROJECT_ROOT / "reports/attention/vit_b_16" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    model, class_to_idx = load_model(device)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    display_transform, tensor_transform = build_transforms(args.image_size)
    image_paths = collect_samples(split_dir, args.samples_per_class, args.seed)

    rows: list[dict[str, object]] = []
    for image_path in image_paths:
        true_label = image_path.parent.name
        pil_image = Image.open(image_path).convert("RGB")
        display_image = display_transform(pil_image)
        rgb_image = np.array(display_image)
        input_tensor = tensor_transform(pil_image).unsqueeze(0).to(device)

        logits, attention_map = forward_with_attention_rollout(model, input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probabilities.argmax().item())
        pred_label = idx_to_class[pred_idx]
        confidence = float(probabilities[pred_idx].item())
        heat, overlay = overlay_attention(rgb_image, attention_map)

        safe_name = f"{true_label}__pred_{pred_label}__{image_path.stem[:40]}.png".replace("/", "_")
        attention_path = output_dir / safe_name
        title = f"True: {true_label} | Pred: {pred_label} | Confidence: {confidence:.2%}"
        save_visualization(attention_path, rgb_image, heat, overlay, title)

        rows.append(
            {
                "image_path": str(image_path),
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": confidence,
                "correct": true_label == pred_label,
                "attention_path": str(attention_path),
            }
        )

    metadata_path = output_dir / "vit_attention_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report_path = PROJECT_ROOT / "markdown_reports" / f"vit_attention_{args.split}_summary.md"
    write_markdown(rows, output_dir, report_path)

    print(f"Generated {len(rows)} ViT attention visualizations.")
    print(f"Output folder : {output_dir}")
    print(f"Metadata CSV  : {metadata_path}")
    print(f"Markdown      : {report_path}")


if __name__ == "__main__":
    main()
