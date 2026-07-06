# Project Cleanup Summary

Date: 2026-07-02

Project root: `/backup/Intern/combine_skindiseaseclassifier_devraj`

## Cleanup Completed

The project folder was reviewed after training, testing, evaluation, explainability, and comparison were completed.

Removed clearly unnecessary files:

- Python cache folders under `src/`:
  - `src/__pycache__`
  - `src/models/__pycache__`
  - `src/training/__pycache__`
  - `src/utils/__pycache__`
- YOLO dataset cache files:
  - `data/yolo_cls_balanced/train.cache`
  - `data/yolo_cls_balanced/val.cache`
  - `data/yolo_cls_balanced/test.cache`
- Accidental pretrained YOLO weight files from `notebooks/`:
  - `notebooks/yolo26n.pt`
  - `notebooks/yolov8n-cls.pt`
- Stale checkpoint-testing ViT plot generated before ViT was intentionally excluded from that sample checkpoint notebook:
  - `reports/checkpoint_testing/vit-b_16_sample_confusion_matrix.png`
- Older duplicated YOLO test-only metrics folder, superseded by full train/val/test YOLO metrics:
  - `reports/yolo_training/full_test_metrics`
- Older partial YOLO run, superseded by the final main YOLO run:
  - `training_outputs/yolo/yolov8n_cls_fixed_12ep_20260701_111244`

## Files Kept Intentionally

- `.venv/` is kept so notebooks and scripts can run immediately on the server.
- `data/selected_images/` is kept as the final selected dataset.
- `data/splits/` is kept for train/validation/test reproducibility.
- `data/yolo_cls_balanced/` is kept for YOLO classification reproduction.
- `data/quarantine_quality/` is kept for auditability of removed images.
- `training_outputs/` is kept because it contains final checkpoints and metrics.
- `reports/` and `markdown_reports/` are kept for review and mentor documentation.
- `notebooks/` is kept for mentor-friendly execution and visualization.
- `scripts/` and `src/` are kept for reproducibility.

## Final Main Folders

```text
README.md
PROJECT_STRUCTURE.txt
data/
markdown_reports/
notebooks/
reports/
scripts/
src/
training_outputs/
.venv/
```

## Final Main Reports

```text
README.md
markdown_reports/overall_model_comparison_summary.md
markdown_reports/checkpoint_testing_summary.md
markdown_reports/gradcam_convnext_tiny_test_summary.md
markdown_reports/gradcam_efficientnet_b0_test_summary.md
markdown_reports/gradcam_psa_eca_resnet_test_summary.md
markdown_reports/vit_attention_test_summary.md
```
