# Skin Disease Classifier

This project builds and evaluates a multi-class skin disease image classifier 

The final dataset contains **36,528 images** across **14 disease classes**. The project includes dataset cleaning, duplicate/leakage control, stratified grouped splitting, imbalance handling, model training, test evaluation, checkpoint verification, model comparison, Grad-CAM, and ViT attention visualization.

## Current Best Model

The strongest model in the current comparison is:

```text
ConvNeXt-Tiny
```

Main test metrics:

| Model | Test Accuracy | Test Macro-F1 | Test Macro Recall |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 85.84% | 83.09% | 83.26% |
| ViT-B/16 | 85.06% | 82.21% | 82.33% |
| EfficientNet-B0 | 82.12% | 78.95% | 79.58% |
| YOLOv8n-cls | 79.78% | 77.62% | 78.57% |
| PSA + ECA ResNet | 67.99% | 63.73% | 64.54% |

For this medical-style classification task, **macro-F1**, **macro recall**, and the **normalized confusion matrix** are more important than accuracy alone because the dataset is class-imbalanced.

Full comparison report:

```text
markdown_reports/overall_model_comparison_summary.md
```

## Folder Structure

```text
combine_skindiseaseclassifier_devraj/
├── data/
│   ├── selected_images/          # Final selected 14-class image dataset
│   ├── splits/                   # Stratified grouped train/val/test split
│   ├── yolo_cls_balanced/         # YOLO classification dataset using train symlinks
│   └── quarantine_quality/        # Removed bad/corrupt/low-quality images
├── markdown_reports/             # Human-readable project reports
├── notebooks/                    # Mentor-friendly notebook workflow
├── reports/                      # CSV, JSON, plots, manifests, metrics
├── scripts/                      # Reusable project scripts
├── src/                          # Python source modules
├── training_outputs/             # Model checkpoints and training outputs
├── PROJECT_STRUCTURE.txt         # Auto-generated structure index
└── README.md                     # This file
```

## Dataset

Final selected dataset:

```text
data/selected_images
```

Total images:

```text
36,528
```

Classes:

```text
acne_vulgaris
atopic_dermatitis
basal_cell_carcinoma
contact_dermatitis
drug_eruptions
folliculitis
fungal_nail_infections
lupus_related_skin_lesions
melanoma
plaque_psoriasis
seborrheic_dermatitis
tinea_corporis
vitiligo
warts
```

Class split summary:

| Split | Images | Percent |
|---|---:|---:|
| Train | 25,576 | 70.02% |
| Validation | 5,478 | 15.00% |
| Test | 5,474 | 14.99% |

Split folder:

```text
data/splits
```

Detailed split report:

```text
markdown_reports/split_summary.md
```

## Data Preparation Workflow



Important reports:

```text
markdown_reports/combination_summary.md
markdown_reports/label_mapping_workflow.md
reports/manifests/combined_manifest.csv
reports/label_mapping/all_source_labels_report.csv
reports/label_mapping/disease_synonyms.csv
```

### 1. Label Mapping

Disease labels were normalized using a synonym mapping workflow. This helps map labels such as clinical/scientific variants into a consistent class folder.

Example:

```text
Basal Cell Carcinoma -> basal_cell_carcinoma
```

### 2. Quality Cleaning

Corrupt, unreadable, zero-byte, too-small, and extreme-quality-problem images were removed before final training.

Quality summary:

```text
markdown_reports/quality_summary.md
reports/quality/quality_summary.json
```

Quarantine folder:

```text
data/quarantine_quality
```

### 3. Duplicate And Leakage Control

Perceptual hashing was used to group visually similar images before splitting.

Important reason:

```text
Near-duplicate images must not appear across train, validation, and test.
```

The split rule kept every `phash_group_id` in only one split.

Reports:

```text
markdown_reports/perceptual_hash_summary.md
reports/perceptual_hash/perceptual_hash_manifest.csv
reports/split/leakage_check.csv
```

### 4. Stratified Group Split

The dataset was split using:

```text
70% train / 15% validation / 15% test
```

Goals:

- preserve class balance as much as possible,
- prevent near-duplicate leakage,
- keep validation and test realistic.

## Imbalance Handling And Augmentation

For PyTorch models:

```text
Train split = WeightedRandomSampler + live augmentation
Validation/test = no augmentation, no balancing
```

No extra physical images are created for PyTorch training.

Train augmentation:

- resize to 256,
- random resized crop to 224,
- horizontal flip with probability 0.5,
- small rotation up to 10 degrees,
- mild brightness/contrast/saturation/hue changes,
- ImageNet normalization.

Validation/test transform:

- resize to 256,
- center crop to 224,
- ImageNet normalization.

Report:

```text
markdown_reports/balancing_and_augmentation_summary.md
```

For YOLO:

```text
data/yolo_cls_balanced
```

YOLO train split was balanced using symlinks. Validation and test were kept natural.

Report:

```text
markdown_reports/yolo_dataset_summary.md
```

## Models Trained

### 1. PSA + ECA ResNet

Custom ResNet-style CNN with attention.

Architecture idea:

- ResNet bottleneck structure,
- final stage uses Improved Pyramid Split Attention,
- ECA channel attention,
- final classifier for 14 classes.

Source:

```text
src/models/psa_eca_resnet.py
```

Notebook:

```text
notebooks/04_train_psa_eca_resnet.ipynb
```

### 2. EfficientNet-B0

Pretrained CNN baseline.

Training strategy:

- ImageNet pretrained weights,
- classifier head replaced for 14 classes,
- two-phase fine-tuning,
- fixed 12 epochs.

Source:

```text
src/models/pretrained_cnns.py
```

Notebook:

```text
notebooks/05_train_pretrained_cnns.ipynb
```

### 3. ConvNeXt-Tiny

Pretrained modern CNN baseline and current best model.

Training strategy:

- ImageNet pretrained weights,
- classifier head replaced for 14 classes,
- two-phase fine-tuning,
- fixed 12 epochs.

Source:

```text
src/models/pretrained_cnns.py
```

Notebook:

```text
notebooks/05_train_pretrained_cnns.ipynb
```

### 4. YOLOv8n-cls

YOLO classification baseline.

Training data:

```text
data/yolo_cls_balanced
```

Notebook:

```text
notebooks/06_train_yolo_classification.ipynb
```

Full YOLO metrics were calculated from exported train/validation/test predictions.

YOLO reports:

```text
reports/yolo_training/full_split_metrics
```

### 5. ViT-B/16

Vision Transformer baseline.

Architecture:

- input image: 224 x 224,
- patch size: 16 x 16,
- total patches: 196,
- transformer layers: 12,
- hidden size: 768,
- attention heads: 12,
- classifier head changed from ImageNet 1000 classes to 14 disease classes.

Source:

```text
src/models/vit.py
```

Notebook:

```text
notebooks/07_train_vit_b16.ipynb
```

## Training Outputs And Checkpoints

All model outputs are saved in:

```text
training_outputs
```

Use `best_model.pth` or `best.pt` for inference/evaluation.

Use `last_model.pth` or `last.pt` only for resuming training.

Main checkpoints:

```text
training_outputs/psa_eca_resnet/psa_eca_resnet_fixed_12ep_20260630_042833/best_model.pth
training_outputs/pretrained_cnns/efficientnet_b0/efficientnet_b0_fixed_12ep_20260630_091338/best_model.pth
training_outputs/pretrained_cnns/convnext_tiny/convnext_tiny_fixed_12ep_20260701_070637/best_model.pth
training_outputs/yolo/yolov8n_cls_fixed_12ep_20260701_121336/weights/best.pt
training_outputs/vit/vit_b_16/vit_b_16_fixed_12ep_20260702_070652/best_model.pth
```

## Evaluation

Each completed model was evaluated using:

- accuracy,
- macro precision,
- macro recall,
- macro-F1,
- weighted-F1,
- confusion matrix,
- normalized confusion matrix,
- classification report,
- prediction CSV.

Main comparison report:

```text
markdown_reports/overall_model_comparison_summary.md
```

Model comparison CSV and plots:

```text
reports/model_comparison/model_comparison.csv
reports/model_comparison/test_accuracy_comparison.png
reports/model_comparison/test_macro_f1_comparison.png
reports/model_comparison/train_val_test_macro_f1.png
reports/model_comparison/overfitting_gap_comparison.png
```

## Checkpoint Testing Notebook

Mentor-facing checkpoint health check notebook:

```text
notebooks/08_test_all_checkpoints.ipynb
```

This notebook loads completed checkpoints and runs them on sample test images to verify that the saved models work.

Outputs:

```text
markdown_reports/checkpoint_testing_summary.md
reports/checkpoint_testing/checkpoint_sample_predictions.csv
reports/checkpoint_testing/checkpoint_sample_metrics.csv
reports/checkpoint_testing/sample_prediction_grid.png
```

Run completed checkpoint test from terminal:

```bash
cd /backup/Intern/combine_skindiseaseclassifier_devraj
source .venv/bin/activate

python scripts/check_all_checkpoints.py \
  --split test \
  --samples-per-class 2 \
  --batch-size 32
```

## Explainability

### CNN Grad-CAM

Grad-CAM visualizations were generated for CNN models.

Reports:

```text
markdown_reports/gradcam_convnext_tiny_test_summary.md
markdown_reports/gradcam_efficientnet_b0_test_summary.md
markdown_reports/gradcam_psa_eca_resnet_test_summary.md
```

Images:

```text
reports/gradcam
```

Generate again:

```bash
python scripts/generate_gradcam_visualizations.py \
  --model convnext_tiny \
  --split test \
  --samples-per-class 3
```

### ViT Attention Rollout

ViT uses attention rollout instead of standard CNN Grad-CAM.

Report:

```text
markdown_reports/vit_attention_test_summary.md
```

Images:

```text
reports/attention/vit_b_16/test
```

Generate again:

```bash
python scripts/generate_vit_attention_visualizations.py \
  --split test \
  --samples-per-class 3
```

## Notebook Workflow

Run notebooks using the kernel:

```text
Combine Skin GPU (.venv)
```

Notebook order:

```text
01_environment_check.ipynb
02_dataset_balance_report.ipynb
03_prepare_balancing_and_augmentation.ipynb
04_train_psa_eca_resnet.ipynb
05_train_pretrained_cnns.ipynb
06_train_yolo_classification.ipynb
07_train_vit_b16.ipynb
08_test_all_checkpoints.ipynb
```

## Scripts

Important scripts:

```text
scripts/step15_stratified_group_split.py
scripts/prepare_yolo_cls_dataset.py
scripts/train_vit_b16.py
scripts/evaluate_yolo_test_metrics.py
scripts/create_model_comparison_report.py
scripts/check_all_checkpoints.py
scripts/generate_gradcam_visualizations.py
scripts/generate_vit_attention_visualizations.py
scripts/update_project_structure.py
```

## Environment

Virtual environment:

```text
.venv
```

Activate:

```bash
cd /backup/Intern/combine_skindiseaseclassifier_devraj
source .venv/bin/activate
```

Check GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected GPU during training:

```text
NVIDIA H100 80GB HBM3
```

## Recreate Main Comparison Report

```bash
cd /backup/Intern/combine_skindiseaseclassifier_devraj
source .venv/bin/activate

python scripts/create_model_comparison_report.py
python scripts/update_project_structure.py
```

## Important Notes

- Validation and test data are intentionally not augmented or balanced.
- PyTorch train balancing uses `WeightedRandomSampler`.
- YOLO train balancing uses symlinks.
- Perceptual hash groups prevent near-duplicate leakage across splits.
- All final model comparisons should use test macro-F1 and macro recall, not accuracy alone.
- Some models show train-validation gaps, so future work should include stronger regularization and external validation.

## Suggested Future Work

- Clean labels for disease pairs that are frequently confused.
- Run external validation on a completely separate dataset.
- Try class-balanced loss or focal loss for weak classes.
- Try final longer training only for the top models with stronger regularization.
- Add calibration and confidence thresholding.
- Add dermatologist/mentor review of Grad-CAM and ViT attention outputs.
- Package the best model into an inference API or simple demo app.

Copyright (c) 2026 Devraj Ghumare. All rights reserved.

This project is shared publicly for portfolio and educational review purposes only.
Reuse, redistribution, or commercial use is not permitted without permission.
