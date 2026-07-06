# Pretrained CNN 12-Epoch Summary

Models trained with fixed 12 epochs, no early stopping, and two-phase fine-tuning.

Phase 1: epochs 1-3, frozen backbone, classifier head only.

Phase 2: epochs 4-12, full model fine-tuning.

| Model | Best Epoch | Test Accuracy | Test Macro-F1 | Test Recall Macro | Test Precision Macro |
|---|---:|---:|---:|---:|---:|
| convnext_tiny | 12 | 0.8584 | 0.8309 | 0.8326 | 0.8307 |

Summary CSV: `reports/pretrained_cnns/pretrained_cnn_test_summary.csv`
