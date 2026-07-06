# Checkpoint Testing Notebook Summary

Split used: **test**

Mode: **2 images per class**

Output folder: `reports/checkpoint_testing`

## Why This Check Was Done

This check loads every saved model checkpoint and runs prediction on the same test images. It verifies that:

- checkpoint files are present,
- model architectures load correctly,
- class mappings are correct,
- each model can produce predictions,
- sample metrics can be calculated from checkpoint predictions,
- mentor can visually inspect sample predictions.

This is not training. It is a checkpoint health check and inference test.

## Sample Metric Summary

| model | images_tested | accuracy | precision_macro | recall_macro | f1_macro | mean_confidence | checkpoint_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ConvNeXt-Tiny | 28 | 85.71% | 89.29% | 85.71% | 85.00% | 84.82% | completed |
| YOLOv8n-cls | 28 | 78.57% | 84.52% | 78.57% | 79.29% | 74.28% | completed |
| EfficientNet-B0 | 28 | 71.43% | 66.90% | 71.43% | 68.47% | 84.79% | completed |
| PSA + ECA ResNet | 28 | 46.43% | 50.00% | 46.43% | 46.31% | 66.53% | completed |

## Important Notes

- These metrics are calculated on the selected notebook sample unless the notebook is run in full-test mode.
- For final reporting, use the full test metrics from `overall_model_comparison_summary.md`.
- ViT-B/16 may be listed as an in-progress checkpoint if final training/test evaluation has not finished yet.

## Saved Files

- Predictions CSV: `reports/checkpoint_testing/checkpoint_sample_predictions.csv`
- Metrics CSV: `reports/checkpoint_testing/checkpoint_sample_metrics.csv`
- Sample prediction grid: `reports/checkpoint_testing/sample_prediction_grid.png`
- Confusion matrix images: `reports/checkpoint_testing/*_sample_confusion_matrix.png`

![Sample prediction grid](../reports/checkpoint_testing/sample_prediction_grid.png)
