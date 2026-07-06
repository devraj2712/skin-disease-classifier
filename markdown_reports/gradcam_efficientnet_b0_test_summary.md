# Grad-CAM Visualization Report

Model: **efficientnet_b0**

Split used: **test**

Samples generated: **42**

Correct predictions in sampled images: **30/42**

Output folder: `reports/gradcam/efficientnet_b0/test`

## What This Shows

Grad-CAM highlights image regions that most influenced the model's predicted class.

Red/yellow areas mean stronger influence. Blue areas mean weaker influence.

For this project, Grad-CAM helps check whether the model is looking at the skin lesion or disease-affected region instead of unrelated background, ruler marks, image borders, hair, lighting, or text artifacts.

## Important Limitation

Grad-CAM is an explanation tool, not proof that the model is medically correct. It should be used together with metrics, confusion matrices, and mentor/clinical review.

## Sample Visualizations

### acne_vulgaris | predicted: acne_vulgaris | confidence: 99.57%

![Grad-CAM](../reports/gradcam/efficientnet_b0/test/acne_vulgaris__pred_acne_vulgaris__dermassist__skin_diseases_kaggle__ec030c.png)
### acne_vulgaris | predicted: acne_vulgaris | confidence: 99.98%

![Grad-CAM](../reports/gradcam/efficientnet_b0/test/acne_vulgaris__pred_acne_vulgaris__dermassist__skin_diseases_kaggle__c9c2ee.png)
### acne_vulgaris | predicted: acne_vulgaris | confidence: 99.99%

![Grad-CAM](../reports/gradcam/efficientnet_b0/test/acne_vulgaris__pred_acne_vulgaris__dermassist__skin_diseases_kaggle__fc624a.png)
### atopic_dermatitis | predicted: atopic_dermatitis | confidence: 52.91%

![Grad-CAM](../reports/gradcam/efficientnet_b0/test/atopic_dermatitis__pred_atopic_dermatitis__devraj__selected_900plus_final__967cb48b.png)
### atopic_dermatitis | predicted: contact_dermatitis | confidence: 71.67%

![Grad-CAM](../reports/gradcam/efficientnet_b0/test/atopic_dermatitis__pred_contact_dermatitis__devraj__selected_900plus_final__bdadc624.png)
### atopic_dermatitis | predicted: atopic_dermatitis | confidence: 94.75%

![Grad-CAM](../reports/gradcam/efficientnet_b0/test/atopic_dermatitis__pred_atopic_dermatitis__dermassist__dermnet__3b47f68d5f6e0565.png)
### basal_cell_carcinoma | predicted: basal_cell_carcinoma | confidence: 70.35%

![Grad-CAM](../reports/gradcam/efficientnet_b0/test/basal_cell_carcinoma__pred_basal_cell_carcinoma__dermassist__skin_diseases_image__9e96f2f.png)
### basal_cell_carcinoma | predicted: basal_cell_carcinoma | confidence: 99.90%

![Grad-CAM](../reports/gradcam/efficientnet_b0/test/basal_cell_carcinoma__pred_basal_cell_carcinoma__dermassist__bcn20000__c09804e64bb7c226.png)
### basal_cell_carcinoma | predicted: basal_cell_carcinoma | confidence: 85.60%

![Grad-CAM](../reports/gradcam/efficientnet_b0/test/basal_cell_carcinoma__pred_basal_cell_carcinoma__dermassist__skin_diseases_image__eb7f6d2.png)
### contact_dermatitis | predicted: contact_dermatitis | confidence: 99.90%

![Grad-CAM](../reports/gradcam/efficientnet_b0/test/contact_dermatitis__pred_contact_dermatitis__devraj__selected_900plus_final__fafabce6.png)
### contact_dermatitis | predicted: tinea_corporis | confidence: 43.38%

![Grad-CAM](../reports/gradcam/efficientnet_b0/test/contact_dermatitis__pred_tinea_corporis__dermassist__dermnet__056f0739bbbaa961.png)
### contact_dermatitis | predicted: contact_dermatitis | confidence: 100.00%

![Grad-CAM](../reports/gradcam/efficientnet_b0/test/contact_dermatitis__pred_contact_dermatitis__devraj__selected_900plus_final__0c823388.png)


## Saved Files

- Metadata CSV: `reports/gradcam/efficientnet_b0/test/gradcam_metadata.csv`
- Individual Grad-CAM images: `reports/gradcam/efficientnet_b0/test`
