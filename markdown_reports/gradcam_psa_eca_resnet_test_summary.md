# Grad-CAM Visualization Report

Model: **psa_eca_resnet**

Split used: **test**

Samples generated: **42**

Correct predictions in sampled images: **28/42**

Output folder: `reports/gradcam/psa_eca_resnet/test`

## What This Shows

Grad-CAM highlights image regions that most influenced the model's predicted class.

Red/yellow areas mean stronger influence. Blue areas mean weaker influence.

For this project, Grad-CAM helps check whether the model is looking at the skin lesion or disease-affected region instead of unrelated background, ruler marks, image borders, hair, lighting, or text artifacts.

## Important Limitation

Grad-CAM is an explanation tool, not proof that the model is medically correct. It should be used together with metrics, confusion matrices, and mentor/clinical review.

## Sample Visualizations

### acne_vulgaris | predicted: acne_vulgaris | confidence: 59.58%

![Grad-CAM](../reports/gradcam/psa_eca_resnet/test/acne_vulgaris__pred_acne_vulgaris__dermassist__skin_diseases_kaggle__ec030c.png)
### acne_vulgaris | predicted: fungal_nail_infections | confidence: 28.22%

![Grad-CAM](../reports/gradcam/psa_eca_resnet/test/acne_vulgaris__pred_fungal_nail_infections__dermassist__skin_diseases_kaggle__c9c2ee.png)
### acne_vulgaris | predicted: acne_vulgaris | confidence: 87.79%

![Grad-CAM](../reports/gradcam/psa_eca_resnet/test/acne_vulgaris__pred_acne_vulgaris__dermassist__skin_diseases_kaggle__fc624a.png)
### atopic_dermatitis | predicted: atopic_dermatitis | confidence: 50.38%

![Grad-CAM](../reports/gradcam/psa_eca_resnet/test/atopic_dermatitis__pred_atopic_dermatitis__devraj__selected_900plus_final__967cb48b.png)
### atopic_dermatitis | predicted: contact_dermatitis | confidence: 78.45%

![Grad-CAM](../reports/gradcam/psa_eca_resnet/test/atopic_dermatitis__pred_contact_dermatitis__devraj__selected_900plus_final__bdadc624.png)
### atopic_dermatitis | predicted: atopic_dermatitis | confidence: 51.40%

![Grad-CAM](../reports/gradcam/psa_eca_resnet/test/atopic_dermatitis__pred_atopic_dermatitis__dermassist__dermnet__3b47f68d5f6e0565.png)
### basal_cell_carcinoma | predicted: basal_cell_carcinoma | confidence: 90.88%

![Grad-CAM](../reports/gradcam/psa_eca_resnet/test/basal_cell_carcinoma__pred_basal_cell_carcinoma__dermassist__skin_diseases_image__9e96f2f.png)
### basal_cell_carcinoma | predicted: basal_cell_carcinoma | confidence: 92.50%

![Grad-CAM](../reports/gradcam/psa_eca_resnet/test/basal_cell_carcinoma__pred_basal_cell_carcinoma__dermassist__bcn20000__c09804e64bb7c226.png)
### basal_cell_carcinoma | predicted: basal_cell_carcinoma | confidence: 86.14%

![Grad-CAM](../reports/gradcam/psa_eca_resnet/test/basal_cell_carcinoma__pred_basal_cell_carcinoma__dermassist__skin_diseases_image__eb7f6d2.png)
### contact_dermatitis | predicted: contact_dermatitis | confidence: 99.98%

![Grad-CAM](../reports/gradcam/psa_eca_resnet/test/contact_dermatitis__pred_contact_dermatitis__devraj__selected_900plus_final__fafabce6.png)
### contact_dermatitis | predicted: plaque_psoriasis | confidence: 49.01%

![Grad-CAM](../reports/gradcam/psa_eca_resnet/test/contact_dermatitis__pred_plaque_psoriasis__dermassist__dermnet__056f0739bbbaa961.png)
### contact_dermatitis | predicted: contact_dermatitis | confidence: 99.68%

![Grad-CAM](../reports/gradcam/psa_eca_resnet/test/contact_dermatitis__pred_contact_dermatitis__devraj__selected_900plus_final__0c823388.png)


## Saved Files

- Metadata CSV: `reports/gradcam/psa_eca_resnet/test/gradcam_metadata.csv`
- Individual Grad-CAM images: `reports/gradcam/psa_eca_resnet/test`
