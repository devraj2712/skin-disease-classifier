# Balancing and Augmentation Summary

## Balancing Method

For PyTorch models, balancing is done using `WeightedRandomSampler` on the training split only.

No images are duplicated or physically copied for PyTorch training.

## Why Only Train Split Is Balanced

Validation and test splits are kept unchanged so evaluation remains realistic and honest.

## Train Augmentation

- Resize to 256
- RandomResizedCrop to 224
- Horizontal flip with probability 0.5
- Small rotation up to 10 degrees
- Mild brightness/contrast/saturation/hue changes
- ImageNet normalization

## Validation/Test Transform

- Resize to 256
- CenterCrop to 224
- ImageNet normalization

## Class Weights

| Class | Train Count | Weight |
|---|---:|---:|
| acne_vulgaris | 1,727 | 1.0578 |
| atopic_dermatitis | 1,376 | 1.3277 |
| basal_cell_carcinoma | 4,247 | 0.4302 |
| contact_dermatitis | 1,062 | 1.7202 |
| drug_eruptions | 1,115 | 1.6384 |
| folliculitis | 825 | 2.2144 |
| fungal_nail_infections | 844 | 2.1645 |
| lupus_related_skin_lesions | 900 | 2.0298 |
| melanoma | 6,367 | 0.2869 |
| plaque_psoriasis | 2,342 | 0.7800 |
| seborrheic_dermatitis | 839 | 2.1774 |
| tinea_corporis | 1,353 | 1.3502 |
| vitiligo | 1,175 | 1.5548 |
| warts | 1,404 | 1.3012 |
