# Dataset Balance Summary

Dataset split folder: `/backup/Intern/combine_skindiseaseclassifier_devraj/data/splits`

## Total Images Per Split

| Split | Images |
|---|---:|
| test | 5,474 |
| train | 25,576 |
| val | 5,478 |

## Class Counts

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| melanoma | 6,367 | 1,364 | 1,364 | 9,095 |
| basal_cell_carcinoma | 4,247 | 910 | 910 | 6,067 |
| plaque_psoriasis | 2,342 | 502 | 502 | 3,346 |
| acne_vulgaris | 1,727 | 370 | 370 | 2,467 |
| warts | 1,404 | 300 | 300 | 2,004 |
| atopic_dermatitis | 1,376 | 295 | 294 | 1,965 |
| tinea_corporis | 1,353 | 290 | 289 | 1,932 |
| vitiligo | 1,175 | 251 | 251 | 1,677 |
| drug_eruptions | 1,115 | 239 | 238 | 1,592 |
| contact_dermatitis | 1,062 | 227 | 227 | 1,516 |
| lupus_related_skin_lesions | 900 | 193 | 193 | 1,286 |
| fungal_nail_infections | 844 | 180 | 180 | 1,204 |
| seborrheic_dermatitis | 839 | 180 | 180 | 1,199 |
| folliculitis | 825 | 177 | 176 | 1,178 |

## Train Imbalance

Largest train class: `melanoma` with **6,367** images.

Smallest train class: `folliculitis` with **825** images.

Train imbalance ratio: **7.72:1**.

## Interpretation

The dataset is imbalanced, so PyTorch models should use `WeightedRandomSampler` during training only. Validation and test sets should remain unchanged to keep evaluation honest.
