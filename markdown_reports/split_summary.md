# Stratified Group Split Summary

Input images: `/backup/Intern/combine_skindiseaseclassifier_devraj/data/selected_images`

Output split folder: `/backup/Intern/combine_skindiseaseclassifier_devraj/data/splits`

Split rule: approximately **70% train / 15% validation / 15% test**.

Leakage rule: each `phash_group_id` is kept in only one split.

Total images: **36,528**

Total classes: **14**

Leakage groups found after split: **0**

## Split Totals

| Split | Images | Percent |
|---|---:|---:|
| train | 25,576 | 70.02% |
| val | 5,478 | 15.00% |
| test | 5,474 | 14.99% |

## Class-Wise Split Counts

| Class | Total | Train | Val | Test |
|---|---:|---:|---:|---:|
| acne_vulgaris | 2,467 | 1,727 | 370 | 370 |
| atopic_dermatitis | 1,965 | 1,376 | 295 | 294 |
| basal_cell_carcinoma | 6,067 | 4,247 | 910 | 910 |
| contact_dermatitis | 1,516 | 1,062 | 227 | 227 |
| drug_eruptions | 1,592 | 1,115 | 239 | 238 |
| folliculitis | 1,178 | 825 | 177 | 176 |
| fungal_nail_infections | 1,204 | 844 | 180 | 180 |
| lupus_related_skin_lesions | 1,286 | 900 | 193 | 193 |
| melanoma | 9,095 | 6,367 | 1,364 | 1,364 |
| plaque_psoriasis | 3,346 | 2,342 | 502 | 502 |
| seborrheic_dermatitis | 1,199 | 839 | 180 | 180 |
| tinea_corporis | 1,932 | 1,353 | 290 | 289 |
| vitiligo | 1,677 | 1,175 | 251 | 251 |
| warts | 2,004 | 1,404 | 300 | 300 |
