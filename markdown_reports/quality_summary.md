# Selected Dataset Quality Summary

Dataset: `/backup/Intern/combine_skindiseaseclassifier_devraj/data/selected_images`

Selection rule: kept disease folders with more than 1000 active images and removed Hair & Scalp Disorders.

Total selected images: **36,528**

## Issue Counts After Selection

| Issue type | Count |
|---|---:|
| ok | 28,167 |
| slightly_blurry | 8,361 |


## Melanoma Count Clarification

Melanoma had **10,001** images before quality cleaning. The quality-cleaning step moved **906** bad melanoma images to quarantine, leaving **9,095** usable melanoma images in the selected dataset.

## Selected Class Quality Counts

| Class | Total | OK | Slightly blurry | Quarantine recommended |
|---|---:|---:|---:|---:|
| acne_vulgaris | 2,467 | 2,279 | 188 | 0 |
| atopic_dermatitis | 1,965 | 1,899 | 66 | 0 |
| basal_cell_carcinoma | 6,067 | 1,715 | 4,352 | 0 |
| contact_dermatitis | 1,516 | 1,505 | 11 | 0 |
| drug_eruptions | 1,592 | 1,574 | 18 | 0 |
| folliculitis | 1,178 | 1,051 | 127 | 0 |
| fungal_nail_infections | 1,204 | 1,185 | 19 | 0 |
| lupus_related_skin_lesions | 1,286 | 1,267 | 19 | 0 |
| melanoma | 9,095 | 5,986 | 3,109 | 0 |
| plaque_psoriasis | 3,346 | 3,144 | 202 | 0 |
| seborrheic_dermatitis | 1,199 | 1,176 | 23 | 0 |
| tinea_corporis | 1,932 | 1,884 | 48 | 0 |
| vitiligo | 1,677 | 1,631 | 46 | 0 |
| warts | 2,004 | 1,871 | 133 | 0 |

All quarantine-recommended active images were already removed before this selected dataset was created.
