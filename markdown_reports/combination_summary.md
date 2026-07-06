# Selected Training Dataset Summary

Project folder: `/backup/Intern/combine_skindiseaseclassifier_devraj`

Training image folder: `/backup/Intern/combine_skindiseaseclassifier_devraj/data/selected_images`

This dataset contains only the disease classes selected for further training.

Selection rule: classes with **more than 1000 images** were kept, and **Hair & Scalp Disorders** were removed.

Total selected classes: **14**

Total selected images: **36,528**

## Selected Classes

| Disease category | Class | Images | Devraj | DermAssist |
|---|---|---:|---:|---:|
| Acne Disorders | `acne_vulgaris` | 2,467 | 59 | 2,408 |
| Eczema / Dermatitis | `atopic_dermatitis` | 1,965 | 802 | 1,163 |
| Skin Cancers | `basal_cell_carcinoma` | 6,067 | 418 | 5,649 |
| Eczema / Dermatitis | `contact_dermatitis` | 1,516 | 1,200 | 316 |
| Allergic Skin Conditions | `drug_eruptions` | 1,592 | 399 | 1,193 |
| Bacterial Infections | `folliculitis` | 1,178 | 1,178 | 0 |
| Nail Disorders | `fungal_nail_infections` | 1,204 | 0 | 1,204 |
| Autoimmune Skin Disorders | `lupus_related_skin_lesions` | 1,286 | 259 | 1,027 |
| Skin Cancers | `melanoma` | 9,095 | 974 | 8,121 |
| Psoriasis | `plaque_psoriasis` | 3,346 | 1,383 | 1,963 |
| Eczema / Dermatitis | `seborrheic_dermatitis` | 1,199 | 1,199 | 0 |
| Fungal Infections | `tinea_corporis` | 1,932 | 178 | 1,754 |
| Pigmentary Disorders | `vitiligo` | 1,677 | 638 | 1,039 |
| Viral Skin Diseases | `warts` | 2,004 | 139 | 1,865 |

## Notes

- These are the only classes to use for further preprocessing, splitting, and model training.
- Quality-bad images were already removed from the active image folders.
- Perceptual hash groups are available in `reports/perceptual_hash/perceptual_hash_manifest.csv` for leakage-safe splitting.
- Melanoma count is **9,095** because **906** bad melanoma images were moved to quarantine during quality cleaning.
