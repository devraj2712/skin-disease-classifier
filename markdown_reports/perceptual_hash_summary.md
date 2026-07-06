# Selected Dataset Perceptual Hash Summary

Images root: `/backup/Intern/combine_skindiseaseclassifier_devraj/data/selected_images`

Selection rule: kept disease folders with more than 1000 active images and removed Hair & Scalp Disorders.

Images hashed: **36,528**

Hash errors: **0**

dHash Hamming threshold: **2**

pHash Hamming threshold: **6**

Near-duplicate groups: **2,573**

Images inside near-duplicate groups: **5,257**

Largest near-duplicate group size: **14**

## Selected Class Summary

| Class | Images | Groups | Near-duplicate groups | Images in near-duplicate groups | Largest group |
|---|---:|---:|---:|---:|---:|
| acne_vulgaris | 2,467 | 2,452 | 14 | 29 | 3 |
| atopic_dermatitis | 1,965 | 1,949 | 16 | 32 | 2 |
| basal_cell_carcinoma | 6,067 | 3,859 | 2,129 | 4,337 | 14 |
| contact_dermatitis | 1,516 | 1,506 | 9 | 19 | 3 |
| drug_eruptions | 1,592 | 1,512 | 77 | 157 | 4 |
| folliculitis | 1,178 | 1,166 | 12 | 24 | 2 |
| fungal_nail_infections | 1,204 | 1,201 | 3 | 6 | 2 |
| lupus_related_skin_lesions | 1,286 | 1,273 | 13 | 26 | 2 |
| melanoma | 9,095 | 8,946 | 137 | 286 | 7 |
| plaque_psoriasis | 3,346 | 3,315 | 31 | 62 | 2 |
| seborrheic_dermatitis | 1,199 | 1,188 | 11 | 22 | 2 |
| tinea_corporis | 1,932 | 1,880 | 38 | 90 | 5 |
| vitiligo | 1,677 | 1,615 | 61 | 123 | 3 |
| warts | 2,004 | 1,982 | 22 | 44 | 2 |

During train/validation/test splitting, keep every `phash_group_id` in only one split.
