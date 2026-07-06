# Label Mapping Workflow

1. Extract folder labels from Devraj datasets and CSV labels from DermAssist.
2. Normalize labels by lowercasing and removing punctuation differences.
3. Match labels against `disease_synonyms.csv`.
4. Use only rows marked `include` or `review_include` for combining.
5. Use DermAssist `specific_label` only; `category_label` and `binary_label` are reported but not used.
6. Keep the original selected 8 Devraj disease folders fixed from `selected_900plus_final`.
7. Add extra Devraj classes only for non-original-8 classes.
8. Remove exact duplicate images with SHA-256; exclude cross-label conflicts.
9. Before final training split, run image quality checks and perceptual-hash leakage checks.

## Current Selected Training Dataset

After copying into `combine_skindiseaseclassifier_devraj`, the dataset was filtered to keep only classes with more than 1000 active images and to remove Hair & Scalp Disorders.

Selected classes: 14. Selected images: 36,528.

## Report Update Note

All copied project reports now describe the selected training dataset only: classes with more than 1000 active images, with Hair & Scalp Disorders removed. Current selected total is **36,528** images across **14** classes.
