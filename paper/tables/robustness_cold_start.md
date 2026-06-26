# Robustness: cold-start comparison

This table compares metadata-only cold-start prediction with the full API reference for the main robustness model, `hist_gradient_boosting_regressor`, on the test split.

| Strategy | Feature set | Test MAE | Full API reference MAE | MAE gap vs full API | Within 200 |
|---|---:|---:|---:|---:|---:|
| contest_grouped | metadata_only_cold_start | 317.52 | 167.47 | +150.05 | 0.399 |
| contest_grouped | full_api_reference | 167.47 | 167.47 | +0.00 | 0.691 |
| forward_time | metadata_only_cold_start | 331.62 | 153.02 | +178.60 | 0.335 |
| forward_time | full_api_reference | 153.02 | 153.02 | +0.00 | 0.712 |

Source: `outputs/robustness/tables/cold_start_comparison.csv`.
