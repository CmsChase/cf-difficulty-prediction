# Robustness: age-normalized solved count

This table reports the main age-normalized robustness results for `hist_gradient_boosting_regressor` on the test split. Age normalization is a simple exposure adjustment and should not be interpreted as a complete correction for solved-count bias.

| Strategy | Feature set | Test MAE | RMSE | R2 | Within 200 |
|---|---:|---:|---:|---:|---:|
| contest_grouped | age_normalized_solved_only | 221.95 | 304.40 | 0.825 | 0.583 |
| contest_grouped | raw_solved_only_reference | 268.51 | 349.08 | 0.770 | 0.465 |
| contest_grouped | full_api_plus_age_norm | 145.74 | 197.77 | 0.926 | 0.736 |
| contest_grouped | full_api_without_raw_solved_but_with_age_norm | 153.56 | 206.59 | 0.919 | 0.724 |
| forward_time | age_normalized_solved_only | 414.62 | 529.14 | 0.583 | 0.311 |
| forward_time | raw_solved_only_reference | 231.85 | 294.37 | 0.871 | 0.503 |
| forward_time | full_api_plus_age_norm | 147.73 | 194.54 | 0.944 | 0.725 |
| forward_time | full_api_without_raw_solved_but_with_age_norm | 305.58 | 408.48 | 0.752 | 0.436 |

Source: `outputs/robustness/tables/age_normalized_comparison.csv`.
