# Revision notes for v3

## What changed from v2 to v3

- Added a new `Robustness Experiments` section after the ablation study.
- Added two robustness subsections: `Cold-start prediction` and `Age-normalized solved count`.
- Updated the abstract, introduction, methods, limitations, and conclusion to distinguish post-publication prediction from cold-start prediction.
- Preserved the main v2 model results: HGB remains the best contest-grouped model in the main baseline table, Random Forest remains the best forward-time model in the main baseline table, and solved-count-only remains the strongest simple baseline.
- Added two paper tables for robustness results.
- Added references to four robustness figures.

## Robustness outputs used

- `outputs/robustness/summary/robustness_summary.json`
- `outputs/robustness/tables/robustness_metrics_test.csv`
- `outputs/robustness/tables/cold_start_comparison.csv`
- `outputs/robustness/tables/age_normalized_comparison.csv`
- `outputs/robustness/tables/age_feature_summary.csv`
- `outputs/robustness/figures/cold_start_mae_comparison.png`
- `outputs/robustness/figures/cold_start_within_200_comparison.png`
- `outputs/robustness/figures/age_normalized_mae_comparison.png`
- `outputs/robustness/figures/age_feature_distributions.png`

## Claims that should be treated cautiously

- Cold-start performance should not be described as comparable to full API performance; the HGB metadata-only setting has much larger test MAE than the full API reference.
- Age normalization should be described as a partial exposure adjustment only.
- Age-normalized solved-only features do not consistently replace raw solved-count features: they improve over raw solved-only in the contest-grouped setting but are worse in the forward-time setting.
- Ridge forward-time age-normalized results are unstable and should not be used as a headline conclusion.
- Full API results use solved statistics observed at snapshot time and should be interpreted as post-publication prediction, not cold-start prediction.

## Figures and tables added

- `paper/tables/robustness_cold_start.md`
- `paper/tables/robustness_age_normalized.md`
- `paper/figures/cold_start_mae_comparison.png`
- `paper/figures/cold_start_within_200_comparison.png`
- `paper/figures/age_normalized_mae_comparison.png`
- `paper/figures/age_feature_distributions.png`
