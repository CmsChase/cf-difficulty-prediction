# Abstract

This project studies Codeforces problem-rating prediction from official API metadata and solved-statistics signals. The current dataset contains 10,979 rated programming problems with ratings from 800 to 3500. The strongest simple baseline is solved-count-only in both evaluation settings (solved_count_only_baseline: MAE 274.4; index_only_baseline: MAE 409.2; tag_only_baseline: MAE 482.9 for contest-grouped; solved_count_only_baseline: MAE 227.2; index_only_baseline: MAE 461.2; tag_only_baseline: MAE 579.0 for forward-time). Full models further improve over that baseline: hist_gradient_boosting_regressor achieved the lowest test MAE (166.9) with within-200 accuracy 0.697 on the contest-grouped split, while random_forest_regressor achieved the lowest test MAE (152.5) with within-200 accuracy 0.712 on the forward-time split. Ablations indicate that removing solved features causes the largest observed MAE increase among one-group drops. The results support solved-statistics as a central signal while leaving room for metadata and tag features to improve prediction.

# Introduction

Codeforces problem ratings are useful for curriculum design, recommendation, and contest analysis. However, ratings are only available after human and platform processes have assigned difficulty labels. This project asks how well public metadata and submission-behavior summaries can predict official ratings. The study is intentionally restricted to the official public Codeforces API, avoiding scraping and private data. The goal is not to replace official ratings, but to quantify which public signals are predictive and where errors remain.

# Data

The processed modeling table contains 10,979 rated `PROGRAMMING` problems from 1,948 contests. Ratings range from 800 to 3500. Solved counts are strongly skewed: p50 is 4167, p99 is 73912, and the maximum is 700377. The contest-grouped split has 0 contest overlap between partitions, and the forward-time split is strictly ordered: True.

See Table `paper/tables/dataset_summary.md` for a compact dataset summary.

![Log solved-count distribution](figures/solved_count_hist_log.png)

# Methods

The feature table preserves problem identifiers, official rating as the target, contest start time, index-derived features, point metadata, solved-count features, and one-hot tag indicators. Evaluation uses two complementary strategies. The contest-grouped split prevents contest leakage by assigning every contest to only one partition. The forward-time split orders contests chronologically to test temporal generalization. The model set includes simple baselines, ridge regression, random forest, and histogram gradient boosting in the baseline stage. The ablation stage evaluates ridge regression and histogram gradient boosting across predefined feature groups.

# Results

On the contest-grouped test split, hist_gradient_boosting_regressor achieved the lowest test MAE (166.9) with within-200 accuracy 0.697. On the forward-time test split, random_forest_regressor achieved the lowest test MAE (152.5) with within-200 accuracy 0.712. Among simple baselines, solved-count-only is strongest in both settings: solved_count_only_baseline: MAE 274.4; index_only_baseline: MAE 409.2; tag_only_baseline: MAE 482.9 for contest-grouped and solved_count_only_baseline: MAE 227.2; index_only_baseline: MAE 461.2; tag_only_baseline: MAE 579.0 for forward-time. The full models still improve over solved-count-only, indicating that additional metadata contributes beyond solved statistics. Forward-time train/test gaps are interpreted as temporal generalization gaps or distribution shift, not automatic evidence of overfitting.

![Test MAE by model](figures/test_mae_by_model.png)

![Within-200 by model](figures/within_200_by_model.png)

# Ablation Study

The best overall ablation result is `hist_gradient_boosting_regressor` with `all_api_features` on `forward_time`, with test MAE 153.0. The one-group drop comparison shows that removing `solved` features produces the largest MAE increase. This supports the central role of solved-count behavior while retaining the usefulness of metadata and tag information.

![Ablation MAE contest grouped](figures/ablation_mae_by_feature_set_contest_grouped.png)

![Ablation MAE forward time](figures/ablation_mae_by_feature_set_forward_time.png)

![Feature drop MAE change](figures/feature_drop_mae_change.png)

# Error Analysis

Error tables summarize the largest absolute errors and aggregate errors by tag and index rank. These artifacts are intended to identify regions where public metadata is less sufficient, such as unusual special problems or tasks whose solved counts diverge from official rating. The analysis does not claim causal explanations; it provides diagnostic patterns for later qualitative review.

![Error by tag contest grouped](figures/error_by_tag_top15_contest_grouped.png)

![Error by index rank](figures/error_by_index_rank.png)

# Limitations

The dataset uses public Codeforces API fields only, so it lacks statement text, editorials, participant-level histories, and temporal details of when solves accumulated. Solved counts are strong predictors but can encode popularity, age, and exposure in addition to intrinsic difficulty. Forward-time evaluation partially addresses temporal generalization, but future snapshots may shift as Codeforces problem styles and participant populations change. The present analysis is best viewed as a reproducible baseline for structured public metadata rather than a final difficulty model.

# Conclusion

Across 10,979 rated programming problems, official API metadata and solved statistics support accurate rating prediction. Solved-count-only is the strongest simple baseline, yet full models improve over it, and ablations show that solved features are the most important group among the tested public signals. The resulting artifacts provide a reproducible foundation for future work on richer textual, temporal, and contest-context features.
