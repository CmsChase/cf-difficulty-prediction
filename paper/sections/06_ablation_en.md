# Ablation Study

The best overall ablation result is `hist_gradient_boosting_regressor` with `all_api_features` on `forward_time`, with test MAE 153.0. The one-group drop comparison shows that removing `solved` features produces the largest MAE increase. This supports the central role of solved-count behavior while retaining the usefulness of metadata and tag information.

![Ablation MAE contest grouped](figures/ablation_mae_by_feature_set_contest_grouped.png)

![Ablation MAE forward time](figures/ablation_mae_by_feature_set_forward_time.png)

![Feature drop MAE change](figures/feature_drop_mae_change.png)
