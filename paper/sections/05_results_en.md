# Results

On the contest-grouped test split, hist_gradient_boosting_regressor achieved the lowest test MAE (166.9) with within-200 accuracy 0.697. On the forward-time test split, random_forest_regressor achieved the lowest test MAE (152.5) with within-200 accuracy 0.712. Among simple baselines, solved-count-only is strongest in both settings: solved_count_only_baseline: MAE 274.4; index_only_baseline: MAE 409.2; tag_only_baseline: MAE 482.9 for contest-grouped and solved_count_only_baseline: MAE 227.2; index_only_baseline: MAE 461.2; tag_only_baseline: MAE 579.0 for forward-time. The full models still improve over solved-count-only, indicating that additional metadata contributes beyond solved statistics. Forward-time train/test gaps are interpreted as temporal generalization gaps or distribution shift, not automatic evidence of overfitting.

![Test MAE by model](figures/test_mae_by_model.png)

![Within-200 by model](figures/within_200_by_model.png)
