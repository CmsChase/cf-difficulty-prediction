# 消融研究

最佳整体消融结果来自 `forward_time` 设置下的 `hist_gradient_boosting_regressor`，特征集合为 `all_api_features`，测试 MAE 为 153.0。单组移除实验显示，移除 `solved` 特征造成最大的 MAE 上升。这与 solved-count 信号的重要性一致。

![Feature drop MAE change](figures/feature_drop_mae_change.png)
