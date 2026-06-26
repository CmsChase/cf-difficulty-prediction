# 结果

contest-grouped 测试集上，hist_gradient_boosting_regressor 的测试 MAE 最低，为 166.9。forward-time 测试集上，random_forest_regressor 的测试 MAE 最低，为 152.5。简单基线中，solved-count-only 强于 index-only 和 tag-only。完整模型仍优于 solved-count-only，说明其他结构化元数据仍有增益。forward-time 的训练/测试差距应被讨论为时间泛化差距或分布漂移，而不应自动解释为过拟合。

![Test MAE by model](figures/test_mae_by_model.png)
