# 摘要

本项目研究如何利用 Codeforces 官方 API 中的公开元数据与解题统计信号预测题目的官方难度评分。当前数据集包含 10,979 道有评分的 `PROGRAMMING` 题目，评分范围为 800 到 3500。在两个评估设置中，solved-count-only 都是最强的简单基线；完整模型进一步优于该基线，其中主结果中 `hist_gradient_boosting_regressor` 在 contest-grouped 划分上表现最好，`random_forest_regressor` 在 forward-time 划分上表现最好。消融实验显示，移除 solved 特征会造成最大 MAE 上升。新增的鲁棒性实验表明，当解题行为不可用时，冷启动预测存在显著性能缺口；年龄归一化解题数可以提供有用但不完整的曝光校正。总体而言，结果说明解题统计是核心公开信号，同时元数据和标签仍提供额外信息，而冷启动难度预测明显比发布后预测更困难。

# 引言

Codeforces 题目评分可用于学习路径设计、题目推荐和竞赛分析。然而，官方评分通常需要在人类和平台流程完成后才可获得。本项目关注一个可复现的问题：仅使用官方公开 API 中的结构化信息，能否有效预测官方评分。研究不使用网页抓取、私有用户数据或题面文本；目标也不是替代官方评分，而是量化公开信号的预测能力与局限。

v3 版本特别区分发布后预测与冷启动预测。发布后预测可以使用快照时刻已经积累的 solved-count 统计，而冷启动预测只能依赖提交积累之前已经存在的元数据。这个区分很重要，因为解题行为具有很强预测能力，但在新题刚发布时并不可用。因此，本文同时报告原始模型结果、特征组消融，以及新的鲁棒性实验，用于评估移除或按题目年龄归一化 solved-count 特征后的变化。

# 相关工作

编程题难度预测与教育数据挖掘、推荐系统和竞赛编程分析相关。在教育场景中，题目难度常通过学习者作答结果、题目元数据、文本或历史交互模式估计。竞赛编程平台具有特殊结构：题目按比赛组织，题号通常反映赛内相对难度，题目带有标签，并最终关联公开解题数。这些字段形成了一个轻量的表格预测问题，可以在不收集私有参与者历史的前提下进行可复现研究。

本项目聚焦结构化公开元数据，而不是题面或题解的语义理解。这一选择使流水线更简单、可复现性更强，但也限制模型只能使用题号、标签、分值、比赛时间和聚合解题数等信号。因此，本文结果可作为只依赖官方 API 元数据与公开提交聚合信息时的透明基线，并可与未来基于自然语言或用户级交互日志的方法互补。

# 数据

处理后的建模表包含来自 1,948 场比赛的 10,979 道有评分 `PROGRAMMING` 题目。评分范围为 800 到 3500。原始数据来自 Codeforces 官方 API，具体包括 `problemset.problems` 与 `contest.list`。其中，`problemset.problems` 同时提供题目元数据和 `problemStatistics`，`contest.list` 提供比赛阶段、开始时间等比赛级元数据。

主分析过滤到具有非空比赛编号的有评分编程题。API 中部分字段可能缺失，因此预处理记录缺失情况，而不是静默改变数据语义。标签在中间层保留为列表值，在建模表中编码为 one-hot 特征。解题数被保留，因为它是 API 公开字段，但解释时需要谨慎：它同时反映内在难度、题目年龄、流行度、曝光度和参与模式。

解题数分布高度偏斜。当前数据集中，解题数中位数为 4,167，99 分位数为 73,912，最大值为 700,377。这种偏斜解释了特征表中的对数变换，也引出了后续关于年龄归一化解题数的鲁棒性实验。contest-grouped 划分在分区之间没有比赛重叠，forward-time 划分严格按比赛开始时间排序。

数据摘要见 `paper/tables/dataset_summary.md`。

![Log solved-count distribution](figures/solved_count_hist_log.png)

# 方法

特征表保留题目标识、官方评分目标、比赛开始时间、题号派生特征、分值元数据、解题数特征以及标签 one-hot 指示变量。题号特征包括开头字母、存在时的数字后缀，以及 A = 1、B = 2 等序数等级。分值元数据同时用数值和缺失指示表示。解题行为由原始 solved count、缺失指示和 `log_solved_count` 表示。

评估采用两种互补划分策略。contest-grouped 划分将每场比赛整体分配到一个分区，从而避免同一比赛泄漏到多个集合。forward-time 划分按时间顺序排列比赛，用于检验时间泛化能力。基线阶段包含简单基线、岭回归、随机森林和直方图梯度提升；消融阶段在预定义特征组上评估岭回归和直方图梯度提升。

鲁棒性实验增加了两个有针对性的检查。第一，冷启动元数据预测排除 solved-count 特征，只使用题号、标签和分值构造特征集，用于近似新题尚未积累足够提交时的场景。第二，年龄归一化解题数根据快照时间和比赛开始时间计算 `problem_age_days`、`solves_per_day` 和 `log_solves_per_day`。这些特征只在鲁棒性模块内部计算，不改变既有处理数据或特征表。

# 结果

在 contest-grouped 测试集上，主基线结果中 `hist_gradient_boosting_regressor` 的测试 MAE 最低，为 166.9，within-200 准确率为 0.697。在 forward-time 测试集上，`random_forest_regressor` 的测试 MAE 最低，为 152.5，within-200 准确率为 0.712。简单基线中，solved-count-only 在两个设置中都最强：contest-grouped 中 `solved_count_only_baseline` 的 MAE 为 274.4，而 index-only 和 tag-only 分别为 409.2 与 482.9；forward-time 中 solved-count-only 的 MAE 为 227.2，而 index-only 和 tag-only 分别为 461.2 与 579.0。

完整模型仍然优于 solved-count-only，说明除解题统计外，其他元数据也提供额外信息。forward-time 的训练/测试差距应解释为时间泛化差距或分布漂移，而不应自动解释为过拟合。因此，最强简单基线并不意味着结构化元数据没有价值；相反，它说明解题行为是核心公开信号，而其他特征在其周围提供补充上下文。

![Test MAE by model](figures/test_mae_by_model.png)

![Within-200 by model](figures/within_200_by_model.png)

# 消融研究

最佳整体消融结果来自 `forward_time` 设置下的 `hist_gradient_boosting_regressor`，特征集为 `all_api_features`，测试 MAE 为 153.0。单组移除比较显示，移除 `solved` 特征会造成最大的 MAE 上升。这支持了 solved-count 行为的核心作用，同时也保留了元数据和标签信息的价值。

消融结果需要与划分设计一起解释。contest-grouped 评估检验模型对未见比赛的泛化，避免比赛身份跨分区泄漏。forward-time 评估从另一个角度更严格：它要求模型从较早比赛迁移到较晚比赛。两种视角下 solved 特征都很重要，但下一节的鲁棒性实验说明，这些特征不能被误认为冷启动信息。

![Ablation MAE contest grouped](figures/ablation_mae_by_feature_set_contest_grouped.png)

![Ablation MAE forward time](figures/ablation_mae_by_feature_set_forward_time.png)

![Feature drop MAE change](figures/feature_drop_mae_change.png)

# 鲁棒性实验

鲁棒性实验区分两个容易混淆的问题。第一个问题是，当移除 solved-count 行为时模型表现如何，这近似冷启动场景。第二个问题是，一个简单的曝光校正，即每日解题数，能否让不同年龄题目的 solved-count 特征更可比。两个实验都复用既有划分并报告测试集表现，不改变主基线、消融或分析输出。

## 8.1 冷启动预测

冷启动预测明显比发布后预测更困难。对于 `hist_gradient_boosting_regressor`，metadata-only cold-start 特征集在 contest-grouped 划分上的测试 MAE 为 317.52，在 forward-time 划分上为 331.62。相比之下，同一模型的 full API reference 在 contest-grouped 上的测试 MAE 为 167.47，在 forward-time 上为 153.02。对应的冷启动 MAE 缺口分别为 +150.05 和 +178.60。

这些结果说明，题号、标签和分值元数据包含有用信息，但不能替代解题行为。冷启动设置应被视为不同任务，而不是完整 API 预测问题的轻微变体。metadata-only 模型的实际用途更接近于在足够提交到来之前进行初步难度估计，而不是替代使用 solved 统计的发布后模型。

冷启动比较见 `paper/tables/robustness_cold_start.md`。

![Cold-start MAE comparison](figures/cold_start_mae_comparison.png)

![Cold-start within-200 comparison](figures/cold_start_within_200_comparison.png)

## 8.2 年龄归一化解题数

年龄归一化实验使用 `solves_per_day` 和 `log_solves_per_day` 对不同曝光时间进行部分调整。对于 `hist_gradient_boosting_regressor`，在 contest-grouped 设置中，age-normalized solved-only 特征优于 raw solved-only 特征：`age_normalized_solved_only` 的 MAE 为 221.95，而 `raw_solved_only_reference` 为 268.51。然而在 forward-time 设置中，模式相反：`age_normalized_solved_only` 的 MAE 为 414.62，而 `raw_solved_only_reference` 为 231.85。

当年龄归一化特征加入完整 HGB 模型时，性能相对原始 full API reference 进一步提升：`full_api_plus_age_norm` 在 contest-grouped 上的 MAE 为 145.74，在 forward-time 上的 MAE 为 147.73。这说明年龄归一化的解题行为可以为灵活的完整模型提供额外信息。但同时，年龄归一化并不能稳定替代原始 solved-count 特征，也不应被表述为消除了 solved-count 偏差。它只是曝光的简单代理，无法完整建模非线性解题积累、平台可见度、参与者构成或比赛流行度变化。

年龄归一化比较见 `paper/tables/robustness_age_normalized.md`。

![Age-normalized MAE comparison](figures/age_normalized_mae_comparison.png)

![Age feature distributions](figures/age_feature_distributions.png)

# 错误分析

错误分析表列出最大绝对误差案例，并按标签和题号等级聚合平均绝对误差。这些结果用于定位公开结构化特征不足的区域，例如特殊题型，或解题数与官方评分不一致的题目。该分析不声称因果解释，而是为后续人工复核提供诊断线索。

大误差在本项目中尤其重要，因为官方评分是一种对推荐和学习路径设计具有实际影响的有序难度信号。一个平均表现良好的模型仍可能在特殊题型、非典型标签组合或流行度与内在难度不一致的题目上失败。因此，错误分析是防止过度解读整体 MAE 的重要补充。

![Error by tag contest grouped](figures/error_by_tag_top15_contest_grouped.png)

![Error by index rank](figures/error_by_index_rank.png)

# 局限性

本研究只使用 Codeforces 官方公开 API 字段，不包含题面文本、题解、用户历史或解题时间序列。解题数虽然预测能力强，但同时反映题目曝光度、发布时间和流行度，并不完全等同于内在难度。forward-time 划分能够部分检验时间泛化，但未来数据仍可能受到题型风格和参赛群体变化影响。当前分析更适合作为结构化公开元数据的可复现基线，而不是最终难度模型。

真正的冷启动预测仍然困难，因为在提交积累之前无法使用解题行为。冷启动鲁棒性结果显示，metadata-only 特征与完整 API 特征之间存在很大的性能差距。这个差距不是实验缺陷，而是说明发布后 solved 统计携带了标签、题号和分值无法提供的信息。

年龄归一化解题数也只是简单代理。将解题数除以经过天数，无法完整建模随时间变化的非线性解题积累、比赛可见度、题目被教学复用的情况，或 Codeforces 活跃用户群体的变化。因此，年龄归一化结果应解释为有用的鲁棒性检查，而不是对曝光偏差的完整校正。

# 结论

在 10,979 道有评分编程题上，官方 API 元数据与解题统计能够支持较准确的难度预测。solved-count-only 是最强简单基线，但完整模型进一步改进了结果；消融实验也显示 solved 特征是测试过的公开信号组中最重要的一组。鲁棒性实验通过区分发布后预测与冷启动预测，并说明年龄归一化 solved-count 特征有用但不完整，从而增强了论文结论。该项目产物为后续加入题面文本、时间动态和比赛上下文特征提供了可复现基础。

# References

- Codeforces 官方公开 API 文档与本项目使用的公开 API 响应。
- 本项目在 `outputs/` 下生成的可复现实验产物。
- 建模与分析流水线使用 pandas、pyarrow、matplotlib 和 scikit-learn。
