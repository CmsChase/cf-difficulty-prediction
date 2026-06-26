"""Create paper-ready Markdown artifacts from completed project results."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pandas as pd

DEFAULT_OUTPUT_DIR: Final[Path] = Path("paper")
DEFAULT_RESULTS_OUTPUT_DIR: Final[Path] = Path("outputs/paper_tables")
DEFAULT_LOG_PATH: Final[Path] = Path("outputs/logs/paper_package.log")

JSON_INPUTS: Final[dict[str, Path]] = {
    "preprocess_summary": Path("data/processed/preprocess_summary.json"),
    "feature_summary": Path("data/processed/features/feature_summary.json"),
    "feature_columns": Path("data/processed/features/feature_columns.json"),
    "split_summary": Path("data/processed/splits/split_summary.json"),
    "dataset_summary": Path("outputs/eda/summary/dataset_summary.json"),
    "analysis_summary": Path("outputs/analysis/summary/analysis_summary.json"),
    "ablation_summary": Path("outputs/ablations/summary/ablation_summary.json"),
}

CSV_INPUTS: Final[dict[str, Path]] = {
    "main_results": Path("outputs/baselines/tables/main_results_table.csv"),
    "contest_grouped_metrics": Path(
        "outputs/baselines/metrics/contest_grouped_metrics.csv"
    ),
    "forward_time_metrics": Path("outputs/baselines/metrics/forward_time_metrics.csv"),
    "model_ranking_test": Path("outputs/analysis/tables/model_ranking_test.csv"),
    "baseline_improvements": Path(
        "outputs/analysis/tables/baseline_improvements.csv"
    ),
    "error_by_tag": Path("outputs/analysis/tables/error_by_tag.csv"),
    "error_by_index_rank": Path("outputs/analysis/tables/error_by_index_rank.csv"),
    "top_error_cases": Path("outputs/analysis/tables/top_error_cases.csv"),
    "ablation_metrics_test": Path("outputs/ablations/tables/ablation_metrics_test.csv"),
    "ablation_drop_comparison": Path(
        "outputs/ablations/tables/ablation_drop_comparison.csv"
    ),
    "feature_group_definitions": Path(
        "outputs/ablations/tables/feature_group_definitions.csv"
    ),
}

FIGURE_DIRS: Final[tuple[Path, ...]] = (
    Path("outputs/eda/figures"),
    Path("outputs/analysis/figures"),
    Path("outputs/ablations/figures"),
)

SECTION_ORDER: Final[tuple[str, ...]] = (
    "01_abstract",
    "02_introduction",
    "03_data",
    "04_methods",
    "05_results",
    "06_ablation",
    "07_error_analysis",
    "08_limitations",
    "09_conclusion",
)


class PaperPackageError(RuntimeError):
    """Raised when paper artifact generation cannot proceed."""


class JsonLogFormatter(logging.Formatter):
    """Format paper-package logs as JSON Lines."""

    def format(self, record: logging.LogRecord) -> str:
        """Return one machine-readable log record."""
        payload: dict[str, object] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
        details = getattr(record, "details", None)
        if isinstance(details, dict):
            payload.update(details)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """Create the dedicated structured paper-package logger."""
    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.paper_package")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(resolved_path, encoding="utf-8")
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    """Flush and close all handlers attached to a dedicated logger."""
    for handler in tuple(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def read_json_file(path: Path) -> dict[str, object]:
    """Read a JSON object from disk with a clear error on invalid shape."""
    if not path.exists():
        raise PaperPackageError(f"Missing required JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PaperPackageError(f"JSON file must contain an object: {path}")
    return payload


def read_csv_file(path: Path) -> pd.DataFrame:
    """Read a CSV table from disk with a clear missing-file error."""
    if not path.exists():
        raise PaperPackageError(f"Missing required CSV file: {path}")
    return pd.read_csv(path)


def load_result_files(
    json_inputs: Mapping[str, Path] = JSON_INPUTS,
    csv_inputs: Mapping[str, Path] = CSV_INPUTS,
) -> tuple[dict[str, dict[str, object]], dict[str, pd.DataFrame]]:
    """Load all JSON and CSV result files required for the paper package."""
    json_results = {name: read_json_file(path) for name, path in json_inputs.items()}
    csv_results = {name: read_csv_file(path) for name, path in csv_inputs.items()}
    return json_results, csv_results


def create_output_directories(
    output_dir: Path,
    results_output_dir: Path,
) -> dict[str, Path]:
    """Create and return all output directories used by the package step."""
    directories = {
        "paper": output_dir,
        "sections": output_dir / "sections",
        "figures": output_dir / "figures",
        "tables": output_dir / "tables",
        "results_tables": results_output_dir,
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _format_value(value: object) -> str:
    """Format one table cell for deterministic Markdown output."""
    if not isinstance(value, (list, tuple, dict)) and pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, dict)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def dataframe_to_markdown(
    frame: pd.DataFrame,
    *,
    max_rows: int | None = None,
) -> str:
    """Format a DataFrame as a GitHub-compatible Markdown table."""
    table = frame.copy()
    if max_rows is not None:
        table = table.head(max_rows)
    if table.empty:
        return "_No rows available._\n"
    columns = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 Markdown with normalized trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def _copy_table(
    frame: pd.DataFrame,
    *,
    markdown_path: Path,
    csv_path: Path,
    max_markdown_rows: int | None = None,
) -> None:
    """Write Markdown and CSV versions of one paper-ready table."""
    _write_text(markdown_path, dataframe_to_markdown(frame, max_rows=max_markdown_rows))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)


def _metric_row(metric: str, value: object) -> dict[str, object]:
    """Create one dataset-summary row."""
    return {"metric": metric, "value": value}


def build_dataset_summary_table(json_results: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    """Build the paper-ready dataset summary table."""
    dataset = json_results["dataset_summary"]
    preprocess = json_results["preprocess_summary"]
    split = json_results["split_summary"]
    quantiles = dataset["solved_count_quantiles"]
    rating_range = dataset["rating_range"]
    top_tags = dataset.get("top_tags", [])
    top_tag_text = ", ".join(
        f"{item['tag']} ({item['count']})" for item in top_tags[:5]
    )
    rows = [
        _metric_row(
            "Rated programming problems",
            preprocess["number_of_rated_programming_problems"],
        ),
        _metric_row(
            "Rating range",
            f"{rating_range['min']}–{rating_range['max']}",
        ),
        _metric_row("Unique contests", dataset["unique_contests"]["processed"]),
        _metric_row("Feature columns", json_results["feature_summary"]["feature_count"]),
        _metric_row("Tag one-hot columns", json_results["feature_summary"]["tag_feature_count"]),
        _metric_row("Solved count p50", quantiles["p50"]),
        _metric_row("Solved count p75", quantiles["p75"]),
        _metric_row("Solved count p90", quantiles["p90"]),
        _metric_row("Solved count p95", quantiles["p95"]),
        _metric_row("Solved count p99", quantiles["p99"]),
        _metric_row("Solved count max", quantiles["max"]),
        _metric_row(
            "Contest-grouped contest overlap",
            split["contest_grouped"]["contest_overlap_count"],
        ),
        _metric_row(
            "Forward-time strictly ordered",
            split["forward_time"]["strictly_ordered"],
        ),
        _metric_row("Most frequent tags", top_tag_text),
    ]
    return pd.DataFrame(rows)


def _select_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return existing columns from a frame in requested order."""
    return frame.loc[:, [column for column in columns if column in frame.columns]].copy()


def build_paper_tables(
    json_results: Mapping[str, Mapping[str, object]],
    csv_results: Mapping[str, pd.DataFrame],
    directories: Mapping[str, Path],
) -> dict[str, pd.DataFrame]:
    """Create all required paper table Markdown and CSV artifacts."""
    tables: dict[str, pd.DataFrame] = {
        "dataset_summary": build_dataset_summary_table(json_results),
        "main_model_results": _select_columns(
            csv_results["model_ranking_test"],
            [
                "strategy",
                "model_name",
                "MAE",
                "RMSE",
                "R2",
                "within_100",
                "within_200",
                "rank_by_MAE",
            ],
        ),
        "baseline_improvements": csv_results["baseline_improvements"].copy(),
        "ablation_drop_comparison": csv_results["ablation_drop_comparison"].copy(),
        "error_by_tag_top": (
            csv_results["error_by_tag"]
            .sort_values(["strategy", "mean_abs_error"], ascending=[True, False])
            .groupby("strategy", group_keys=False)
            .head(10)
            .reset_index(drop=True)
        ),
        "error_by_index_rank": csv_results["error_by_index_rank"].copy(),
    }
    max_rows = {
        "main_model_results": None,
        "baseline_improvements": None,
        "ablation_drop_comparison": None,
        "error_by_tag_top": None,
        "error_by_index_rank": None,
        "dataset_summary": None,
    }
    for name, table in tables.items():
        _copy_table(
            table,
            markdown_path=directories["tables"] / f"{name}.md",
            csv_path=directories["results_tables"] / f"{name}.csv",
            max_markdown_rows=max_rows[name],
        )
    return tables


def copy_key_figures(
    figure_dirs: Sequence[Path],
    destination_dir: Path,
) -> list[str]:
    """Copy available PNG figures into the paper figure directory."""
    copied: list[str] = []
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source_dir in figure_dirs:
        if not source_dir.exists():
            continue
        for source in sorted(source_dir.glob("*.png")):
            destination = destination_dir / source.name
            shutil.copy2(source, destination)
            copied.append(source.name)
    return sorted(set(copied))


def _ranking_line(csv_results: Mapping[str, pd.DataFrame], strategy: str) -> str:
    """Summarize simple baseline ordering for one strategy."""
    ranking = csv_results["model_ranking_test"]
    subset = ranking.loc[
        ranking["strategy"].eq(strategy)
        & ranking["model_name"].isin(
            [
                "solved_count_only_baseline",
                "index_only_baseline",
                "tag_only_baseline",
            ]
        )
    ].sort_values("MAE")
    return "; ".join(
        f"{row.model_name}: MAE {row.MAE:.1f}" for row in subset.itertuples()
    )


def _best_model_sentence(analysis: Mapping[str, object], strategy: str) -> str:
    """Return a factual best-model sentence fragment."""
    best = analysis["best_model_by_strategy"][strategy]
    return (
        f"{best['model_name']} achieved the lowest test MAE "
        f"({best['test_MAE']:.1f}) with within-200 accuracy "
        f"{best['within_200']:.3f}"
    )


def _best_drop_group(ablation_drop_comparison: pd.DataFrame) -> str:
    """Return the feature group whose removal most often hurts MAE."""
    positive = ablation_drop_comparison.sort_values(
        ["MAE_difference", "removed_group"],
        ascending=[False, True],
    )
    if positive.empty:
        return "no single group"
    return str(positive.iloc[0]["removed_group"])


def build_section_texts(
    json_results: Mapping[str, Mapping[str, object]],
    csv_results: Mapping[str, pd.DataFrame],
    copied_figures: Sequence[str],
) -> dict[str, dict[str, str]]:
    """Create bilingual section drafts using only loaded project results."""
    dataset = json_results["dataset_summary"]
    preprocess = json_results["preprocess_summary"]
    split = json_results["split_summary"]
    analysis = json_results["analysis_summary"]
    ablation = json_results["ablation_summary"]
    quantiles = dataset["solved_count_quantiles"]
    rating = dataset["rating_range"]
    rows = preprocess["number_of_rated_programming_problems"]
    best_drop_group = _best_drop_group(csv_results["ablation_drop_comparison"])
    figures = set(copied_figures)

    figure_refs = []
    if "test_mae_by_model.png" in figures:
        figure_refs.append("![Test MAE by model](figures/test_mae_by_model.png)")
    if "feature_drop_mae_change.png" in figures:
        figure_refs.append("![Feature-drop MAE change](figures/feature_drop_mae_change.png)")
    if "solved_count_hist_log.png" in figures:
        figure_refs.append("![Log solved-count distribution](figures/solved_count_hist_log.png)")

    en = {
        "01_abstract": (
            "# Abstract\n\n"
            "This project studies Codeforces problem-rating prediction from official API metadata and solved-statistics signals. "
            f"The current dataset contains {rows:,} rated programming problems with ratings from {rating['min']} to {rating['max']}. "
            f"The strongest simple baseline is solved-count-only in both evaluation settings ({_ranking_line(csv_results, 'contest_grouped')} for contest-grouped; {_ranking_line(csv_results, 'forward_time')} for forward-time). "
            f"Full models further improve over that baseline: {_best_model_sentence(analysis, 'contest_grouped')} on the contest-grouped split, while {_best_model_sentence(analysis, 'forward_time')} on the forward-time split. "
            f"Ablations indicate that removing {best_drop_group} features causes the largest observed MAE increase among one-group drops. "
            "The results support solved-statistics as a central signal while leaving room for metadata and tag features to improve prediction."
        ),
        "02_introduction": (
            "# Introduction\n\n"
            "Codeforces problem ratings are useful for curriculum design, recommendation, and contest analysis. "
            "However, ratings are only available after human and platform processes have assigned difficulty labels. "
            "This project asks how well public metadata and submission-behavior summaries can predict official ratings. "
            "The study is intentionally restricted to the official public Codeforces API, avoiding scraping and private data. "
            "The goal is not to replace official ratings, but to quantify which public signals are predictive and where errors remain."
        ),
        "03_data": (
            "# Data\n\n"
            f"The processed modeling table contains {rows:,} rated `PROGRAMMING` problems from {dataset['unique_contests']['processed']:,} contests. "
            f"Ratings range from {rating['min']} to {rating['max']}. "
            f"Solved counts are strongly skewed: p50 is {quantiles['p50']:.0f}, p99 is {quantiles['p99']:.0f}, and the maximum is {quantiles['max']:.0f}. "
            f"The contest-grouped split has {split['contest_grouped']['contest_overlap_count']} contest overlap between partitions, and the forward-time split is strictly ordered: {split['forward_time']['strictly_ordered']}.\n\n"
            "See Table `paper/tables/dataset_summary.md` for a compact dataset summary.\n\n"
            + ("\n\n".join(ref for ref in figure_refs if "solved-count" in ref) if figure_refs else "")
        ),
        "04_methods": (
            "# Methods\n\n"
            "The feature table preserves problem identifiers, official rating as the target, contest start time, index-derived features, point metadata, solved-count features, and one-hot tag indicators. "
            "Evaluation uses two complementary strategies. The contest-grouped split prevents contest leakage by assigning every contest to only one partition. "
            "The forward-time split orders contests chronologically to test temporal generalization. "
            "The model set includes simple baselines, ridge regression, random forest, and histogram gradient boosting in the baseline stage. "
            "The ablation stage evaluates ridge regression and histogram gradient boosting across predefined feature groups."
        ),
        "05_results": (
            "# Results\n\n"
            f"On the contest-grouped test split, {_best_model_sentence(analysis, 'contest_grouped')}. "
            f"On the forward-time test split, {_best_model_sentence(analysis, 'forward_time')}. "
            f"Among simple baselines, solved-count-only is strongest in both settings: {_ranking_line(csv_results, 'contest_grouped')} for contest-grouped and {_ranking_line(csv_results, 'forward_time')} for forward-time. "
            "The full models still improve over solved-count-only, indicating that additional metadata contributes beyond solved statistics. "
            "Forward-time train/test gaps are interpreted as temporal generalization gaps or distribution shift, not automatic evidence of overfitting.\n\n"
            "![Test MAE by model](figures/test_mae_by_model.png)\n\n"
            "![Within-200 by model](figures/within_200_by_model.png)"
        ),
        "06_ablation": (
            "# Ablation Study\n\n"
            f"The best overall ablation result is `{ablation['best_overall_ablation_by_test_MAE']['model_name']}` with `{ablation['best_overall_ablation_by_test_MAE']['feature_set_name']}` on `{ablation['best_overall_ablation_by_test_MAE']['strategy']}`, with test MAE {ablation['best_overall_ablation_by_test_MAE']['test_MAE']:.1f}. "
            f"The one-group drop comparison shows that removing `{best_drop_group}` features produces the largest MAE increase. "
            "This supports the central role of solved-count behavior while retaining the usefulness of metadata and tag information.\n\n"
            "![Ablation MAE contest grouped](figures/ablation_mae_by_feature_set_contest_grouped.png)\n\n"
            "![Ablation MAE forward time](figures/ablation_mae_by_feature_set_forward_time.png)\n\n"
            "![Feature drop MAE change](figures/feature_drop_mae_change.png)"
        ),
        "07_error_analysis": (
            "# Error Analysis\n\n"
            "Error tables summarize the largest absolute errors and aggregate errors by tag and index rank. "
            "These artifacts are intended to identify regions where public metadata is less sufficient, such as unusual special problems or tasks whose solved counts diverge from official rating. "
            "The analysis does not claim causal explanations; it provides diagnostic patterns for later qualitative review.\n\n"
            "![Error by tag contest grouped](figures/error_by_tag_top15_contest_grouped.png)\n\n"
            "![Error by index rank](figures/error_by_index_rank.png)"
        ),
        "08_limitations": (
            "# Limitations\n\n"
            "The dataset uses public Codeforces API fields only, so it lacks statement text, editorials, participant-level histories, and temporal details of when solves accumulated. "
            "Solved counts are strong predictors but can encode popularity, age, and exposure in addition to intrinsic difficulty. "
            "Forward-time evaluation partially addresses temporal generalization, but future snapshots may shift as Codeforces problem styles and participant populations change. "
            "The present analysis is best viewed as a reproducible baseline for structured public metadata rather than a final difficulty model."
        ),
        "09_conclusion": (
            "# Conclusion\n\n"
            f"Across {rows:,} rated programming problems, official API metadata and solved statistics support accurate rating prediction. "
            "Solved-count-only is the strongest simple baseline, yet full models improve over it, and ablations show that solved features are the most important group among the tested public signals. "
            "The resulting artifacts provide a reproducible foundation for future work on richer textual, temporal, and contest-context features."
        ),
    }

    cn = {
        "01_abstract": (
            "# 摘要\n\n"
            "本项目研究如何利用 Codeforces 官方 API 中的公开元数据与解题统计来预测题目官方难度评分。"
            f"当前数据集包含 {rows:,} 道有评分的编程题，评分范围为 {rating['min']} 到 {rating['max']}。"
            "在两个评估设置中，solved-count-only 都是最强的简单基线；完整模型仍进一步降低 MAE。"
            f"消融实验显示，移除 `{best_drop_group}` 特征带来的 MAE 增幅最大。"
            "这些结果说明解题统计是核心信号，但元数据和标签仍能提供额外预测信息。"
        ),
        "02_introduction": (
            "# 引言\n\n"
            "Codeforces 题目评分可用于学习路径设计、题目推荐和竞赛分析。"
            "本研究关注一个可复现的问题：仅使用官方公开 API 中的结构化信息，能否有效预测官方评分。"
            "项目不使用网页抓取、私有数据或题面文本；目标不是替代官方评分，而是量化公开信号的预测能力与局限。"
        ),
        "03_data": (
            "# 数据\n\n"
            f"建模表包含 {rows:,} 道有评分的 `PROGRAMMING` 题目，来自 {dataset['unique_contests']['processed']:,} 场比赛。"
            f"评分范围为 {rating['min']} 到 {rating['max']}。"
            f"解题数分布高度偏斜：p50 为 {quantiles['p50']:.0f}，p99 为 {quantiles['p99']:.0f}，最大值为 {quantiles['max']:.0f}。"
            f"contest-grouped 切分的比赛重叠数为 {split['contest_grouped']['contest_overlap_count']}；forward-time 切分严格按时间排序：{split['forward_time']['strictly_ordered']}。\n\n"
            "数据概览见 `paper/tables/dataset_summary.md`。\n\n"
            "![Log solved-count distribution](figures/solved_count_hist_log.png)"
        ),
        "04_methods": (
            "# 方法\n\n"
            "特征表保留题目标识、官方评分、比赛开始时间、题号派生特征、分值信息、解题数特征以及标签 one-hot 特征。"
            "评估采用两种切分：contest-grouped 用于避免同一比赛泄漏到多个集合；forward-time 用于检验时间外推能力。"
            "基线阶段包含简单基线、岭回归、随机森林和直方图梯度提升；消融阶段重点比较岭回归和直方图梯度提升在不同特征组下的表现。"
        ),
        "05_results": (
            "# 结果\n\n"
            f"contest-grouped 测试集上，{analysis['best_model_by_strategy']['contest_grouped']['model_name']} 的测试 MAE 最低，为 {analysis['best_model_by_strategy']['contest_grouped']['test_MAE']:.1f}。"
            f"forward-time 测试集上，{analysis['best_model_by_strategy']['forward_time']['model_name']} 的测试 MAE 最低，为 {analysis['best_model_by_strategy']['forward_time']['test_MAE']:.1f}。"
            "简单基线中，solved-count-only 强于 index-only 和 tag-only。"
            "完整模型仍优于 solved-count-only，说明其他结构化元数据仍有增益。"
            "forward-time 的训练/测试差距应被讨论为时间泛化差距或分布漂移，而不应自动解释为过拟合。\n\n"
            "![Test MAE by model](figures/test_mae_by_model.png)"
        ),
        "06_ablation": (
            "# 消融研究\n\n"
            f"最佳整体消融结果来自 `{ablation['best_overall_ablation_by_test_MAE']['strategy']}` 设置下的 `{ablation['best_overall_ablation_by_test_MAE']['model_name']}`，特征集合为 `{ablation['best_overall_ablation_by_test_MAE']['feature_set_name']}`，测试 MAE 为 {ablation['best_overall_ablation_by_test_MAE']['test_MAE']:.1f}。"
            f"单组移除实验显示，移除 `{best_drop_group}` 特征造成最大的 MAE 上升。"
            "这与 solved-count 信号的重要性一致。\n\n"
            "![Feature drop MAE change](figures/feature_drop_mae_change.png)"
        ),
        "07_error_analysis": (
            "# 错误分析\n\n"
            "错误分析表列出最大绝对误差案例，并按标签和题号等级聚合平均绝对误差。"
            "这些结果用于定位公开结构化特征不足的区域，例如特殊题型或解题数与官方评分不一致的题目。"
            "该分析不声称因果解释，而是为后续人工复核提供诊断线索。\n\n"
            "![Error by tag contest grouped](figures/error_by_tag_top15_contest_grouped.png)"
        ),
        "08_limitations": (
            "# 局限性\n\n"
            "本研究只使用 Codeforces 官方公开 API 字段，不包含题面文本、题解、用户历史或解题时间序列。"
            "解题数虽然预测能力强，但同时反映题目曝光度、发布时间和流行度，不完全等同于内在难度。"
            "forward-time 切分能部分检验时间泛化，但未来数据仍可能受题型风格和参赛群体变化影响。"
        ),
        "09_conclusion": (
            "# 结论\n\n"
            f"在 {rows:,} 道有评分编程题上，官方 API 元数据与解题统计能够支持较准确的难度预测。"
            "solved-count-only 是最强简单基线，但完整模型进一步改进了结果；消融实验也显示 solved 特征是最重要的公开信号组。"
            "这些产物为后续加入文本、时间动态和比赛上下文特征提供了可复现基础。"
        ),
    }
    return {"en": en, "cn": cn}


def write_sections_and_papers(
    sections: Mapping[str, Mapping[str, str]],
    directories: Mapping[str, Path],
) -> None:
    """Write individual section files and combined English/Chinese papers."""
    for language, language_sections in sections.items():
        for section_name in SECTION_ORDER:
            _write_text(
                directories["sections"] / f"{section_name}_{language}.md",
                language_sections[section_name],
            )
        combined = "\n\n".join(language_sections[name] for name in SECTION_ORDER)
        _write_text(directories["paper"] / f"paper_{language}.md", combined)


def build_reproducibility_text() -> str:
    """Return the reproducibility record with exact full-pipeline commands."""
    return """# Reproducibility Record

Results are preliminary until independently reproduced.

Run from the project root with Python and the project dependencies installed:

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

Fetch official Codeforces API data:

```powershell
python -m cf_diff.fetch_api --output-root data/raw/codeforces --lang en --sleep-seconds 2.1 --timeout 30 --latest-dir data/raw/codeforces/latest
```

Preprocess raw data:

```powershell
python -m cf_diff.preprocess --raw-dir data/raw/codeforces/latest --interim-dir data/interim --processed-dir data/processed --log-path outputs/logs/preprocess.log
```

Build features:

```powershell
python -m cf_diff.features --config configs/experiment.yaml --input-path data/processed/rated_programming_problems.parquet --output-dir data/processed/features --log-path outputs/logs/features.log
```

Generate splits:

```powershell
python -m cf_diff.splits --config configs/experiment.yaml --input-path data/processed/features/model_table.parquet --output-dir data/processed/splits --log-path outputs/logs/splits.log
```

Run EDA:

```powershell
python -m cf_diff.eda --config configs/experiment.yaml --processed-path data/processed/rated_programming_problems.parquet --feature-path data/processed/features/model_table.parquet --contest-split-path data/processed/splits/contest_grouped_split.parquet --time-split-path data/processed/splits/forward_time_split.parquet --output-dir outputs/eda --log-path outputs/logs/eda.log
```

Train baselines:

```powershell
python -m cf_diff.baselines --config configs/experiment.yaml --feature-path data/processed/features/model_table.parquet --feature-columns-path data/processed/features/feature_columns.json --contest-split-path data/processed/splits/contest_grouped_split.parquet --time-split-path data/processed/splits/forward_time_split.parquet --output-dir outputs/baselines --log-path outputs/logs/baselines.log
```

Analyze baseline outputs:

```powershell
python -m cf_diff.analysis --config configs/experiment.yaml --feature-path data/processed/features/model_table.parquet --feature-columns-path data/processed/features/feature_columns.json --contest-split-path data/processed/splits/contest_grouped_split.parquet --time-split-path data/processed/splits/forward_time_split.parquet --baseline-metrics-dir outputs/baselines/metrics --baseline-predictions-dir outputs/baselines/predictions --output-dir outputs/analysis --log-path outputs/logs/analysis.log
```

Run ablations:

```powershell
python -m cf_diff.ablations --config configs/experiment.yaml --feature-path data/processed/features/model_table.parquet --feature-columns-path data/processed/features/feature_columns.json --contest-split-path data/processed/splits/contest_grouped_split.parquet --time-split-path data/processed/splits/forward_time_split.parquet --output-dir outputs/ablations --log-path outputs/logs/ablations.log
```

Generate the paper package:

```powershell
python -m cf_diff.paper_package --output-dir paper --results-output-dir outputs/paper_tables --log-path outputs/logs/paper_package.log
```

Run tests:

```powershell
python -m pytest -q
```
"""


def run_paper_package(
    *,
    output_dir: Path,
    results_output_dir: Path,
    log_path: Path,
) -> dict[str, Path]:
    """Create the complete paper artifact package."""
    logger = configure_logger(log_path)
    try:
        directories = create_output_directories(output_dir, results_output_dir)
        json_results, csv_results = load_result_files()
        copied_figures = copy_key_figures(FIGURE_DIRS, directories["figures"])
        build_paper_tables(json_results, csv_results, directories)
        sections = build_section_texts(json_results, csv_results, copied_figures)
        write_sections_and_papers(sections, directories)
        _write_text(directories["paper"] / "reproducibility.md", build_reproducibility_text())
        paths = {
            "paper_en": directories["paper"] / "paper_en.md",
            "paper_cn": directories["paper"] / "paper_cn.md",
            "reproducibility": directories["paper"] / "reproducibility.md",
        }
        logger.info(
            "Completed paper artifact package",
            extra={
                "event": "paper_package_completed",
                "details": {
                    "output_dir": directories["paper"].as_posix(),
                    "results_output_dir": directories["results_tables"].as_posix(),
                    "figure_count": len(copied_figures),
                },
            },
        )
        return paths
    except Exception:
        logger.exception(
            "Paper artifact package failed",
            extra={"event": "paper_package_failed"},
        )
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the paper-package command-line parser."""
    parser = argparse.ArgumentParser(
        description="Create paper-ready Markdown artifacts from project results."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--results-output-dir",
        type=Path,
        default=DEFAULT_RESULTS_OUTPUT_DIR,
    )
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the paper package CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        paths = run_paper_package(
            output_dir=args.output_dir,
            results_output_dir=args.results_output_dir,
            log_path=args.log_path,
        )
    except (PaperPackageError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote English paper draft: {paths['paper_en']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
