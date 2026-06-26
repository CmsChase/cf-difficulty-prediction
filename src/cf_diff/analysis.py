"""Analyze baseline experiment outputs for Codeforces rating prediction."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cf_diff.features import write_json

DEFAULT_CONFIG_PATH: Final[Path] = Path("configs/experiment.yaml")
DEFAULT_FEATURE_PATH: Final[Path] = Path(
    "data/processed/features/model_table.parquet"
)
DEFAULT_FEATURE_COLUMNS_PATH: Final[Path] = Path(
    "data/processed/features/feature_columns.json"
)
DEFAULT_CONTEST_SPLIT_PATH: Final[Path] = Path(
    "data/processed/splits/contest_grouped_split.parquet"
)
DEFAULT_TIME_SPLIT_PATH: Final[Path] = Path(
    "data/processed/splits/forward_time_split.parquet"
)
DEFAULT_BASELINE_METRICS_DIR: Final[Path] = Path("outputs/baselines/metrics")
DEFAULT_BASELINE_PREDICTIONS_DIR: Final[Path] = Path(
    "outputs/baselines/predictions"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path("outputs/analysis")
DEFAULT_LOG_PATH: Final[Path] = Path("outputs/logs/analysis.log")
STRATEGIES: Final[tuple[str, str]] = ("contest_grouped", "forward_time")
STANDARD_BASELINES: Final[tuple[str, ...]] = (
    "mean_baseline",
    "index_only_baseline",
    "tag_only_baseline",
    "solved_count_only_baseline",
)
METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "MAE",
    "RMSE",
    "R2",
    "within_100",
    "within_200",
)
IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (
    "contest_id",
    "index",
)


class AnalysisError(RuntimeError):
    """Raised when analysis artifacts cannot be generated safely."""


class JsonLogFormatter(logging.Formatter):
    """Format analysis logs as JSON Lines."""

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
    """Create the dedicated structured analysis logger."""
    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.analysis")
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


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    table_name: str,
) -> None:
    """Raise a clear error when required columns are absent."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise AnalysisError(f"{table_name} lacks required columns: {missing}")


def _finite_float(value: object, digits: int = 6) -> float | None:
    """Convert numeric output to a rounded JSON-safe float."""
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    """Load both baseline metrics CSV files."""
    frames = []
    for strategy in STRATEGIES:
        path = metrics_dir / f"{strategy}_metrics.csv"
        if not path.exists():
            raise AnalysisError(f"Missing metrics file: {path}")
        frame = pd.read_csv(path)
        _require_columns(
            frame,
            (
                "strategy",
                "model_name",
                "split_name",
                "row_count",
                "feature_count",
                *METRIC_COLUMNS,
            ),
            path.name,
        )
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    for column in ("row_count", "feature_count", *METRIC_COLUMNS):
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    return metrics


def build_model_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank models by test MAE separately for each split strategy."""
    _require_columns(
        metrics,
        (
            "strategy",
            "model_name",
            "split_name",
            "row_count",
            "feature_count",
            *METRIC_COLUMNS,
        ),
        "metrics",
    )
    ranking = metrics.loc[metrics["split_name"].eq("test")].copy()
    ranking = ranking.loc[
        :,
        [
            "strategy",
            "model_name",
            *METRIC_COLUMNS,
            "feature_count",
            "row_count",
        ],
    ]
    ranking = ranking.sort_values(
        ["strategy", "MAE", "model_name"],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking["rank_by_MAE"] = (
        ranking.groupby("strategy").cumcount().add(1).astype(int)
    )
    return ranking.loc[
        :,
        [
            "strategy",
            "model_name",
            *METRIC_COLUMNS,
            "feature_count",
            "row_count",
            "rank_by_MAE",
        ],
    ]


def best_model_by_strategy(ranking: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Return the best test-MAE model for each split strategy."""
    best: dict[str, dict[str, object]] = {}
    for strategy, group in ranking.groupby("strategy", sort=True):
        row = group.sort_values(["rank_by_MAE", "model_name"]).iloc[0]
        best[strategy] = {
            "model_name": str(row["model_name"]),
            "test_MAE": _finite_float(row["MAE"]),
            "test_RMSE": _finite_float(row["RMSE"]),
            "test_R2": _finite_float(row["R2"]),
            "within_100": _finite_float(row["within_100"]),
            "within_200": _finite_float(row["within_200"]),
        }
    return best


def _best_full_model_row(strategy_ranking: pd.DataFrame) -> pd.Series:
    """Select the best non-standard-baseline row, falling back to best overall."""
    full_models = strategy_ranking.loc[
        ~strategy_ranking["model_name"].isin(STANDARD_BASELINES)
    ]
    if full_models.empty:
        full_models = strategy_ranking
    return full_models.sort_values(["MAE", "model_name"], kind="mergesort").iloc[0]


def build_baseline_improvements(ranking: pd.DataFrame) -> pd.DataFrame:
    """Compare the best full model against standard simple baselines."""
    rows: list[dict[str, object]] = []
    for strategy, group in ranking.groupby("strategy", sort=True):
        best_full = _best_full_model_row(group)
        best_mae = float(best_full["MAE"])
        for baseline_name in STANDARD_BASELINES:
            baseline = group.loc[group["model_name"].eq(baseline_name)]
            if baseline.empty:
                continue
            baseline_mae = float(baseline.iloc[0]["MAE"])
            absolute = baseline_mae - best_mae
            percent = (absolute / baseline_mae * 100.0) if baseline_mae else np.nan
            rows.append(
                {
                    "strategy": strategy,
                    "best_full_model": str(best_full["model_name"]),
                    "comparison_model": baseline_name,
                    "best_full_model_MAE": round(best_mae, 6),
                    "comparison_model_MAE": round(baseline_mae, 6),
                    "absolute_MAE_improvement": round(float(absolute), 6),
                    "percent_MAE_improvement": round(float(percent), 6),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["strategy", "comparison_model"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_train_test_gap_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize train/test MAE gaps without assuming their cause."""
    pivot = metrics.pivot_table(
        index=["strategy", "model_name"],
        columns="split_name",
        values="MAE",
        aggfunc="first",
    ).reset_index()
    rows: list[dict[str, object]] = []
    for row in pivot.to_dict(orient="records"):
        train = row.get("train")
        test = row.get("test")
        if pd.isna(train) or pd.isna(test):
            continue
        gap = float(test) - float(train)
        ratio = float(test) / float(train) if float(train) else np.nan
        suggests = bool(gap > max(50.0, 0.20 * float(train)))
        rows.append(
            {
                "strategy": row["strategy"],
                "model_name": row["model_name"],
                "train_MAE": round(float(train), 6),
                "test_MAE": round(float(test), 6),
                "test_minus_train_MAE": round(gap, 6),
                "test_to_train_MAE_ratio": round(float(ratio), 6),
                "suggests_overfitting": suggests,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["strategy", "test_minus_train_MAE", "model_name"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _read_feature_metadata(path: Path) -> Mapping[str, object]:
    """Load feature metadata when available."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _tag_columns_from_metadata(
    feature_frame: pd.DataFrame,
    metadata: Mapping[str, object],
) -> dict[str, str]:
    """Return mapping from one-hot columns to paper-readable tag names."""
    raw_map = metadata.get("tag_feature_map")
    if isinstance(raw_map, dict):
        return {
            str(column): str(tag)
            for tag, column in raw_map.items()
            if isinstance(column, str) and column in feature_frame.columns
        }
    return {
        column: column.removeprefix("tag__").replace("_", " ")
        for column in feature_frame.columns
        if column.startswith("tag__")
    }


def enrich_predictions(
    predictions: pd.DataFrame,
    feature_frame: pd.DataFrame,
    feature_metadata: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Join prediction rows with feature columns useful for error analysis."""
    _require_columns(
        predictions,
        (
            "contest_id",
            "index",
            "name",
            "actual_rating",
            "predicted_rating",
            "model_name",
            "split_name",
        ),
        "prediction table",
    )
    metadata = feature_metadata or {}
    useful_columns = [
        "contest_id",
        "index",
        "solved_count",
        "log_solved_count",
        "tag_count",
        "index_rank",
    ]
    if "tags" in feature_frame.columns:
        useful_columns.append("tags")
    tag_column_map = _tag_columns_from_metadata(feature_frame, metadata)
    useful_columns.extend(sorted(tag_column_map))
    existing_columns = [column for column in useful_columns if column in feature_frame]
    enriched = predictions.merge(
        feature_frame.loc[:, existing_columns],
        on=["contest_id", "index"],
        how="left",
        validate="many_to_one",
    )
    if "tags" not in enriched.columns and tag_column_map:
        tag_columns = sorted(tag_column_map)

        def _row_tags(row: pd.Series) -> list[str]:
            return [
                tag_column_map[column]
                for column in tag_columns
                if column in row and pd.notna(row[column]) and int(row[column]) == 1
            ]

        enriched["tags"] = enriched.apply(_row_tags, axis=1)
    enriched["rating"] = pd.to_numeric(enriched["actual_rating"], errors="coerce")
    enriched["prediction"] = pd.to_numeric(
        enriched["predicted_rating"],
        errors="coerce",
    )
    enriched["error"] = enriched["prediction"] - enriched["rating"]
    enriched["abs_error"] = enriched["error"].abs()
    return enriched


def load_predictions(
    predictions_dir: Path,
    feature_frame: pd.DataFrame,
    feature_metadata: Mapping[str, object],
) -> dict[str, pd.DataFrame]:
    """Load and enrich prediction files by strategy."""
    result: dict[str, pd.DataFrame] = {}
    for strategy in STRATEGIES:
        path = predictions_dir / f"{strategy}_predictions.parquet"
        if not path.exists():
            raise AnalysisError(f"Missing prediction file: {path}")
        result[strategy] = enrich_predictions(
            pd.read_parquet(path, engine="pyarrow"),
            feature_frame,
            feature_metadata,
        )
    return result


def build_top_error_cases(
    predictions_by_strategy: Mapping[str, pd.DataFrame],
    best_models: Mapping[str, str],
    *,
    top_n: int = 50,
) -> pd.DataFrame:
    """Return largest absolute test errors for the best model in each strategy."""
    rows = []
    for strategy, predictions in predictions_by_strategy.items():
        best_model = best_models[strategy]
        subset = predictions.loc[
            predictions["split_name"].eq("test")
            & predictions["model_name"].eq(best_model)
        ].copy()
        subset["strategy"] = strategy
        rows.append(
            subset.sort_values(
                ["abs_error", "contest_id", "index"],
                ascending=[False, True, True],
                kind="mergesort",
            ).head(top_n)
        )
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    output_columns = [
        "strategy",
        "model_name",
        "contest_id",
        "index",
        "name",
        "rating",
        "prediction",
        "error",
        "abs_error",
        "solved_count",
        "log_solved_count",
        "tag_count",
        "tags",
    ]
    for column in output_columns:
        if column not in combined.columns:
            combined[column] = pd.NA
    return combined.loc[:, output_columns].reset_index(drop=True)


def _explode_tags(frame: pd.DataFrame) -> pd.DataFrame:
    """Explode list-like tags for grouped error summaries."""
    if "tags" not in frame.columns:
        return pd.DataFrame(columns=[*frame.columns, "tag"])
    exploded = frame.copy()
    exploded["tag"] = exploded["tags"].map(
        lambda value: value if isinstance(value, list) else []
    )
    exploded = exploded.explode("tag")
    return exploded.loc[exploded["tag"].notna() & exploded["tag"].ne("")]


def build_error_by_tag(
    predictions_by_strategy: Mapping[str, pd.DataFrame],
    best_models: Mapping[str, str],
    *,
    min_count: int = 30,
) -> pd.DataFrame:
    """Compute mean absolute error by tag for each best strategy model."""
    rows = []
    for strategy, predictions in predictions_by_strategy.items():
        subset = predictions.loc[
            predictions["split_name"].eq("test")
            & predictions["model_name"].eq(best_models[strategy])
        ].copy()
        exploded = _explode_tags(subset)
        if exploded.empty:
            continue
        grouped = (
            exploded.groupby("tag", dropna=False)
            .agg(count=("abs_error", "size"), mean_abs_error=("abs_error", "mean"))
            .reset_index()
        )
        grouped = grouped.loc[grouped["count"] >= min_count].copy()
        grouped["strategy"] = strategy
        grouped["model_name"] = best_models[strategy]
        rows.append(grouped)
    if not rows:
        return pd.DataFrame(
            columns=["strategy", "model_name", "tag", "count", "mean_abs_error"]
        )
    result = pd.concat(rows, ignore_index=True)
    result["mean_abs_error"] = result["mean_abs_error"].round(6)
    return result.loc[
        :,
        ["strategy", "model_name", "tag", "count", "mean_abs_error"],
    ].sort_values(
        ["strategy", "mean_abs_error", "tag"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_error_by_index_rank(
    predictions_by_strategy: Mapping[str, pd.DataFrame],
    best_models: Mapping[str, str],
) -> pd.DataFrame:
    """Compute mean absolute error by problem index rank for best models."""
    rows = []
    for strategy, predictions in predictions_by_strategy.items():
        subset = predictions.loc[
            predictions["split_name"].eq("test")
            & predictions["model_name"].eq(best_models[strategy])
        ].copy()
        if "index_rank" not in subset.columns:
            continue
        grouped = (
            subset.groupby("index_rank", dropna=False)
            .agg(count=("abs_error", "size"), mean_abs_error=("abs_error", "mean"))
            .reset_index()
        )
        grouped["strategy"] = strategy
        grouped["model_name"] = best_models[strategy]
        rows.append(grouped)
    if not rows:
        return pd.DataFrame(
            columns=[
                "strategy",
                "model_name",
                "index_rank",
                "count",
                "mean_abs_error",
            ]
        )
    result = pd.concat(rows, ignore_index=True)
    result["mean_abs_error"] = result["mean_abs_error"].round(6)
    return result.loc[
        :,
        ["strategy", "model_name", "index_rank", "count", "mean_abs_error"],
    ].sort_values(
        ["strategy", "index_rank"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_analysis_summary(
    ranking: pd.DataFrame,
    improvements: pd.DataFrame,
    train_test_gap: pd.DataFrame,
) -> dict[str, object]:
    """Build the machine-readable analysis summary."""
    best = best_model_by_strategy(ranking)
    rankings = {
        strategy: group.sort_values("rank_by_MAE").to_dict(orient="records")
        for strategy, group in ranking.groupby("strategy", sort=True)
    }
    baseline_rankings = {
        strategy: (
            group.loc[group["model_name"].isin(STANDARD_BASELINES)]
            .sort_values(["MAE", "model_name"], kind="mergesort")
            .loc[:, ["model_name", "MAE", "rank_by_MAE"]]
            .to_dict(orient="records")
        )
        for strategy, group in ranking.groupby("strategy", sort=True)
    }
    generalization_gap_notes = []
    overfitting_notes = []
    for row in train_test_gap.to_dict(orient="records"):
        if bool(row["suggests_overfitting"]):
            is_forward_time = row["strategy"] == "forward_time"
            is_simple_baseline = row["model_name"] in STANDARD_BASELINES
            if is_forward_time:
                note = (
                    "Test MAE is materially higher than train MAE in the "
                    "forward-time split; this is a generalization gap and may "
                    "reflect temporal distribution shift."
                )
                category = "temporal_generalization_gap"
            elif is_simple_baseline:
                note = (
                    "Test MAE is materially higher than train MAE; this is a "
                    "generalization gap for a simple baseline rather than a "
                    "model-capacity diagnosis."
                )
                category = "simple_baseline_generalization_gap"
            else:
                note = (
                    "Test MAE is materially higher than train MAE in the "
                    "contest-grouped split; this suggests possible overfitting."
                )
                category = "possible_overfitting"
            gap_note = {
                "strategy": row["strategy"],
                "model_name": row["model_name"],
                "gap_type": category,
                "note": note,
                "train_MAE": row["train_MAE"],
                "test_MAE": row["test_MAE"],
                "test_minus_train_MAE": row["test_minus_train_MAE"],
            }
            generalization_gap_notes.append(gap_note)
            if category == "possible_overfitting":
                overfitting_notes.append(gap_note)
    if not generalization_gap_notes:
        generalization_gap_notes.append(
            {
                "gap_type": "none_flagged",
                "note": (
                    "No model crossed the configured train/test MAE-gap "
                    "heuristic for a material generalization gap."
                ),
            }
        )
    if not overfitting_notes:
        overfitting_notes.append(
            {
                "note": (
                    "No contest-grouped non-baseline model crossed the "
                    "configured train/test MAE-gap heuristic for possible "
                    "overfitting."
                )
            }
        )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_model_by_strategy": best,
        "model_ranking_by_test_MAE": rankings,
        "standard_baseline_ranking_by_test_MAE": baseline_rankings,
        "baseline_improvements": improvements.to_dict(orient="records"),
        "train_test_gap": train_test_gap.to_dict(orient="records"),
        "generalization_gap_notes": generalization_gap_notes,
        "overfitting_notes": overfitting_notes,
    }


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Persist a matplotlib figure with stable output settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "cf-diff-analysis"},
    )
    plt.close(fig)


def _empty_figure(title: str) -> plt.Figure:
    """Create a readable placeholder figure for empty inputs."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title(title)
    ax.set_axis_off()
    return fig


def plot_metric_by_model(
    ranking: pd.DataFrame,
    metric: str,
    path: Path,
    *,
    title: str,
) -> None:
    """Save grouped bar chart for a test-set metric by model."""
    if ranking.empty:
        _save_figure(_empty_figure(title), path)
        return
    models = ranking.sort_values(["rank_by_MAE", "model_name"])[
        "model_name"
    ].drop_duplicates().tolist()
    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for offset, strategy in enumerate(STRATEGIES):
        values = (
            ranking.loc[ranking["strategy"].eq(strategy)]
            .set_index("model_name")
            .reindex(models)[metric]
        )
        ax.bar(
            x + (offset - 0.5) * width,
            values,
            width=width,
            label=strategy,
        )
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save_figure(fig, path)


def plot_predicted_vs_actual(
    predictions: pd.DataFrame,
    model_name: str,
    path: Path,
    *,
    title: str,
) -> None:
    """Save predicted-vs-actual scatter for one best test-set model."""
    subset = predictions.loc[
        predictions["split_name"].eq("test")
        & predictions["model_name"].eq(model_name)
    ].dropna(subset=["rating", "prediction"])
    if subset.empty:
        _save_figure(_empty_figure(title), path)
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        subset["rating"],
        subset["prediction"],
        s=12,
        alpha=0.35,
        edgecolors="none",
    )
    low = float(min(subset["rating"].min(), subset["prediction"].min()))
    high = float(max(subset["rating"].max(), subset["prediction"].max()))
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Actual rating")
    ax.set_ylabel("Predicted rating")
    ax.grid(alpha=0.25)
    _save_figure(fig, path)


def plot_error_by_tag(error_by_tag: pd.DataFrame, strategy: str, path: Path) -> None:
    """Save top-15 highest-error tags for one strategy."""
    subset = (
        error_by_tag.loc[error_by_tag["strategy"].eq(strategy)]
        .sort_values(["mean_abs_error", "tag"], ascending=[False, True])
        .head(15)
        .sort_values(["mean_abs_error", "tag"], ascending=[True, False])
    )
    title = f"Mean absolute error by tag: {strategy}"
    if subset.empty:
        _save_figure(_empty_figure(title), path)
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(subset["tag"], subset["mean_abs_error"])
    ax.set_title(title)
    ax.set_xlabel("Mean absolute error")
    ax.set_ylabel("Tag")
    ax.grid(axis="x", alpha=0.25)
    _save_figure(fig, path)


def plot_error_by_index_rank(error_by_index_rank: pd.DataFrame, path: Path) -> None:
    """Save mean absolute error by index rank for both strategies."""
    title = "Mean absolute error by index rank"
    if error_by_index_rank.empty:
        _save_figure(_empty_figure(title), path)
        return
    pivot = error_by_index_rank.pivot_table(
        index="index_rank",
        columns="strategy",
        values="mean_abs_error",
        aggfunc="first",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(ax=ax, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Index rank")
    ax.set_ylabel("Mean absolute error")
    ax.grid(alpha=0.25)
    _save_figure(fig, path)


def run_analysis(
    *,
    config_path: Path,
    feature_path: Path,
    feature_columns_path: Path,
    contest_split_path: Path,
    time_split_path: Path,
    baseline_metrics_dir: Path,
    baseline_predictions_dir: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Path]:
    """Run baseline result analysis and write paper-ready artifacts."""
    del config_path, contest_split_path, time_split_path
    logger = configure_logger(log_path)
    try:
        metrics = load_metrics(baseline_metrics_dir)
        ranking = build_model_ranking(metrics)
        improvements = build_baseline_improvements(ranking)
        train_test_gap = build_train_test_gap_summary(metrics)
        best_models = {
            strategy: row["model_name"]
            for strategy, row in ranking.loc[
                ranking["rank_by_MAE"].eq(1)
            ].set_index("strategy").to_dict(orient="index").items()
        }
        feature_frame = pd.read_parquet(feature_path, engine="pyarrow")
        feature_metadata = _read_feature_metadata(feature_columns_path)
        predictions_by_strategy = load_predictions(
            baseline_predictions_dir,
            feature_frame,
            feature_metadata,
        )
        top_errors = build_top_error_cases(predictions_by_strategy, best_models)
        error_by_tag = build_error_by_tag(predictions_by_strategy, best_models)
        error_by_index = build_error_by_index_rank(
            predictions_by_strategy,
            best_models,
        )
        summary = build_analysis_summary(ranking, improvements, train_test_gap)

        output_dir = output_dir.resolve()
        summary_dir = output_dir / "summary"
        tables_dir = output_dir / "tables"
        figures_dir = output_dir / "figures"
        for directory in (summary_dir, tables_dir, figures_dir):
            directory.mkdir(parents=True, exist_ok=True)

        paths = {
            "analysis_summary": summary_dir / "analysis_summary.json",
            "model_ranking_test": tables_dir / "model_ranking_test.csv",
            "baseline_improvements": tables_dir / "baseline_improvements.csv",
            "top_error_cases": tables_dir / "top_error_cases.csv",
            "error_by_tag": tables_dir / "error_by_tag.csv",
            "error_by_index_rank_table": tables_dir / "error_by_index_rank.csv",
            "test_mae_by_model": figures_dir / "test_mae_by_model.png",
            "within_200_by_model": figures_dir / "within_200_by_model.png",
            "predicted_vs_actual_contest_grouped": (
                figures_dir / "predicted_vs_actual_contest_grouped.png"
            ),
            "predicted_vs_actual_forward_time": (
                figures_dir / "predicted_vs_actual_forward_time.png"
            ),
            "error_by_tag_top15_contest_grouped": (
                figures_dir / "error_by_tag_top15_contest_grouped.png"
            ),
            "error_by_tag_top15_forward_time": (
                figures_dir / "error_by_tag_top15_forward_time.png"
            ),
            "error_by_index_rank_figure": figures_dir / "error_by_index_rank.png",
        }

        write_json(paths["analysis_summary"], summary)
        ranking.to_csv(paths["model_ranking_test"], index=False)
        improvements.to_csv(paths["baseline_improvements"], index=False)
        top_errors.to_csv(paths["top_error_cases"], index=False)
        error_by_tag.to_csv(paths["error_by_tag"], index=False)
        error_by_index.to_csv(paths["error_by_index_rank_table"], index=False)

        plot_metric_by_model(
            ranking,
            "MAE",
            paths["test_mae_by_model"],
            title="Test MAE by model",
        )
        plot_metric_by_model(
            ranking,
            "within_200",
            paths["within_200_by_model"],
            title="Within-200 accuracy by model",
        )
        for strategy in STRATEGIES:
            plot_predicted_vs_actual(
                predictions_by_strategy[strategy],
                best_models[strategy],
                paths[f"predicted_vs_actual_{strategy}"],
                title=f"Predicted vs actual rating: {strategy}",
            )
            plot_error_by_tag(
                error_by_tag,
                strategy,
                paths[f"error_by_tag_top15_{strategy}"],
            )
        plot_error_by_index_rank(error_by_index, paths["error_by_index_rank_figure"])

        logger.info(
            "Completed Codeforces baseline analysis",
            extra={
                "event": "analysis_completed",
                "details": {
                    "output_dir": output_dir.as_posix(),
                    "strategies": list(STRATEGIES),
                },
            },
        )
        return paths
    except Exception:
        logger.exception(
            "Codeforces baseline analysis failed",
            extra={"event": "analysis_failed"},
        )
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the analysis command-line parser."""
    parser = argparse.ArgumentParser(
        description="Analyze Codeforces baseline experiment outputs."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument(
        "--feature-columns-path",
        type=Path,
        default=DEFAULT_FEATURE_COLUMNS_PATH,
    )
    parser.add_argument(
        "--contest-split-path",
        type=Path,
        default=DEFAULT_CONTEST_SPLIT_PATH,
    )
    parser.add_argument(
        "--time-split-path",
        type=Path,
        default=DEFAULT_TIME_SPLIT_PATH,
    )
    parser.add_argument(
        "--baseline-metrics-dir",
        type=Path,
        default=DEFAULT_BASELINE_METRICS_DIR,
    )
    parser.add_argument(
        "--baseline-predictions-dir",
        type=Path,
        default=DEFAULT_BASELINE_PREDICTIONS_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the analysis CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        paths = run_analysis(
            config_path=args.config,
            feature_path=args.feature_path,
            feature_columns_path=args.feature_columns_path,
            contest_split_path=args.contest_split_path,
            time_split_path=args.time_split_path,
            baseline_metrics_dir=args.baseline_metrics_dir,
            baseline_predictions_dir=args.baseline_predictions_dir,
            output_dir=args.output_dir,
            log_path=args.log_path,
        )
    except (AnalysisError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote analysis summary: {paths['analysis_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
