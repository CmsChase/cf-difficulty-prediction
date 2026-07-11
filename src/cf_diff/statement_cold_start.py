"""Cold-start experiments with lightweight statement-structure features."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cf_diff import RANDOM_SEED
from cf_diff.baselines import (
    compute_regression_metrics,
    load_baseline_config,
    make_preprocessed_estimator,
)
from cf_diff.features import write_json
from cf_diff.model_selection import (
    DEFAULT_METRIC_COLUMNS,
    build_validation_ranked_report,
    select_rank_one,
)
from cf_diff.statement_features import (
    STATEMENT_FEATURE_COLUMNS as TEXT_LIGHT_FEATURE_COLUMNS,
)

DEFAULT_CONFIG_PATH: Final[Path] = Path("configs/experiment.yaml")
DEFAULT_FEATURE_PATH: Final[Path] = Path(
    "data/processed/features/model_table.parquet"
)
DEFAULT_FEATURE_COLUMNS_PATH: Final[Path] = Path(
    "data/processed/features/feature_columns.json"
)
DEFAULT_STATEMENT_FEATURE_PATH: Final[Path] = Path(
    "data/processed/statement_features/statement_features.parquet"
)
DEFAULT_CONTEST_SPLIT_PATH: Final[Path] = Path(
    "data/processed/splits/contest_grouped_split.parquet"
)
DEFAULT_TIME_SPLIT_PATH: Final[Path] = Path(
    "data/processed/splits/forward_time_split.parquet"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path("outputs/statement_cold_start")
DEFAULT_LOG_PATH: Final[Path] = Path("outputs/logs/statement_cold_start.log")

TARGET_COLUMN: Final[str] = "rating"
SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "valid", "test")
STRATEGIES: Final[tuple[str, str]] = ("contest_grouped", "forward_time")
IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (
    "contest_id",
    "index",
    "name",
    "start_time_seconds",
)
FEATURE_SETTING_ORDER: Final[tuple[str, ...]] = (
    "metadata_only",
    "text_light_only",
    "metadata_plus_text_light",
    "full_api_reference",
)
FEATURE_SETTING_LABELS: Final[dict[str, str]] = {
    "metadata_only": "metadata",
    "text_light_only": "text-light",
    "metadata_plus_text_light": "metadata + text-light",
    "full_api_reference": "full API reference",
}
MODEL_ORDER: Final[tuple[str, ...]] = (
    "ridge_regression",
    "random_forest_regressor",
    "hist_gradient_boosting_regressor",
)
MODEL_LABELS: Final[dict[str, str]] = {
    "ridge_regression": "Ridge",
    "random_forest_regressor": "Random Forest",
    "hist_gradient_boosting_regressor": "HistGBR",
}
STATEMENT_EXCLUDED_COLUMNS: Final[set[str]] = {
    "contest_id",
    "index",
    "name",
    "url",
    "statement_fetch_status",
    "statement_parse_status",
    "statement_error",
    "name_statement",
}
LEAKAGE_PATTERNS: Final[tuple[str, ...]] = (
    "solved",
    "solve",
    "accepted",
    "submission",
    "submissions",
    "participant",
    "participants",
    "successfulhack",
    "unsuccessfulhack",
)


class StatementColdStartError(RuntimeError):
    """Raised when statement cold-start experiments cannot proceed safely."""


class JsonLogFormatter(logging.Formatter):
    """Format statement cold-start logs as JSON Lines."""

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


@dataclass(frozen=True)
class ModelSpec:
    """Define one deterministic statement cold-start model."""

    model_name: str
    estimator_factory: Callable[[int], object]


def configure_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """Create the dedicated structured statement cold-start logger."""
    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.statement_cold_start")
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
        raise StatementColdStartError(f"{table_name} lacks columns: {missing}")


def _finite_float(value: object, digits: int = 6) -> float | None:
    """Convert finite values to JSON-safe rounded floats."""
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _dedupe_preserve_order(columns: Sequence[str]) -> list[str]:
    """Return unique columns while preserving first appearance."""
    seen: set[str] = set()
    output: list[str] = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            output.append(column)
    return output


def _key_series(series: pd.Series) -> pd.Series:
    """Normalize join keys to stable strings."""
    return series.map(
        lambda value: (
            ""
            if value is None or pd.isna(value)
            else str(int(value))
            if isinstance(value, (int, np.integer, float, np.floating))
            and float(value).is_integer()
            else str(value)
        )
    )


def load_feature_columns(path: Path = DEFAULT_FEATURE_COLUMNS_PATH) -> list[str]:
    """Read model feature columns from feature metadata when available."""
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StatementColdStartError("feature_columns metadata must be an object.")
    raw_columns = payload.get("feature_columns", [])
    if not isinstance(raw_columns, list) or not all(
        isinstance(column, str) for column in raw_columns
    ):
        raise StatementColdStartError("Invalid feature_columns metadata.")
    return list(raw_columns)


def has_solved_leakage(column: str) -> bool:
    """Return whether a column name suggests post-publication solved behavior."""
    lowered = re.sub(r"[^a-z0-9]+", "", column.lower())
    return any(pattern in lowered for pattern in LEAKAGE_PATTERNS)


def join_statement_features(
    model_table: pd.DataFrame,
    statement_features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Left-join statement features onto the model table by contest and index."""
    _require_columns(model_table, ("contest_id", "index"), "model table")
    _require_columns(statement_features, ("contest_id", "index"), "statement features")

    model = model_table.copy()
    statement = statement_features.copy()
    model["_contest_key"] = _key_series(model["contest_id"])
    model["_index_key"] = _key_series(model["index"])
    statement["_contest_key"] = _key_series(statement["contest_id"])
    statement["_index_key"] = _key_series(statement["index"])
    statement = statement.drop_duplicates(["_contest_key", "_index_key"], keep="first")

    statement_keys = statement.loc[:, ["_contest_key", "_index_key"]].copy()
    model_keys = model.loc[:, ["_contest_key", "_index_key"]].copy()
    matched_keys = model_keys.merge(
        statement_keys.drop_duplicates(),
        on=["_contest_key", "_index_key"],
        how="inner",
    )
    unmatched_model_rows = len(model) - len(matched_keys)
    unmatched_statement_rows = len(
        statement_keys.merge(
            model_keys.drop_duplicates(),
            on=["_contest_key", "_index_key"],
            how="left",
            indicator=True,
        ).loc[lambda frame: frame["_merge"].eq("left_only")]
    )

    joined = model.merge(
        statement.drop(columns=["contest_id", "index"]),
        on=["_contest_key", "_index_key"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_statement"),
    ).drop(columns=["_contest_key", "_index_key"])
    for column in TEXT_LIGHT_FEATURE_COLUMNS:
        if column in joined.columns:
            joined[column] = pd.to_numeric(joined[column], errors="coerce")
    counts = {
        "input_model_rows": int(len(model_table)),
        "statement_feature_rows": int(len(statement_features)),
        "matched_rows": int(len(matched_keys)),
        "unmatched_model_rows": int(unmatched_model_rows),
        "unmatched_statement_rows": int(unmatched_statement_rows),
    }
    return joined, counts


def select_metadata_features(
    frame: pd.DataFrame,
    model_feature_columns: Sequence[str],
) -> list[str]:
    """Select cold-start API metadata features without solved-count leakage."""
    if model_feature_columns:
        candidates = [column for column in model_feature_columns if column in frame.columns]
    else:
        excluded = {TARGET_COLUMN, *IDENTIFIER_COLUMNS}
        candidates = [column for column in frame.columns if column not in excluded]
    columns = [
        column
        for column in candidates
        if not has_solved_leakage(column)
        and not column.startswith("statement_")
        and column not in STATEMENT_EXCLUDED_COLUMNS
        and not column.endswith("_statement")
    ]
    if not columns:
        raise StatementColdStartError("No metadata-only features were selected.")
    return _dedupe_preserve_order(columns)


def select_text_light_features(frame: pd.DataFrame) -> list[str]:
    """Select numeric statement-derived text-light feature columns only."""
    columns: list[str] = []
    for column in TEXT_LIGHT_FEATURE_COLUMNS:
        if column not in frame.columns:
            continue
        if has_solved_leakage(column):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    if not columns:
        raise StatementColdStartError("No text-light statement features were selected.")
    return sorted(_dedupe_preserve_order(columns))


def select_full_api_reference_features(
    frame: pd.DataFrame,
    model_feature_columns: Sequence[str],
) -> list[str]:
    """Select original API features, allowing solved-count behavior."""
    if model_feature_columns:
        columns = [column for column in model_feature_columns if column in frame.columns]
    else:
        excluded = {TARGET_COLUMN, *IDENTIFIER_COLUMNS}
        columns = [
            column
            for column in frame.columns
            if column not in excluded
            and column not in STATEMENT_EXCLUDED_COLUMNS
            and not column.endswith("_statement")
            and not column.startswith("statement_")
            and column not in TEXT_LIGHT_FEATURE_COLUMNS
        ]
    if not columns:
        raise StatementColdStartError("No full API reference features were selected.")
    return _dedupe_preserve_order(columns)


def build_feature_sets(
    frame: pd.DataFrame,
    model_feature_columns: Sequence[str],
) -> dict[str, list[str]]:
    """Build all required experiment feature settings."""
    metadata = select_metadata_features(frame, model_feature_columns)
    text = select_text_light_features(frame)
    full_api = select_full_api_reference_features(frame, model_feature_columns)
    return {
        "metadata_only": metadata,
        "text_light_only": text,
        "metadata_plus_text_light": _dedupe_preserve_order([*metadata, *text]),
        "full_api_reference": full_api,
    }


def _ridge(seed: int) -> Ridge:
    """Build a deterministic ridge regressor."""
    del seed
    return Ridge(alpha=1.0)


def _random_forest(seed: int) -> RandomForestRegressor:
    """Build a deterministic random forest regressor."""
    return RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )


def _hist_gradient_boosting(seed: int) -> HistGradientBoostingRegressor:
    """Build a deterministic histogram gradient boosting regressor."""
    return HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=200,
        l2_regularization=0.01,
        random_state=seed,
    )


def build_model_specs() -> list[ModelSpec]:
    """Return required model specifications."""
    return [
        ModelSpec("ridge_regression", _ridge),
        ModelSpec("random_forest_regressor", _random_forest),
        ModelSpec("hist_gradient_boosting_regressor", _hist_gradient_boosting),
    ]


def join_split_assignments(
    frame: pd.DataFrame,
    split_assignment: pd.DataFrame,
) -> pd.DataFrame:
    """Join row-level split assignments to experiment rows."""
    _require_columns(frame, ("contest_id", "index", TARGET_COLUMN), "experiment table")
    _require_columns(split_assignment, ("contest_id", "index", "split_name"), "split")
    joined = frame.merge(
        split_assignment.loc[:, ["contest_id", "index", "split_name"]],
        on=["contest_id", "index"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(frame):
        raise StatementColdStartError(
            f"Split assignment matched {len(joined)} of {len(frame)} rows."
        )
    missing_splits = [
        split_name
        for split_name in SPLIT_NAMES
        if not joined["split_name"].eq(split_name).any()
    ]
    if missing_splits:
        raise StatementColdStartError(f"Split has empty partitions: {missing_splits}")
    return joined


def _fit_predict(
    joined: pd.DataFrame,
    feature_columns: Sequence[str],
    model_spec: ModelSpec,
    seed: int,
) -> pd.Series:
    """Fit one model on train rows and predict all rows."""
    train = joined.loc[joined["split_name"].eq("train")].copy()
    train_x = train.loc[:, list(feature_columns)]
    train_y = pd.to_numeric(train[TARGET_COLUMN], errors="raise").to_numpy(float)
    model = make_preprocessed_estimator(model_spec.estimator_factory(seed), train_x)
    model.fit(train_x, train_y)
    predictions = model.predict(joined.loc[:, list(feature_columns)])
    return pd.Series(predictions, index=joined.index, dtype=float)


def evaluate_statement_cold_start_strategy(
    experiment_table: pd.DataFrame,
    split_assignment: pd.DataFrame,
    *,
    strategy: str,
    feature_sets: Mapping[str, Sequence[str]],
    model_specs: Sequence[ModelSpec],
    seed: int,
) -> pd.DataFrame:
    """Evaluate every feature setting/model for one split strategy."""
    joined = join_split_assignments(experiment_table, split_assignment)
    rows: list[dict[str, object]] = []
    for feature_setting in FEATURE_SETTING_ORDER:
        feature_columns = list(feature_sets[feature_setting])
        for model_spec in model_specs:
            predictions = _fit_predict(joined, feature_columns, model_spec, seed)
            for split_name in SPLIT_NAMES:
                mask = joined["split_name"].eq(split_name)
                metrics = compute_regression_metrics(
                    joined.loc[mask, TARGET_COLUMN],
                    predictions.loc[mask],
                )
                rows.append(
                    {
                        "strategy": strategy,
                        "split_name": split_name,
                        "feature_setting": feature_setting,
                        "model_name": model_spec.model_name,
                        "row_count": int(mask.sum()),
                        "feature_count": len(feature_columns),
                        "is_cold_start": feature_setting != "full_api_reference",
                        **metrics,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["strategy", "feature_setting", "model_name", "split_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_best_by_setting(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select each setting's model on validation and report test metrics."""
    ranked = build_validation_ranked_report(
        metrics,
        group_columns=("strategy", "feature_setting"),
        candidate_columns=("model_name",),
        metric_columns=DEFAULT_METRIC_COLUMNS,
    )
    return select_rank_one(ranked).sort_values(
        ["strategy", "feature_setting"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_pairwise_comparisons(best_by_setting: pd.DataFrame) -> dict[str, object]:
    """Build summary comparisons among feature settings."""
    improvements: list[dict[str, object]] = []
    full_api_comparisons: list[dict[str, object]] = []
    for strategy, group in best_by_setting.groupby("strategy", sort=True):
        by_setting = group.set_index("feature_setting")
        if {"metadata_only", "metadata_plus_text_light"}.issubset(by_setting.index):
            metadata_mae = float(by_setting.loc["metadata_only", "MAE"])
            combined_mae = float(by_setting.loc["metadata_plus_text_light", "MAE"])
            delta = metadata_mae - combined_mae
            improvements.append(
                {
                    "strategy": strategy,
                    "metadata_only_MAE": round(metadata_mae, 6),
                    "metadata_plus_text_light_MAE": round(combined_mae, 6),
                    "absolute_MAE_improvement": round(delta, 6),
                    "percent_MAE_improvement": round(
                        (delta / metadata_mae * 100.0) if metadata_mae else 0.0,
                        6,
                    ),
                }
            )
        if {"full_api_reference", "metadata_plus_text_light"}.issubset(
            by_setting.index
        ):
            full_mae = float(by_setting.loc["full_api_reference", "MAE"])
            combined_mae = float(by_setting.loc["metadata_plus_text_light", "MAE"])
            full_api_comparisons.append(
                {
                    "strategy": strategy,
                    "metadata_plus_text_light_MAE": round(combined_mae, 6),
                    "full_api_reference_MAE": round(full_mae, 6),
                    "MAE_gap_vs_full_api_reference": round(combined_mae - full_mae, 6),
                    "note": (
                        "full_api_reference uses post-publication solved behavior "
                        "and is not a cold-start setting"
                    ),
                }
            )
    return {
        "metadata_plus_text_light_over_metadata_only": improvements,
        "metadata_plus_text_light_vs_full_api_reference": full_api_comparisons,
    }


def _statement_counts(experiment_table: pd.DataFrame) -> dict[str, int | float]:
    """Summarize statement matching and parse availability."""
    parse_status = (
        experiment_table["statement_parse_status"]
        if "statement_parse_status" in experiment_table.columns
        else pd.Series(dtype=object)
    )
    available = (
        pd.to_numeric(experiment_table["statement_available"], errors="coerce")
        if "statement_available" in experiment_table.columns
        else pd.Series(dtype=float)
    )
    parsed_success = int(parse_status.eq("parsed").sum())
    missing = int(len(experiment_table) - parsed_success)
    available_count = int(available.fillna(0).sum())
    return {
        "statement_parsed_success_count": parsed_success,
        "statement_missing_count": missing,
        "statement_available_count": available_count,
        "statement_feature_coverage_rate": round(
            parsed_success / len(experiment_table), 6
        )
        if len(experiment_table)
        else 0.0,
    }


def build_summary(
    *,
    join_counts: Mapping[str, int],
    experiment_table: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    best_by_setting: pd.DataFrame,
) -> dict[str, object]:
    """Build machine-readable statement cold-start summary."""
    comparisons = build_pairwise_comparisons(best_by_setting)
    best_records = (
        best_by_setting.sort_values(
            ["strategy", "feature_setting"],
            kind="mergesort",
        ).to_dict(orient="records")
        if not best_by_setting.empty
        else []
    )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_model_table_rows": int(join_counts["input_model_rows"]),
        "statement_feature_rows": int(join_counts["statement_feature_rows"]),
        "matched_rows": int(join_counts["matched_rows"]),
        "unmatched_model_rows": int(join_counts["unmatched_model_rows"]),
        "unmatched_statement_rows": int(join_counts["unmatched_statement_rows"]),
        **_statement_counts(experiment_table),
        "feature_counts": {
            setting: int(len(columns)) for setting, columns in feature_sets.items()
        },
        "validation_selected_model_test_report": best_records,
        "improvement_of_metadata_plus_text_light_over_metadata_only": comparisons[
            "metadata_plus_text_light_over_metadata_only"
        ],
        "comparison_against_full_api_reference": comparisons[
            "metadata_plus_text_light_vs_full_api_reference"
        ],
        "conservative_notes": [
            "Statement text-light features are approximate HTML-derived features.",
            "Statement features are intended for cold-start analysis.",
            (
                "Full API reference uses post-publication solved behavior and is "
                "not a cold-start setting."
            ),
            (
                "Improved cold-start performance, if observed, does not prove "
                "semantic understanding of problem difficulty."
            ),
            (
                "These experiments do not use deep NLP or problem-statement "
                "embeddings."
            ),
            (
                "Missing-statement rows are handled by imputation rather than "
                "being treated as successful parsed statements."
            ),
            (
                "Model choice is made only on validation MAE; test metrics are "
                "reported after that choice is fixed."
            ),
        ],
    }


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Persist a matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "cf-diff-statement-cold-start"},
    )
    plt.close(fig)


def plot_mae_comparison(test_metrics: pd.DataFrame, path: Path) -> None:
    """Plot test MAE by feature setting, model, and strategy."""
    if test_metrics.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_axis_off()
        _save_figure(fig, path)
        return
    strategies = [s for s in STRATEGIES if test_metrics["strategy"].eq(s).any()]
    fig, axes = plt.subplots(
        1,
        len(strategies),
        figsize=(18, 6.5),
        sharey=True,
        squeeze=False,
    )
    x = np.arange(len(FEATURE_SETTING_ORDER))
    width = min(0.24, 0.8 / len(MODEL_ORDER))
    for axis_index, strategy in enumerate(strategies):
        ax = axes[0, axis_index]
        strategy_frame = test_metrics.loc[test_metrics["strategy"].eq(strategy)]
        for model_index, model_name in enumerate(MODEL_ORDER):
            values = (
                strategy_frame.loc[strategy_frame["model_name"].eq(model_name)]
                .set_index("feature_setting")
                .reindex(FEATURE_SETTING_ORDER)["MAE"]
            )
            offset = (model_index - (len(MODEL_ORDER) - 1) / 2) * width
            ax.bar(
                x + offset,
                pd.to_numeric(values, errors="coerce"),
                width=width,
                label=MODEL_LABELS.get(model_name, model_name),
            )
        ax.set_title(strategy)
        ax.set_xlabel("Feature setting")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [FEATURE_SETTING_LABELS[setting] for setting in FEATURE_SETTING_ORDER],
            rotation=25,
            ha="right",
        )
        ax.grid(axis="y", alpha=0.25)
        if axis_index == 0:
            ax.set_ylabel("Test MAE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Statement text-light cold-start comparison", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    _save_figure(fig, path)


def plot_statement_feature_coverage(experiment_table: pd.DataFrame, path: Path) -> None:
    """Plot statement parse/availability coverage counts."""
    status = (
        experiment_table["statement_parse_status"].fillna("unmatched")
        if "statement_parse_status" in experiment_table.columns
        else pd.Series(["unmatched"] * len(experiment_table))
    )
    counts = status.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Statement feature coverage")
    ax.set_xlabel("Statement parse status")
    ax.set_ylabel("Problem count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, path)


def run_statement_cold_start(
    *,
    config_path: Path,
    feature_path: Path,
    feature_columns_path: Path,
    statement_feature_path: Path,
    contest_split_path: Path,
    time_split_path: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Path]:
    """Run statement text-light cold-start experiments."""
    logger = configure_logger(log_path)
    try:
        config = load_baseline_config(config_path)
        model_table = pd.read_parquet(feature_path, engine="pyarrow")
        statement_features = pd.read_parquet(statement_feature_path, engine="pyarrow")
        model_feature_columns = load_feature_columns(feature_columns_path)
        experiment_table, join_counts = join_statement_features(
            model_table,
            statement_features,
        )
        feature_sets = build_feature_sets(experiment_table, model_feature_columns)
        for setting, columns in feature_sets.items():
            logger.info(
                "Selected statement cold-start feature columns",
                extra={
                    "event": "feature_columns_selected",
                    "details": {
                        "feature_setting": setting,
                        "feature_count": len(columns),
                        "feature_columns": list(columns),
                    },
                },
            )

        model_specs = build_model_specs()
        contest_split = pd.read_parquet(contest_split_path, engine="pyarrow")
        time_split = pd.read_parquet(time_split_path, engine="pyarrow")
        metrics = pd.concat(
            [
                evaluate_statement_cold_start_strategy(
                    experiment_table,
                    contest_split,
                    strategy="contest_grouped",
                    feature_sets=feature_sets,
                    model_specs=model_specs,
                    seed=config.random_seed,
                ),
                evaluate_statement_cold_start_strategy(
                    experiment_table,
                    time_split,
                    strategy="forward_time",
                    feature_sets=feature_sets,
                    model_specs=model_specs,
                    seed=config.random_seed,
                ),
            ],
            ignore_index=True,
        ).sort_values(
            ["strategy", "feature_setting", "model_name", "split_name"],
            kind="mergesort",
        )
        test_metrics = metrics.loc[metrics["split_name"].eq("test")].copy()
        best_by_setting = build_best_by_setting(metrics)
        summary = build_summary(
            join_counts=join_counts,
            experiment_table=experiment_table,
            feature_sets=feature_sets,
            best_by_setting=best_by_setting,
        )

        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "results": output_dir / "statement_cold_start_results.csv",
            "summary": output_dir / "statement_cold_start_summary.json",
            "best_by_setting": output_dir
            / "statement_cold_start_best_by_setting.csv",
            "mae_comparison": output_dir / "statement_cold_start_mae_comparison.png",
            "statement_feature_coverage": output_dir
            / "statement_feature_coverage.png",
        }
        metrics.to_csv(paths["results"], index=False)
        best_by_setting.to_csv(paths["best_by_setting"], index=False)
        write_json(paths["summary"], summary)
        plot_mae_comparison(test_metrics, paths["mae_comparison"])
        plot_statement_feature_coverage(
            experiment_table,
            paths["statement_feature_coverage"],
        )
        logger.info(
            "Completed statement text-light cold-start experiments",
            extra={
                "event": "statement_cold_start_completed",
                "details": {
                    "output_dir": output_dir.as_posix(),
                    "rows": len(experiment_table),
                    "random_seed": config.random_seed,
                },
            },
        )
        return paths
    except Exception:
        logger.exception(
            "Statement text-light cold-start experiments failed",
            extra={"event": "statement_cold_start_failed"},
        )
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the statement cold-start CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run statement text-light cold-start experiments."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument(
        "--feature-columns-path",
        type=Path,
        default=DEFAULT_FEATURE_COLUMNS_PATH,
    )
    parser.add_argument(
        "--statement-feature-path",
        type=Path,
        default=DEFAULT_STATEMENT_FEATURE_PATH,
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the statement cold-start CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        paths = run_statement_cold_start(
            config_path=args.config,
            feature_path=args.feature_path,
            feature_columns_path=args.feature_columns_path,
            statement_feature_path=args.statement_feature_path,
            contest_split_path=args.contest_split_path,
            time_split_path=args.time_split_path,
            output_dir=args.output_dir,
            log_path=args.log_path,
        )
    except (StatementColdStartError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote statement cold-start results: {paths['results']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
