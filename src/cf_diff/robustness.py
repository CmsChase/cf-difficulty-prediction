"""Run robustness experiments for Codeforces rating prediction."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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

DEFAULT_CONFIG_PATH: Final[Path] = Path("configs/experiment.yaml")
DEFAULT_PROCESSED_PATH: Final[Path] = Path(
    "data/processed/rated_programming_problems.parquet"
)
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
DEFAULT_RAW_MANIFEST_PATH: Final[Path] = Path(
    "data/raw/codeforces/latest/manifest.json"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path("outputs/robustness")
DEFAULT_LOG_PATH: Final[Path] = Path("outputs/logs/robustness.log")
TARGET_COLUMN: Final[str] = "rating"
STRATEGIES: Final[tuple[str, str]] = ("contest_grouped", "forward_time")
SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "valid", "test")
RAW_SOLVED_COLUMNS: Final[tuple[str, ...]] = (
    "solved_count",
    "log_solved_count",
    "solved_count_missing",
)
AGE_NORMALIZED_COLUMNS: Final[tuple[str, ...]] = (
    "problem_age_days",
    "solves_per_day",
    "log_solves_per_day",
)
METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "MAE",
    "RMSE",
    "R2",
    "within_100",
    "within_200",
)
COLD_START_FEATURE_SETS: Final[tuple[str, ...]] = (
    "metadata_only_cold_start",
    "index_tags_only",
    "index_points_only",
    "tags_points_only",
    "full_api_reference",
)
AGE_NORMALIZED_FEATURE_SETS: Final[tuple[str, ...]] = (
    "age_normalized_solved_only",
    "raw_solved_only_reference",
    "index_tags_points_age_norm",
    "full_api_plus_age_norm",
    "full_api_without_raw_solved_but_with_age_norm",
)


class RobustnessError(RuntimeError):
    """Raised when robustness experiments cannot proceed safely."""


class JsonLogFormatter(logging.Formatter):
    """Format robustness logs as JSON Lines."""

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
class RobustnessModelSpec:
    """Define one deterministic robustness model."""

    model_name: str
    estimator_factory: Callable[[int], object]


def configure_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """Create the dedicated structured robustness logger."""
    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.robustness")
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
        raise RobustnessError(f"{table_name} lacks required columns: {missing}")


def _finite_float(value: object, digits: int = 6) -> float | None:
    """Convert numeric output to a rounded JSON-safe float."""
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _dedupe_preserve_order(columns: Sequence[str]) -> list[str]:
    """Return unique columns while preserving first appearance."""
    seen: set[str] = set()
    result: list[str] = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            result.append(column)
    return result


def _available_columns(
    frame: pd.DataFrame,
    requested_columns: Sequence[str],
) -> list[str]:
    """Return requested columns that exist in the frame."""
    return [column for column in requested_columns if column in frame.columns]


def load_feature_columns(path: Path) -> list[str]:
    """Read feature columns from feature metadata, if available."""
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RobustnessError("Feature-column metadata must be a JSON object.")
    raw_columns = payload.get("feature_columns", [])
    if not isinstance(raw_columns, list) or not all(
        isinstance(column, str) for column in raw_columns
    ):
        raise RobustnessError("feature_columns.json has invalid feature_columns.")
    return list(raw_columns)


def _parse_utc_datetime(value: str) -> pd.Timestamp:
    """Parse an ISO-8601 timestamp as a timezone-aware UTC timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    return timestamp.tz_convert(timezone.utc)


def load_snapshot_time(
    manifest_path: Path,
    fallback_frame: pd.DataFrame,
) -> pd.Timestamp:
    """Load snapshot time from raw manifest, falling back to max contest time."""
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_created_at = payload.get("created_at_utc")
            if isinstance(raw_created_at, str) and raw_created_at.strip():
                return _parse_utc_datetime(raw_created_at)

    if "start_time_seconds" not in fallback_frame.columns:
        raise RobustnessError(
            "Cannot infer snapshot time without manifest created_at_utc "
            "or start_time_seconds."
        )
    start_times = pd.to_numeric(
        fallback_frame["start_time_seconds"],
        errors="coerce",
    ).dropna()
    if start_times.empty:
        raise RobustnessError("Cannot infer snapshot time from empty start times.")
    return pd.Timestamp.fromtimestamp(float(start_times.max()), tz=timezone.utc)


def _source_series(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a column from primary, or merge it from secondary if needed."""
    if column in primary.columns:
        return primary[column]
    if column not in secondary.columns:
        raise RobustnessError(f"Neither table contains required column {column!r}.")
    _require_columns(primary, ("contest_id", "index"), "feature table")
    _require_columns(secondary, ("contest_id", "index", column), "processed table")
    merged = primary.loc[:, ["contest_id", "index"]].merge(
        secondary.loc[:, ["contest_id", "index", column]],
        on=["contest_id", "index"],
        how="left",
        validate="one_to_one",
    )
    return merged[column]


def add_age_normalized_features(
    model_table: pd.DataFrame,
    processed_table: pd.DataFrame,
    snapshot_time: pd.Timestamp,
) -> pd.DataFrame:
    """Return a copy of the model table with age-normalized solve features."""
    frame = model_table.copy()
    snapshot_utc = (
        snapshot_time.tz_localize(timezone.utc)
        if snapshot_time.tzinfo is None
        else snapshot_time.tz_convert(timezone.utc)
    )
    start_seconds = pd.to_numeric(
        _source_series(frame, processed_table, "start_time_seconds"),
        errors="coerce",
    )
    solved_count = pd.to_numeric(
        _source_series(frame, processed_table, "solved_count"),
        errors="coerce",
    ).fillna(0.0)
    solved_count = solved_count.clip(lower=0.0)

    age_days = (snapshot_utc.timestamp() - start_seconds) / 86400.0
    age_days = age_days.where(age_days.notna(), 1.0).clip(lower=1.0)
    solves_per_day = solved_count / age_days

    frame["problem_age_days"] = age_days.astype(float)
    frame["solves_per_day"] = solves_per_day.astype(float)
    frame["log_solves_per_day"] = np.log1p(solves_per_day).astype(float)
    return frame


def build_robustness_feature_groups(
    model_table: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Select available columns for robustness feature groups."""
    feature_columns = list(feature_columns or [])
    index = _available_columns(
        model_table,
        ("index_letter", "index_number", "index_rank"),
    )
    points = _available_columns(model_table, ("has_points", "points"))
    tags = sorted(
        column
        for column in model_table.columns
        if column == "tag_count" or column.startswith("tag__")
    )
    solved = _available_columns(model_table, RAW_SOLVED_COLUMNS)
    age_norm = _available_columns(model_table, AGE_NORMALIZED_COLUMNS)
    if feature_columns:
        full_api = _available_columns(model_table, feature_columns)
    else:
        full_api = _dedupe_preserve_order([*index, *tags, *points, *solved])
    full_api = [
        column
        for column in full_api
        if column not in AGE_NORMALIZED_COLUMNS and column in model_table.columns
    ]
    groups = {
        "index": sorted(index),
        "tags": sorted(tags),
        "points": sorted(points),
        "solved": sorted(solved),
        "age_normalized": sorted(age_norm),
        "full_api": _dedupe_preserve_order(full_api),
    }
    if not groups["index"]:
        raise RobustnessError("No index feature columns are available.")
    if not groups["full_api"]:
        raise RobustnessError("No full API feature columns are available.")
    if not groups["age_normalized"]:
        raise RobustnessError("Age-normalized feature columns were not created.")
    return groups


def _combine_groups(
    feature_groups: Mapping[str, Sequence[str]],
    group_names: Sequence[str],
) -> list[str]:
    """Return all columns from a list of feature groups."""
    columns: list[str] = []
    for group_name in group_names:
        columns.extend(feature_groups.get(group_name, []))
    return _dedupe_preserve_order(columns)


def build_robustness_feature_sets(
    feature_groups: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, object]]:
    """Build named feature-set definitions for robustness experiments."""
    definitions: dict[str, tuple[str, tuple[str, ...]]] = {
        "metadata_only_cold_start": (
            "cold_start",
            ("index", "tags", "points"),
        ),
        "index_tags_only": ("cold_start", ("index", "tags")),
        "index_points_only": ("cold_start", ("index", "points")),
        "tags_points_only": ("cold_start", ("tags", "points")),
        "full_api_reference": ("full_api_reference", ("full_api",)),
        "age_normalized_solved_only": (
            "age_normalized",
            ("age_normalized",),
        ),
        "raw_solved_only_reference": ("age_normalized_reference", ("solved",)),
        "index_tags_points_age_norm": (
            "age_normalized",
            ("index", "tags", "points", "age_normalized"),
        ),
        "full_api_plus_age_norm": (
            "age_normalized",
            ("full_api", "age_normalized"),
        ),
        "full_api_without_raw_solved_but_with_age_norm": (
            "age_normalized",
            ("index", "tags", "points", "age_normalized"),
        ),
    }
    feature_sets: dict[str, dict[str, object]] = {}
    for name, (family, groups) in definitions.items():
        columns = _combine_groups(feature_groups, groups)
        should_exclude_raw_solved = name in {
            "metadata_only_cold_start",
            "index_tags_only",
            "index_points_only",
            "tags_points_only",
            "age_normalized_solved_only",
            "index_tags_points_age_norm",
            "full_api_without_raw_solved_but_with_age_norm",
        }
        if should_exclude_raw_solved:
            columns = [
                column
                for column in columns
                if column not in RAW_SOLVED_COLUMNS
            ]
        if not columns:
            raise RobustnessError(f"Feature set {name!r} has no columns.")
        feature_sets[name] = {
            "experiment_family": family,
            "included_groups": list(groups),
            "feature_columns": columns,
            "feature_count": len(columns),
        }
    return feature_sets


def _ridge_estimator(seed: int) -> Ridge:
    """Build a deterministic ridge regressor."""
    del seed
    return Ridge(alpha=1.0)


def _hist_gradient_boosting_estimator(
    seed: int,
) -> HistGradientBoostingRegressor:
    """Build a deterministic histogram gradient boosting regressor."""
    return HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=160,
        l2_regularization=0.01,
        random_state=seed,
    )


def build_model_specs() -> list[RobustnessModelSpec]:
    """Return deterministic robustness models."""
    return [
        RobustnessModelSpec("ridge_regression", _ridge_estimator),
        RobustnessModelSpec(
            "hist_gradient_boosting_regressor",
            _hist_gradient_boosting_estimator,
        ),
    ]


def join_split_assignments(
    model_table: pd.DataFrame,
    split_assignment: pd.DataFrame,
) -> pd.DataFrame:
    """Join model rows with row-level split assignments."""
    _require_columns(
        model_table,
        ("contest_id", "index", TARGET_COLUMN),
        "model table",
    )
    _require_columns(
        split_assignment,
        ("contest_id", "index", "split_name"),
        "split assignment",
    )
    joined = model_table.merge(
        split_assignment.loc[:, ["contest_id", "index", "split_name"]],
        on=["contest_id", "index"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(model_table):
        raise RobustnessError(
            "Split assignment did not match every model-table row: "
            f"{len(joined)} of {len(model_table)} rows matched."
        )
    missing_splits = [
        split_name
        for split_name in SPLIT_NAMES
        if not joined["split_name"].eq(split_name).any()
    ]
    if missing_splits:
        raise RobustnessError(f"Split assignment has empty splits: {missing_splits}")
    return joined


def _fit_predict(
    joined: pd.DataFrame,
    feature_columns: Sequence[str],
    model_spec: RobustnessModelSpec,
    seed: int,
) -> pd.Series:
    """Fit one robustness model on train rows and predict all rows."""
    train = joined.loc[joined["split_name"].eq("train")].copy()
    train_x = train.loc[:, list(feature_columns)]
    train_y = pd.to_numeric(train[TARGET_COLUMN], errors="raise").to_numpy(
        dtype=float
    )
    model = make_preprocessed_estimator(model_spec.estimator_factory(seed), train_x)
    model.fit(train_x, train_y)
    predictions = model.predict(joined.loc[:, list(feature_columns)])
    return pd.Series(predictions, index=joined.index, dtype=float)


def evaluate_robustness_strategy(
    model_table: pd.DataFrame,
    split_assignment: pd.DataFrame,
    *,
    strategy: str,
    feature_sets: Mapping[str, Mapping[str, object]],
    model_specs: Sequence[RobustnessModelSpec],
    seed: int,
) -> pd.DataFrame:
    """Evaluate every model and feature set for one split strategy."""
    joined = join_split_assignments(model_table, split_assignment)
    rows: list[dict[str, object]] = []
    for model_spec in model_specs:
        for feature_set_name, definition in feature_sets.items():
            feature_columns = list(definition["feature_columns"])
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
                        "model_name": model_spec.model_name,
                        "feature_set_name": feature_set_name,
                        "experiment_family": definition["experiment_family"],
                        "split_name": split_name,
                        **metrics,
                        "feature_count": len(feature_columns),
                        "row_count": int(mask.sum()),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["strategy", "model_name", "feature_set_name", "split_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_cold_start_comparison(test_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare cold-start feature sets against the full API reference."""
    rows: list[dict[str, object]] = []
    for (strategy, model_name), group in test_metrics.groupby(
        ["strategy", "model_name"],
        sort=True,
    ):
        reference = group.loc[group["feature_set_name"].eq("full_api_reference")]
        if reference.empty:
            continue
        reference_row = reference.iloc[0]
        reference_mae = float(reference_row["MAE"])
        for feature_set in COLD_START_FEATURE_SETS:
            candidate = group.loc[group["feature_set_name"].eq(feature_set)]
            if candidate.empty:
                continue
            row = candidate.iloc[0]
            mae = float(row["MAE"])
            gap = mae - reference_mae
            pct_gap = (gap / reference_mae * 100.0) if reference_mae else np.nan
            rows.append(
                {
                    "strategy": strategy,
                    "model_name": model_name,
                    "feature_set_name": feature_set,
                    "reference_feature_set": "full_api_reference",
                    "MAE": round(mae, 6),
                    "RMSE": _finite_float(row["RMSE"]),
                    "R2": _finite_float(row["R2"]),
                    "within_100": _finite_float(row["within_100"]),
                    "within_200": _finite_float(row["within_200"]),
                    "full_api_reference_MAE": round(reference_mae, 6),
                    "absolute_MAE_gap_vs_full_api": round(float(gap), 6),
                    "percent_MAE_gap_vs_full_api": round(float(pct_gap), 6),
                    "feature_count": int(row["feature_count"]),
                    "row_count": int(row["row_count"]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["strategy", "model_name", "feature_set_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_locked_robustness_report(metrics: pd.DataFrame) -> pd.DataFrame:
    """Lock one algorithm on reference validation MAE for every feature set."""
    reference = metrics.loc[
        metrics["feature_set_name"].eq("full_api_reference")
    ]
    reference_ranking = build_validation_ranked_report(
        reference,
        group_columns=("strategy",),
        candidate_columns=("model_name",),
        metric_columns=DEFAULT_METRIC_COLUMNS,
    )
    selected_models = select_rank_one(reference_ranking).loc[
        :, ["strategy", "model_name"]
    ]
    locked = metrics.merge(
        selected_models,
        on=["strategy", "model_name"],
        how="inner",
        validate="many_to_one",
    )
    report = build_validation_ranked_report(
        locked,
        group_columns=("strategy", "feature_set_name"),
        candidate_columns=("model_name",),
        metric_columns=DEFAULT_METRIC_COLUMNS,
    )
    return select_rank_one(report).sort_values(
        ["strategy", "feature_set_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_age_normalized_comparison(test_metrics: pd.DataFrame) -> pd.DataFrame:
    """Create a test-metric table for age-normalized experiments."""
    wanted = set(AGE_NORMALIZED_FEATURE_SETS)
    columns = [
        "strategy",
        "model_name",
        "feature_set_name",
        *METRIC_COLUMNS,
        "feature_count",
        "row_count",
    ]
    subset = test_metrics.loc[test_metrics["feature_set_name"].isin(wanted), columns]
    return subset.sort_values(
        ["strategy", "model_name", "feature_set_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_raw_vs_age_solved_comparison(test_metrics: pd.DataFrame) -> list[dict[str, object]]:
    """Compare raw solved-only features with age-normalized solved features."""
    rows: list[dict[str, object]] = []
    for (strategy, model_name), group in test_metrics.groupby(
        ["strategy", "model_name"],
        sort=True,
    ):
        raw = group.loc[group["feature_set_name"].eq("raw_solved_only_reference")]
        age = group.loc[group["feature_set_name"].eq("age_normalized_solved_only")]
        if raw.empty or age.empty:
            continue
        raw_mae = float(raw.iloc[0]["MAE"])
        age_mae = float(age.iloc[0]["MAE"])
        rows.append(
            {
                "strategy": strategy,
                "model_name": model_name,
                "raw_solved_only_MAE": round(raw_mae, 6),
                "age_normalized_solved_only_MAE": round(age_mae, 6),
                "age_minus_raw_MAE": round(age_mae - raw_mae, 6),
                "note": (
                    "Positive values mean age-normalized solved-only features "
                    "had higher MAE than raw solved-only features."
                ),
            }
        )
    return rows


def build_age_feature_summary(model_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize age-normalized solve feature distributions."""
    rows: list[dict[str, object]] = []
    for column in AGE_NORMALIZED_COLUMNS:
        if column not in model_table.columns:
            continue
        series = pd.to_numeric(model_table[column], errors="coerce").dropna()
        if series.empty:
            stats = {
                "feature": column,
                "count": 0,
                "mean": None,
                "median": None,
                "p75": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "max": None,
            }
        else:
            stats = {
                "feature": column,
                "count": int(series.size),
                "mean": _finite_float(series.mean()),
                "median": _finite_float(series.median()),
                "p75": _finite_float(series.quantile(0.75)),
                "p90": _finite_float(series.quantile(0.90)),
                "p95": _finite_float(series.quantile(0.95)),
                "p99": _finite_float(series.quantile(0.99)),
                "max": _finite_float(series.max()),
            }
        rows.append(stats)
    return pd.DataFrame(rows).sort_values("feature", kind="mergesort")


def _best_rows(
    test_metrics: pd.DataFrame,
    feature_sets: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Select a feature set on validation and attach its test report."""
    result: dict[str, dict[str, object]] = {}
    subset = test_metrics.loc[test_metrics["feature_set_name"].isin(feature_sets)]
    for strategy, group in subset.groupby("strategy", sort=True):
        selection_column = (
            "validation_MAE" if "validation_MAE" in group.columns else "MAE"
        )
        row = group.sort_values(
            [selection_column, "model_name", "feature_set_name"],
            kind="mergesort",
        ).iloc[0]
        result[strategy] = {
            "model_name": row["model_name"],
            "feature_set_name": row["feature_set_name"],
            "test_MAE": _finite_float(row["MAE"]),
            "validation_MAE": _finite_float(row.get("validation_MAE")),
            "RMSE": _finite_float(row["RMSE"]),
            "R2": _finite_float(row["R2"]),
            "within_200": _finite_float(row["within_200"]),
            "feature_count": int(row["feature_count"]),
            "row_count": int(row["row_count"]),
        }
    return result


def build_robustness_summary(
    test_metrics: pd.DataFrame,
    cold_start_comparison: pd.DataFrame,
    raw_vs_age_solved: Sequence[Mapping[str, object]],
    snapshot_time: pd.Timestamp,
) -> dict[str, object]:
    """Build machine-readable robustness summary."""
    cold_feature_sets = [
        feature_set
        for feature_set in COLD_START_FEATURE_SETS
        if feature_set != "full_api_reference"
    ]
    age_feature_sets = [
        feature_set
        for feature_set in AGE_NORMALIZED_FEATURE_SETS
        if feature_set != "raw_solved_only_reference"
    ]
    metadata_comparison = cold_start_comparison.loc[
        cold_start_comparison["feature_set_name"].eq("metadata_only_cold_start")
    ]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_time_utc": snapshot_time.isoformat(),
        "experiment_families": {
            "post_publication_full_api_reference": (
                "Uses current snapshot solve counts and all current API-derived "
                "features as a reference, not as a cold-start setting."
            ),
            "cold_start": (
                "Excludes solved-count features to approximate prediction before "
                "submissions accumulate."
            ),
            "age_normalized": (
                "Adds solves-per-day features computed inside this module to "
                "partially adjust for unequal exposure time."
            ),
        },
        "validation_selected_cold_start_test_report": _best_rows(
            test_metrics,
            cold_feature_sets,
        ),
        "validation_selected_age_normalized_test_report": _best_rows(
            test_metrics,
            age_feature_sets,
        ),
        "metadata_only_cold_start_vs_full_api_reference": (
            metadata_comparison.to_dict(orient="records")
        ),
        "raw_solved_features_vs_age_normalized_solved_features": list(
            raw_vs_age_solved
        ),
        "conservative_notes": [
            (
                "Age normalization divides observed solves by elapsed days and "
                "therefore only partially adjusts exposure bias."
            ),
            (
                "The robustness experiments reuse existing splits and do not "
                "retrain or modify baseline artefacts."
            ),
            (
                "Full API reference results use solved statistics observed at "
                "snapshot time and should not be described as cold-start "
                "performance."
            ),
            (
                "The algorithm is selected on full-API validation MAE and then "
                "locked across feature-set comparisons on test data."
            ),
        ],
    }


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Persist a matplotlib figure with stable output settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "cf-diff-robustness"},
    )
    plt.close(fig)


def _empty_figure(title: str) -> plt.Figure:
    """Create a readable placeholder figure for empty inputs."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title(title)
    ax.set_axis_off()
    return fig


def plot_metric_comparison(
    test_metrics: pd.DataFrame,
    *,
    feature_sets: Sequence[str],
    metric: str,
    title: str,
    path: Path,
) -> None:
    """Save a deterministic bar chart for selected test metrics."""
    subset = test_metrics.loc[test_metrics["feature_set_name"].isin(feature_sets)]
    if subset.empty:
        _save_figure(_empty_figure(title), path)
        return
    subset = subset.sort_values(
        ["strategy", "model_name", "feature_set_name"],
        kind="mergesort",
    )
    labels = (
        subset["strategy"]
        + "\n"
        + subset["model_name"]
        + "\n"
        + subset["feature_set_name"]
    ).tolist()
    fig, ax = plt.subplots(figsize=(max(10, len(subset) * 0.55), 6))
    ax.bar(np.arange(len(subset)), pd.to_numeric(subset[metric], errors="coerce"))
    ax.set_title(title)
    ax.set_xlabel("Strategy / model / feature set")
    ax.set_ylabel(metric)
    ax.set_xticks(np.arange(len(subset)))
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, path)


def plot_age_feature_distributions(age_summary_frame: pd.DataFrame, path: Path) -> None:
    """Save histograms for age-normalized solve features."""
    title = "Age-normalized solve feature distributions"
    if age_summary_frame.empty:
        _save_figure(_empty_figure(title), path)
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, column in zip(axes, AGE_NORMALIZED_COLUMNS, strict=False):
        if column not in age_summary_frame.columns:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center")
            ax.set_title(column)
            continue
        series = pd.to_numeric(age_summary_frame[column], errors="coerce").dropna()
        if series.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        else:
            upper = series.quantile(0.99)
            clipped = series.clip(upper=upper)
            ax.hist(clipped, bins=30, color="#4C78A8", edgecolor="white")
        ax.set_title(column)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    _save_figure(fig, path)


def run_robustness(
    *,
    config_path: Path,
    processed_path: Path,
    feature_path: Path,
    feature_columns_path: Path,
    contest_split_path: Path,
    time_split_path: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Path]:
    """Run robustness experiments and write tables, figures, and summary."""
    logger = configure_logger(log_path)
    try:
        config = load_baseline_config(config_path)
        processed_table = pd.read_parquet(processed_path, engine="pyarrow")
        model_table = pd.read_parquet(feature_path, engine="pyarrow")
        feature_columns = load_feature_columns(feature_columns_path)
        snapshot_time = load_snapshot_time(DEFAULT_RAW_MANIFEST_PATH, processed_table)
        model_table = add_age_normalized_features(
            model_table,
            processed_table,
            snapshot_time,
        )
        feature_groups = build_robustness_feature_groups(
            model_table,
            feature_columns,
        )
        feature_sets = build_robustness_feature_sets(feature_groups)
        model_specs = build_model_specs()
        contest_split = pd.read_parquet(contest_split_path, engine="pyarrow")
        time_split = pd.read_parquet(time_split_path, engine="pyarrow")

        metrics = pd.concat(
            [
                evaluate_robustness_strategy(
                    model_table,
                    contest_split,
                    strategy="contest_grouped",
                    feature_sets=feature_sets,
                    model_specs=model_specs,
                    seed=config.random_seed,
                ),
                evaluate_robustness_strategy(
                    model_table,
                    time_split,
                    strategy="forward_time",
                    feature_sets=feature_sets,
                    model_specs=model_specs,
                    seed=config.random_seed,
                ),
            ],
            ignore_index=True,
        ).sort_values(
            ["strategy", "model_name", "feature_set_name", "split_name"],
            kind="mergesort",
        )
        test_metrics = metrics.loc[metrics["split_name"].eq("test")].copy()
        test_metrics = test_metrics.sort_values(
            ["strategy", "model_name", "MAE", "feature_set_name"],
            kind="mergesort",
        ).reset_index(drop=True)
        locked_test_report = build_locked_robustness_report(metrics)
        cold_start_comparison = build_cold_start_comparison(locked_test_report)
        age_normalized_comparison = build_age_normalized_comparison(
            locked_test_report
        )
        age_feature_summary = build_age_feature_summary(model_table)
        raw_vs_age_solved = build_raw_vs_age_solved_comparison(
            locked_test_report
        )
        summary = build_robustness_summary(
            locked_test_report,
            cold_start_comparison,
            raw_vs_age_solved,
            snapshot_time,
        )

        output_dir = output_dir.resolve()
        summary_dir = output_dir / "summary"
        tables_dir = output_dir / "tables"
        figures_dir = output_dir / "figures"
        for directory in (summary_dir, tables_dir, figures_dir):
            directory.mkdir(parents=True, exist_ok=True)

        paths = {
            "robustness_summary": summary_dir / "robustness_summary.json",
            "robustness_metrics_all": tables_dir / "robustness_metrics_all.csv",
            "robustness_metrics_test": tables_dir / "robustness_metrics_test.csv",
            "locked_test_results": (
                tables_dir / "robustness_validation_locked_test.csv"
            ),
            "cold_start_comparison": tables_dir / "cold_start_comparison.csv",
            "age_normalized_comparison": (
                tables_dir / "age_normalized_comparison.csv"
            ),
            "age_feature_summary": tables_dir / "age_feature_summary.csv",
            "cold_start_mae_comparison": (
                figures_dir / "cold_start_mae_comparison.png"
            ),
            "cold_start_within_200_comparison": (
                figures_dir / "cold_start_within_200_comparison.png"
            ),
            "age_normalized_mae_comparison": (
                figures_dir / "age_normalized_mae_comparison.png"
            ),
            "age_feature_distributions": (
                figures_dir / "age_feature_distributions.png"
            ),
        }

        write_json(paths["robustness_summary"], summary)
        metrics.to_csv(paths["robustness_metrics_all"], index=False)
        test_metrics.to_csv(paths["robustness_metrics_test"], index=False)
        locked_test_report.to_csv(paths["locked_test_results"], index=False)
        cold_start_comparison.to_csv(paths["cold_start_comparison"], index=False)
        age_normalized_comparison.to_csv(
            paths["age_normalized_comparison"],
            index=False,
        )
        age_feature_summary.to_csv(paths["age_feature_summary"], index=False)
        plot_metric_comparison(
            locked_test_report,
            feature_sets=COLD_START_FEATURE_SETS,
            metric="MAE",
            title="Cold-start robustness: test MAE",
            path=paths["cold_start_mae_comparison"],
        )
        plot_metric_comparison(
            locked_test_report,
            feature_sets=COLD_START_FEATURE_SETS,
            metric="within_200",
            title="Cold-start robustness: within 200 rating points",
            path=paths["cold_start_within_200_comparison"],
        )
        plot_metric_comparison(
            locked_test_report,
            feature_sets=AGE_NORMALIZED_FEATURE_SETS,
            metric="MAE",
            title="Age-normalized solved-count robustness: test MAE",
            path=paths["age_normalized_mae_comparison"],
        )
        plot_age_feature_distributions(
            model_table.loc[:, list(AGE_NORMALIZED_COLUMNS)],
            paths["age_feature_distributions"],
        )

        logger.info(
            "Completed Codeforces robustness experiments",
            extra={
                "event": "robustness_completed",
                "details": {
                    "output_dir": output_dir.as_posix(),
                    "random_seed": config.random_seed,
                    "snapshot_time_utc": snapshot_time.isoformat(),
                    "experiment_count": len(test_metrics),
                },
            },
        )
        return paths
    except Exception:
        logger.exception(
            "Codeforces robustness experiments failed",
            extra={"event": "robustness_failed"},
        )
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the robustness command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run Codeforces robustness experiments."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=DEFAULT_PROCESSED_PATH,
    )
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the robustness CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        paths = run_robustness(
            config_path=args.config,
            processed_path=args.processed_path,
            feature_path=args.feature_path,
            feature_columns_path=args.feature_columns_path,
            contest_split_path=args.contest_split_path,
            time_split_path=args.time_split_path,
            output_dir=args.output_dir,
            log_path=args.log_path,
        )
    except (RobustnessError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote robustness summary: {paths['robustness_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
