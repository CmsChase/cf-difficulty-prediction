"""Exposure-aware analysis for Codeforces difficulty prediction."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cf_diff.baselines import (
    compute_regression_metrics,
    load_baseline_config,
    make_preprocessed_estimator,
)
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
DEFAULT_RAW_MANIFEST_PATH: Final[Path] = Path(
    "data/raw/codeforces/latest/manifest.json"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path("outputs/exposure")
DEFAULT_LOG_PATH: Final[Path] = Path("outputs/logs/exposure_analysis.log")

TARGET_COLUMN: Final[str] = "rating"
STRATEGIES: Final[tuple[str, str]] = ("contest_grouped", "forward_time")
SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "valid", "test")
RAW_SOLVED_COLUMNS: Final[tuple[str, ...]] = (
    "solved_count",
    "log_solved_count",
    "solved_count_missing",
)
AGE_EXPOSURE_COLUMNS: Final[tuple[str, ...]] = (
    "problem_age_days",
    "solves_per_day",
    "log_solves_per_day",
)
AGE_BUCKET_ORDER: Final[tuple[str, str, str, str]] = (
    "0-1y",
    "1-3y",
    "3-5y",
    "5y+",
)
FEATURE_SET_ORDER: Final[tuple[str, ...]] = (
    "metadata_only",
    "raw_solved_only",
    "age_norm_solved_only",
    "metadata_plus_age_norm",
    "full_api_plus_age_norm",
)
FEATURE_SET_LABELS: Final[dict[str, str]] = {
    "metadata_only": "metadata",
    "raw_solved_only": "raw solved",
    "age_norm_solved_only": "age norm",
    "metadata_plus_age_norm": "metadata + age norm",
    "full_api_plus_age_norm": "full + age norm",
}
MISMATCH_GROUP_ORDER: Final[tuple[str, str, str, str]] = (
    "popular_hard",
    "underexposed_easy",
    "popular_easy",
    "rare_hard",
)


class ExposureAnalysisError(RuntimeError):
    """Raised when exposure-aware analysis cannot proceed safely."""


class JsonLogFormatter(logging.Formatter):
    """Format exposure-analysis logs as JSON Lines."""

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
class ExposureModelSpec:
    """Define one deterministic exposure-analysis model."""

    model_name: str = "hist_gradient_boosting_regressor"

    def build(self, seed: int) -> HistGradientBoostingRegressor:
        """Create the configured estimator."""
        return HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=160,
            l2_regularization=0.01,
            random_state=seed,
        )


def configure_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """Create the dedicated structured exposure-analysis logger."""
    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.exposure_analysis")
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
        raise ExposureAnalysisError(
            f"{table_name} lacks required columns: {missing}"
        )


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


def _parse_utc_datetime(value: str) -> pd.Timestamp:
    """Parse an ISO-8601 timestamp as a timezone-aware UTC timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    return timestamp.tz_convert(timezone.utc)


def load_feature_columns(path: Path) -> list[str]:
    """Read feature columns from feature metadata, if available."""
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExposureAnalysisError("Feature-column metadata must be a JSON object.")
    raw_columns = payload.get("feature_columns", [])
    if not isinstance(raw_columns, list) or not all(
        isinstance(column, str) for column in raw_columns
    ):
        raise ExposureAnalysisError("feature_columns.json has invalid feature_columns.")
    return list(raw_columns)


def load_snapshot_time(
    manifest_path: Path,
    feature_table: pd.DataFrame,
) -> pd.Timestamp:
    """Load snapshot time from raw manifest, falling back to max start time."""
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_created_at = payload.get("created_at_utc")
            if isinstance(raw_created_at, str) and raw_created_at.strip():
                return _parse_utc_datetime(raw_created_at)

    _require_columns(feature_table, ("start_time_seconds",), "feature table")
    start_times = pd.to_numeric(
        feature_table["start_time_seconds"],
        errors="coerce",
    ).dropna()
    if start_times.empty:
        raise ExposureAnalysisError("Cannot infer snapshot time from start times.")
    return pd.Timestamp.fromtimestamp(float(start_times.max()), tz=timezone.utc)


def add_exposure_features(
    feature_table: pd.DataFrame,
    snapshot_time: pd.Timestamp,
) -> pd.DataFrame:
    """Return a copy with age-normalized exposure features."""
    _require_columns(
        feature_table,
        ("start_time_seconds", "solved_count"),
        "feature table",
    )
    frame = feature_table.copy()
    snapshot_utc = (
        snapshot_time.tz_localize(timezone.utc)
        if snapshot_time.tzinfo is None
        else snapshot_time.tz_convert(timezone.utc)
    )
    start_seconds = pd.to_numeric(frame["start_time_seconds"], errors="coerce")
    solved_count = (
        pd.to_numeric(frame["solved_count"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    age_days = (snapshot_utc.timestamp() - start_seconds) / 86400.0
    age_days = age_days.where(age_days.notna(), 1.0).clip(lower=1.0)
    solves_per_day = solved_count / age_days

    frame["problem_age_days"] = age_days.astype(float)
    frame["solves_per_day"] = solves_per_day.astype(float)
    frame["log_solves_per_day"] = np.log1p(solves_per_day).astype(float)
    frame["age_bucket"] = assign_age_buckets(frame["problem_age_days"])
    return frame


def assign_age_buckets(age_days: pd.Series) -> pd.Series:
    """Assign problem ages in days to fixed exposure buckets."""
    numeric_age = pd.to_numeric(age_days, errors="coerce")
    values = np.select(
        [
            numeric_age <= 365.0,
            numeric_age <= 3 * 365.0,
            numeric_age <= 5 * 365.0,
            numeric_age > 5 * 365.0,
        ],
        list(AGE_BUCKET_ORDER),
        default=pd.NA,
    )
    return pd.Series(
        pd.Categorical(values, categories=list(AGE_BUCKET_ORDER), ordered=True),
        index=age_days.index,
        name="age_bucket",
    )


def build_feature_groups(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Select available columns for exposure-analysis feature groups."""
    feature_columns = list(feature_columns or [])
    index = [
        column
        for column in ("index_letter", "index_number", "index_rank")
        if column in frame.columns
    ]
    tags = sorted(
        column
        for column in frame.columns
        if column == "tag_count" or column.startswith("tag__")
    )
    points = [
        column
        for column in ("has_points", "points")
        if column in frame.columns
    ]
    raw_solved = [column for column in RAW_SOLVED_COLUMNS if column in frame.columns]
    age_norm = [column for column in AGE_EXPOSURE_COLUMNS if column in frame.columns]
    if feature_columns:
        full_api = [column for column in feature_columns if column in frame.columns]
    else:
        full_api = _dedupe_preserve_order([*index, *tags, *points, *raw_solved])
    full_api = [
        column
        for column in full_api
        if column not in AGE_EXPOSURE_COLUMNS and column in frame.columns
    ]

    groups = {
        "metadata": sorted(_dedupe_preserve_order([*index, *tags, *points])),
        "raw_solved": sorted(raw_solved),
        "age_norm": sorted(age_norm),
        "full_api": _dedupe_preserve_order(full_api),
    }
    if not groups["metadata"]:
        raise ExposureAnalysisError("No metadata feature columns are available.")
    if not groups["raw_solved"]:
        raise ExposureAnalysisError("No raw solved-count columns are available.")
    if len(groups["age_norm"]) != len(AGE_EXPOSURE_COLUMNS):
        raise ExposureAnalysisError("Age-normalized exposure columns are missing.")
    if not groups["full_api"]:
        raise ExposureAnalysisError("No full API feature columns are available.")
    return groups


def build_feature_sets(
    feature_groups: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Build the required exposure-analysis feature settings."""
    metadata = list(feature_groups["metadata"])
    raw_solved = list(feature_groups["raw_solved"])
    age_norm = list(feature_groups["age_norm"])
    full_api = list(feature_groups["full_api"])
    sets = {
        "metadata_only": metadata,
        "raw_solved_only": raw_solved,
        "age_norm_solved_only": age_norm,
        "metadata_plus_age_norm": _dedupe_preserve_order([*metadata, *age_norm]),
        "full_api_plus_age_norm": _dedupe_preserve_order([*full_api, *age_norm]),
    }
    solved_forbidden = set(RAW_SOLVED_COLUMNS)
    age_forbidden = set(AGE_EXPOSURE_COLUMNS)
    sets["metadata_only"] = [
        column
        for column in sets["metadata_only"]
        if column not in solved_forbidden and column not in age_forbidden
    ]
    sets["metadata_plus_age_norm"] = [
        column
        for column in sets["metadata_plus_age_norm"]
        if column not in solved_forbidden
    ]
    for name, columns in sets.items():
        if not columns:
            raise ExposureAnalysisError(f"Feature set {name!r} has no columns.")
    return sets


def join_split_assignments(
    frame: pd.DataFrame,
    split_assignment: pd.DataFrame,
) -> pd.DataFrame:
    """Join model rows with row-level split assignments."""
    _require_columns(frame, ("contest_id", "index", TARGET_COLUMN), "feature table")
    _require_columns(
        split_assignment,
        ("contest_id", "index", "split_name"),
        "split assignment",
    )
    joined = frame.merge(
        split_assignment.loc[:, ["contest_id", "index", "split_name"]],
        on=["contest_id", "index"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(frame):
        raise ExposureAnalysisError(
            "Split assignment did not match every feature-table row: "
            f"{len(joined)} of {len(frame)} rows matched."
        )
    missing_splits = [
        split_name
        for split_name in ("train", "test")
        if not joined["split_name"].eq(split_name).any()
    ]
    if missing_splits:
        raise ExposureAnalysisError(f"Split assignment has empty splits: {missing_splits}")
    return joined


def _fit_predict(
    joined: pd.DataFrame,
    feature_columns: Sequence[str],
    model_spec: ExposureModelSpec,
    seed: int,
) -> pd.Series:
    """Fit one exposure-analysis model on train rows and predict all rows."""
    train = joined.loc[joined["split_name"].eq("train")].copy()
    train_x = train.loc[:, list(feature_columns)]
    train_y = pd.to_numeric(train[TARGET_COLUMN], errors="raise").to_numpy(
        dtype=float
    )
    model = make_preprocessed_estimator(model_spec.build(seed), train_x)
    model.fit(train_x, train_y)
    predictions = model.predict(joined.loc[:, list(feature_columns)])
    return pd.Series(predictions, index=joined.index, dtype=float)


def evaluate_age_bucket_metrics(
    feature_table: pd.DataFrame,
    split_assignment: pd.DataFrame,
    *,
    strategy: str,
    feature_sets: Mapping[str, Sequence[str]],
    seed: int,
    model_spec: ExposureModelSpec | None = None,
) -> pd.DataFrame:
    """Evaluate feature settings by age bucket on the test split."""
    model_spec = model_spec or ExposureModelSpec()
    joined = join_split_assignments(feature_table, split_assignment)
    test = joined.loc[joined["split_name"].eq("test")]
    rows: list[dict[str, object]] = []
    for feature_set_name in FEATURE_SET_ORDER:
        feature_columns = list(feature_sets[feature_set_name])
        predictions = _fit_predict(joined, feature_columns, model_spec, seed)
        test_predictions = predictions.loc[test.index]
        for age_bucket in AGE_BUCKET_ORDER:
            mask = test["age_bucket"].astype(str).eq(age_bucket)
            if not mask.any():
                continue
            metrics = compute_regression_metrics(
                test.loc[mask, TARGET_COLUMN],
                test_predictions.loc[mask],
            )
            rows.append(
                {
                    "strategy": strategy,
                    "model_name": model_spec.model_name,
                    "feature_set_name": feature_set_name,
                    "split_name": "test",
                    "age_bucket": age_bucket,
                    "row_count": int(mask.sum()),
                    "feature_count": len(feature_columns),
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["strategy", "feature_set_name", "age_bucket"],
        kind="mergesort",
    ).reset_index(drop=True)


def _tags_from_row(row: pd.Series, tag_columns: Sequence[str]) -> str:
    """Extract semicolon-separated tags from raw or one-hot tag fields."""
    if "tags" in row.index and isinstance(row["tags"], (list, tuple, set)):
        return ";".join(sorted(str(tag) for tag in row["tags"]))
    tags: list[str] = []
    for column in tag_columns:
        value = row.get(column, 0)
        if pd.notna(value) and float(value) > 0:
            tags.append(column.removeprefix("tag__").replace("_", " "))
    return ";".join(tags)


def select_mismatch_examples(
    frame: pd.DataFrame,
    *,
    max_examples_per_group: int = 20,
) -> pd.DataFrame:
    """Select popularity-difficulty mismatch examples by quantile thresholds."""
    _require_columns(
        frame,
        (
            "contest_id",
            "index",
            "name",
            "rating",
            "solved_count",
            "problem_age_days",
            "solves_per_day",
            "log_solves_per_day",
        ),
        "feature table",
    )
    source = frame.copy()
    rating = pd.to_numeric(source["rating"], errors="coerce")
    popularity = pd.to_numeric(source["log_solves_per_day"], errors="coerce")
    high_rating = rating >= rating.quantile(0.85)
    low_rating = rating <= rating.quantile(0.15)
    high_popularity = popularity >= popularity.quantile(0.85)
    low_popularity = popularity <= popularity.quantile(0.15)
    conditions = {
        "popular_hard": high_rating & high_popularity,
        "underexposed_easy": low_rating & low_popularity,
        "popular_easy": low_rating & high_popularity,
        "rare_hard": high_rating & low_popularity,
    }
    tag_columns = sorted(column for column in source.columns if column.startswith("tag__"))
    rows: list[pd.DataFrame] = []
    for group_name in MISMATCH_GROUP_ORDER:
        group = source.loc[conditions[group_name]].copy()
        if group.empty:
            continue
        group["mismatch_group"] = group_name
        group["tags"] = group.apply(
            lambda row: _tags_from_row(row, tag_columns),
            axis=1,
        )
        group = group.sort_values(
            ["rating", "log_solves_per_day", "contest_id", "index"],
            ascending=[
                group_name not in {"popular_hard", "rare_hard"},
                group_name in {"underexposed_easy", "rare_hard"},
                True,
                True,
            ],
            kind="mergesort",
        ).head(max_examples_per_group)
        rows.append(group)
    if not rows:
        return pd.DataFrame(
            columns=[
                "contest_id",
                "index",
                "name",
                "rating",
                "solved_count",
                "problem_age_days",
                "solves_per_day",
                "log_solves_per_day",
                "tags",
                "mismatch_group",
            ]
        )
    output = pd.concat(rows, ignore_index=True)
    output_columns = [
        "contest_id",
        "index",
        "name",
        "rating",
        "solved_count",
        "problem_age_days",
        "solves_per_day",
        "log_solves_per_day",
        "tags",
        "mismatch_group",
    ]
    return output.loc[:, output_columns].sort_values(
        ["mismatch_group", "rating", "log_solves_per_day", "contest_id", "index"],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_correlation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute rating correlations with exposure-related signals."""
    signals = [
        "solved_count",
        "log_solved_count",
        "solves_per_day",
        "log_solves_per_day",
        "problem_age_days",
    ]
    rows: list[dict[str, object]] = []

    def append_rows(scope: str, age_bucket: str | None, group: pd.DataFrame) -> None:
        y = pd.to_numeric(group[TARGET_COLUMN], errors="coerce")
        for signal in signals:
            if signal not in group.columns:
                continue
            x = pd.to_numeric(group[signal], errors="coerce")
            valid = pd.DataFrame({"rating": y, signal: x}).dropna()
            correlation = (
                valid["rating"].corr(valid[signal])
                if len(valid) >= 2
                else np.nan
            )
            rows.append(
                {
                    "scope": scope,
                    "age_bucket": age_bucket or "",
                    "signal": signal,
                    "row_count": int(len(valid)),
                    "pearson_correlation": _finite_float(correlation),
                }
            )

    append_rows("overall", None, frame)
    for age_bucket in AGE_BUCKET_ORDER:
        bucket = frame.loc[frame["age_bucket"].astype(str).eq(age_bucket)]
        append_rows("age_bucket", age_bucket, bucket)
    return pd.DataFrame(rows).sort_values(
        ["scope", "age_bucket", "signal"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_exposure_summary(
    frame: pd.DataFrame,
    age_bucket_metrics: pd.DataFrame,
    mismatch_examples: pd.DataFrame,
    snapshot_time: pd.Timestamp,
) -> dict[str, object]:
    """Create the machine-readable exposure-analysis summary."""
    age_counts = (
        frame["age_bucket"].astype(str).value_counts().reindex(AGE_BUCKET_ORDER, fill_value=0)
    )
    best = {}
    if not age_bucket_metrics.empty:
        average_metrics = (
            age_bucket_metrics.groupby(
                ["strategy", "model_name", "feature_set_name"],
                as_index=False,
            )["MAE"]
            .mean()
            .sort_values(["MAE", "strategy", "feature_set_name"], kind="mergesort")
        )
        row = average_metrics.iloc[0]
        best = {
            "strategy": row["strategy"],
            "model_name": row["model_name"],
            "feature_set_name": row["feature_set_name"],
            "mean_age_bucket_MAE": _finite_float(row["MAE"]),
        }
    mismatch_counts = (
        mismatch_examples["mismatch_group"].value_counts()
        if "mismatch_group" in mismatch_examples.columns
        else pd.Series(dtype=int)
    )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_time_utc": snapshot_time.isoformat(),
        "total_rows": int(len(frame)),
        "age_bucket_counts": {
            bucket: int(age_counts.loc[bucket]) for bucket in AGE_BUCKET_ORDER
        },
        "main_finding_from_age_bucket_analysis": {
            **best,
            "note": (
                "Age-bucket metrics compare exposure-sensitive feature settings "
                "on held-out test rows; lower MAE does not imply causality."
            ),
        },
        "main_finding_from_mismatch_analysis": {
            "mismatch_group_counts": {
                group: int(mismatch_counts.get(group, 0))
                for group in MISMATCH_GROUP_ORDER
            },
            "note": (
                "Mismatch examples identify problems where rating and "
                "age-normalized popularity do not align cleanly."
            ),
        },
        "conservative_interpretation_notes": [
            "Solved count is useful but not a pure difficulty signal.",
            "Age normalization is only a simple proxy for exposure.",
            "Mismatch examples are diagnostic, not causal proof.",
            "This analysis does not use submission-level time series.",
        ],
    }


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Persist a matplotlib figure with stable output settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "cf-diff-exposure-analysis"},
    )
    plt.close(fig)


def _empty_figure(title: str) -> plt.Figure:
    """Create a readable placeholder figure for empty inputs."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title(title)
    ax.set_axis_off()
    return fig


def plot_age_bucket_mae(age_bucket_metrics: pd.DataFrame, path: Path) -> None:
    """Plot age-bucket MAE with separate panels by split strategy."""
    title = "Test MAE by age bucket and feature setting"
    if age_bucket_metrics.empty:
        _save_figure(_empty_figure(title), path)
        return
    strategies = [
        strategy
        for strategy in STRATEGIES
        if age_bucket_metrics["strategy"].eq(strategy).any()
    ]
    feature_sets = [
        feature_set
        for feature_set in FEATURE_SET_ORDER
        if age_bucket_metrics["feature_set_name"].eq(feature_set).any()
    ]
    fig, axes = plt.subplots(
        1,
        len(strategies),
        figsize=(18, 6.5),
        sharey=True,
        squeeze=False,
    )
    x = np.arange(len(AGE_BUCKET_ORDER))
    width = min(0.16, 0.8 / max(len(feature_sets), 1))
    cmap = plt.get_cmap("tab10")
    for axis_index, strategy in enumerate(strategies):
        ax = axes[0, axis_index]
        strategy_frame = age_bucket_metrics.loc[
            age_bucket_metrics["strategy"].eq(strategy)
        ]
        for feature_index, feature_set in enumerate(feature_sets):
            values = (
                strategy_frame.loc[
                    strategy_frame["feature_set_name"].eq(feature_set)
                ]
                .set_index("age_bucket")
                .reindex(AGE_BUCKET_ORDER)["MAE"]
            )
            offset = (feature_index - (len(feature_sets) - 1) / 2) * width
            ax.bar(
                x + offset,
                pd.to_numeric(values, errors="coerce"),
                width=width,
                label=FEATURE_SET_LABELS.get(feature_set, feature_set),
                color=cmap(feature_index),
            )
        ax.set_title(strategy)
        ax.set_xlabel("Problem age bucket")
        ax.set_xticks(x)
        ax.set_xticklabels(AGE_BUCKET_ORDER)
        ax.grid(axis="y", alpha=0.25)
        if axis_index == 0:
            ax.set_ylabel("MAE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.95),
    )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    _save_figure(fig, path)


def plot_rating_vs_log_solves(
    frame: pd.DataFrame,
    mismatch_examples: pd.DataFrame,
    path: Path,
) -> None:
    """Plot official rating against age-normalized popularity."""
    title = "Official rating vs log solves per day"
    if frame.empty:
        _save_figure(_empty_figure(title), path)
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        pd.to_numeric(frame["log_solves_per_day"], errors="coerce"),
        pd.to_numeric(frame["rating"], errors="coerce"),
        s=14,
        alpha=0.22,
        color="#6B7280",
        label="all problems",
        linewidths=0,
    )
    colors = {
        "popular_hard": "#D62728",
        "underexposed_easy": "#1F77B4",
        "popular_easy": "#2CA02C",
        "rare_hard": "#9467BD",
    }
    if not mismatch_examples.empty:
        for group_name in MISMATCH_GROUP_ORDER:
            group = mismatch_examples.loc[
                mismatch_examples["mismatch_group"].eq(group_name)
            ]
            if group.empty:
                continue
            ax.scatter(
                pd.to_numeric(group["log_solves_per_day"], errors="coerce"),
                pd.to_numeric(group["rating"], errors="coerce"),
                s=42,
                alpha=0.85,
                color=colors[group_name],
                label=group_name,
                edgecolors="white",
                linewidths=0.5,
            )
    ax.set_title(title)
    ax.set_xlabel("log_solves_per_day")
    ax.set_ylabel("Official rating")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, markerscale=1.2)
    fig.tight_layout()
    _save_figure(fig, path)


def run_exposure_analysis(
    *,
    config_path: Path,
    feature_path: Path,
    feature_columns_path: Path,
    contest_split_path: Path,
    time_split_path: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Path]:
    """Run exposure-aware analysis and write tables, figures, and summary."""
    logger = configure_logger(log_path)
    try:
        config = load_baseline_config(config_path)
        feature_table = pd.read_parquet(feature_path, engine="pyarrow")
        feature_columns = load_feature_columns(feature_columns_path)
        snapshot_time = load_snapshot_time(DEFAULT_RAW_MANIFEST_PATH, feature_table)
        feature_table = add_exposure_features(feature_table, snapshot_time)
        feature_groups = build_feature_groups(feature_table, feature_columns)
        feature_sets = build_feature_sets(feature_groups)
        contest_split = pd.read_parquet(contest_split_path, engine="pyarrow")
        time_split = pd.read_parquet(time_split_path, engine="pyarrow")

        age_bucket_metrics = pd.concat(
            [
                evaluate_age_bucket_metrics(
                    feature_table,
                    contest_split,
                    strategy="contest_grouped",
                    feature_sets=feature_sets,
                    seed=config.random_seed,
                ),
                evaluate_age_bucket_metrics(
                    feature_table,
                    time_split,
                    strategy="forward_time",
                    feature_sets=feature_sets,
                    seed=config.random_seed,
                ),
            ],
            ignore_index=True,
        ).sort_values(
            ["strategy", "feature_set_name", "age_bucket"],
            kind="mergesort",
        )
        mismatch_examples = select_mismatch_examples(feature_table)
        correlation_summary = build_correlation_summary(feature_table)
        summary = build_exposure_summary(
            feature_table,
            age_bucket_metrics,
            mismatch_examples,
            snapshot_time,
        )

        output_dir = output_dir.resolve()
        summary_dir = output_dir / "summary"
        tables_dir = output_dir / "tables"
        figures_dir = output_dir / "figures"
        for directory in (summary_dir, tables_dir, figures_dir):
            directory.mkdir(parents=True, exist_ok=True)

        paths = {
            "exposure_summary": summary_dir / "exposure_summary.json",
            "age_bucket_metrics": tables_dir / "age_bucket_metrics.csv",
            "popularity_difficulty_mismatch_examples": (
                tables_dir / "popularity_difficulty_mismatch_examples.csv"
            ),
            "exposure_correlation_summary": (
                tables_dir / "exposure_correlation_summary.csv"
            ),
            "age_bucket_mae_by_feature_set": (
                figures_dir / "age_bucket_mae_by_feature_set.png"
            ),
            "rating_vs_log_solves_per_day": (
                figures_dir / "rating_vs_log_solves_per_day.png"
            ),
        }
        write_json(paths["exposure_summary"], summary)
        age_bucket_metrics.to_csv(paths["age_bucket_metrics"], index=False)
        mismatch_examples.to_csv(
            paths["popularity_difficulty_mismatch_examples"],
            index=False,
        )
        correlation_summary.to_csv(
            paths["exposure_correlation_summary"],
            index=False,
        )
        plot_age_bucket_mae(
            age_bucket_metrics,
            paths["age_bucket_mae_by_feature_set"],
        )
        plot_rating_vs_log_solves(
            feature_table,
            mismatch_examples,
            paths["rating_vs_log_solves_per_day"],
        )

        logger.info(
            "Completed Codeforces exposure-aware analysis",
            extra={
                "event": "exposure_analysis_completed",
                "details": {
                    "output_dir": output_dir.as_posix(),
                    "random_seed": config.random_seed,
                    "snapshot_time_utc": snapshot_time.isoformat(),
                    "row_count": len(feature_table),
                },
            },
        )
        return paths
    except Exception:
        logger.exception(
            "Codeforces exposure-aware analysis failed",
            extra={"event": "exposure_analysis_failed"},
        )
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the exposure-analysis command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run exposure-aware Codeforces difficulty analysis."
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the exposure-analysis CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        paths = run_exposure_analysis(
            config_path=args.config,
            feature_path=args.feature_path,
            feature_columns_path=args.feature_columns_path,
            contest_split_path=args.contest_split_path,
            time_split_path=args.time_split_path,
            output_dir=args.output_dir,
            log_path=args.log_path,
        )
    except (ExposureAnalysisError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote exposure summary: {paths['exposure_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
