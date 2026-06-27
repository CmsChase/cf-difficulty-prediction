"""Rolling-window temporal validation and concept-drift diagnostics."""

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
DEFAULT_RAW_MANIFEST_PATH: Final[Path] = Path(
    "data/raw/codeforces/latest/manifest.json"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path("outputs/temporal_validation")
DEFAULT_LOG_PATH: Final[Path] = Path("outputs/logs/temporal_validation.log")

TARGET_COLUMN: Final[str] = "rating"
TIME_COLUMNS: Final[tuple[str, str]] = (
    "start_time_seconds",
    "contest_start_time_seconds",
)
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
FEATURE_SET_ORDER: Final[tuple[str, ...]] = (
    "metadata_only",
    "raw_solved_only",
    "full_api",
    "full_api_plus_age_norm",
)
FEATURE_SET_LABELS: Final[dict[str, str]] = {
    "metadata_only": "metadata",
    "raw_solved_only": "raw solved",
    "full_api": "full API",
    "full_api_plus_age_norm": "full API + age norm",
}
DRIFT_CANDIDATE_COLUMNS: Final[tuple[str, ...]] = (
    "rating",
    "solved_count",
    "log_solved_count",
    "problem_age_days",
    "solves_per_day",
    "log_solves_per_day",
    "tag_count",
    "index_rank",
    "points",
)
ROLLING_TRAIN_FRACTIONS: Final[tuple[float, float, float, float]] = (
    0.5,
    0.6,
    0.7,
    0.8,
)
ROLLING_TEST_FRACTION: Final[float] = 0.1


class TemporalValidationError(RuntimeError):
    """Raised when temporal validation cannot proceed safely."""


class JsonLogFormatter(logging.Formatter):
    """Format temporal-validation logs as JSON Lines."""

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
class RollingFold:
    """Contest-level expanding-window fold definition."""

    fold_id: int
    train_contest_ids: tuple[object, ...]
    test_contest_ids: tuple[object, ...]
    train_start_time_seconds: float
    train_end_time_seconds: float
    test_start_time_seconds: float
    test_end_time_seconds: float


def configure_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """Create the dedicated structured temporal-validation logger."""
    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.temporal_validation")
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
        raise TemporalValidationError(
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


def _time_column(frame: pd.DataFrame) -> str:
    """Return the available contest-start-time column or fail clearly."""
    for column in TIME_COLUMNS:
        if column in frame.columns:
            return column
    raise TemporalValidationError(
        "Feature table must contain start_time_seconds or "
        "contest_start_time_seconds for temporal validation."
    )


def _seconds_to_utc(seconds: float) -> str:
    """Convert Unix seconds to an ISO-8601 UTC timestamp."""
    return pd.Timestamp.fromtimestamp(float(seconds), tz=timezone.utc).isoformat()


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
        raise TemporalValidationError(
            "Feature-column metadata must be a JSON object."
        )
    raw_columns = payload.get("feature_columns", [])
    if not isinstance(raw_columns, list) or not all(
        isinstance(column, str) for column in raw_columns
    ):
        raise TemporalValidationError(
            "feature_columns.json has invalid feature_columns."
        )
    return list(raw_columns)


def load_snapshot_time(
    manifest_path: Path,
    feature_table: pd.DataFrame,
) -> pd.Timestamp:
    """Load snapshot time from raw manifest, falling back to max contest time."""
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_created_at = payload.get("created_at_utc")
            if isinstance(raw_created_at, str) and raw_created_at.strip():
                return _parse_utc_datetime(raw_created_at)

    time_column = _time_column(feature_table)
    start_times = pd.to_numeric(feature_table[time_column], errors="coerce").dropna()
    if start_times.empty:
        raise TemporalValidationError("Cannot infer snapshot time from start times.")
    return pd.Timestamp.fromtimestamp(float(start_times.max()), tz=timezone.utc)


def add_age_normalized_features(
    feature_table: pd.DataFrame,
    snapshot_time: pd.Timestamp,
) -> pd.DataFrame:
    """Return a copy with age-normalized exposure features."""
    time_column = _time_column(feature_table)
    _require_columns(feature_table, ("solved_count",), "feature table")
    frame = feature_table.copy()
    snapshot_utc = (
        snapshot_time.tz_localize(timezone.utc)
        if snapshot_time.tzinfo is None
        else snapshot_time.tz_convert(timezone.utc)
    )
    start_seconds = pd.to_numeric(frame[time_column], errors="coerce")
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
    return frame


def build_contest_time_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Create one sorted row per contest with a stable start time."""
    time_column = _time_column(feature_table)
    _require_columns(feature_table, ("contest_id", time_column), "feature table")
    contests = feature_table.loc[:, ["contest_id", time_column]].copy()
    contests[time_column] = pd.to_numeric(contests[time_column], errors="coerce")
    contests = contests.dropna(subset=["contest_id", time_column])
    if contests.empty:
        raise TemporalValidationError("No contests with valid start times.")
    contest_times = (
        contests.groupby("contest_id", as_index=False)[time_column]
        .min()
        .rename(columns={time_column: "start_time_seconds"})
        .sort_values(["start_time_seconds", "contest_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(contest_times) < 4:
        raise TemporalValidationError(
            "At least four contests are required for rolling temporal validation."
        )
    return contest_times


def build_rolling_folds(
    contest_times: pd.DataFrame,
    *,
    train_fractions: Sequence[float] = ROLLING_TRAIN_FRACTIONS,
    test_fraction: float = ROLLING_TEST_FRACTION,
) -> list[RollingFold]:
    """Build expanding-window contest-level folds sorted by contest time."""
    _require_columns(
        contest_times,
        ("contest_id", "start_time_seconds"),
        "contest time table",
    )
    contests = contest_times.sort_values(
        ["start_time_seconds", "contest_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    n_contests = len(contests)
    folds: list[RollingFold] = []
    for fold_id, train_fraction in enumerate(train_fractions, start=1):
        if train_fraction <= 0 or train_fraction >= 1:
            raise TemporalValidationError("Train fractions must be in (0, 1).")
        train_end = int(round(n_contests * train_fraction))
        train_end = min(max(train_end, 1), n_contests - 1)
        suggested_test_end = int(round(n_contests * (train_fraction + test_fraction)))
        test_end = min(max(suggested_test_end, train_end + 1), n_contests)
        if test_end <= train_end:
            continue
        train = contests.iloc[:train_end]
        test = contests.iloc[train_end:test_end]
        if train.empty or test.empty:
            continue
        folds.append(
            RollingFold(
                fold_id=fold_id,
                train_contest_ids=tuple(train["contest_id"].tolist()),
                test_contest_ids=tuple(test["contest_id"].tolist()),
                train_start_time_seconds=float(train["start_time_seconds"].min()),
                train_end_time_seconds=float(train["start_time_seconds"].max()),
                test_start_time_seconds=float(test["start_time_seconds"].min()),
                test_end_time_seconds=float(test["start_time_seconds"].max()),
            )
        )
    if not folds:
        raise TemporalValidationError("No non-empty rolling folds could be created.")
    return folds


def build_feature_groups(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Select columns for temporal-validation feature settings."""
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
        raise TemporalValidationError("No metadata feature columns are available.")
    if not groups["raw_solved"]:
        raise TemporalValidationError("No raw solved-count columns are available.")
    if len(groups["age_norm"]) != len(AGE_EXPOSURE_COLUMNS):
        raise TemporalValidationError("Age-normalized exposure columns are missing.")
    if not groups["full_api"]:
        raise TemporalValidationError("No full API feature columns are available.")
    return groups


def build_feature_sets(
    feature_groups: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Build the required temporal-validation feature settings."""
    metadata = list(feature_groups["metadata"])
    raw_solved = list(feature_groups["raw_solved"])
    full_api = list(feature_groups["full_api"])
    age_norm = list(feature_groups["age_norm"])
    sets = {
        "metadata_only": [
            column
            for column in metadata
            if column not in RAW_SOLVED_COLUMNS
            and column not in AGE_EXPOSURE_COLUMNS
        ],
        "raw_solved_only": raw_solved,
        "full_api": full_api,
        "full_api_plus_age_norm": _dedupe_preserve_order([*full_api, *age_norm]),
    }
    for name, columns in sets.items():
        if not columns:
            raise TemporalValidationError(f"Feature set {name!r} has no columns.")
    return sets


def _fit_predict_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    seed: int,
) -> np.ndarray:
    """Fit HGB on one train window and predict the following test window."""
    train_x = train.loc[:, list(feature_columns)]
    train_y = pd.to_numeric(train[TARGET_COLUMN], errors="raise").to_numpy(
        dtype=float
    )
    model = make_preprocessed_estimator(
        HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=160,
            l2_regularization=0.01,
            random_state=seed,
        ),
        train_x,
    )
    model.fit(train_x, train_y)
    return np.asarray(
        model.predict(test.loc[:, list(feature_columns)]),
        dtype=float,
    )


def evaluate_rolling_windows(
    feature_table: pd.DataFrame,
    folds: Sequence[RollingFold],
    feature_sets: Mapping[str, Sequence[str]],
    *,
    seed: int,
) -> pd.DataFrame:
    """Evaluate HGB across rolling contest-level folds."""
    _require_columns(
        feature_table,
        ("contest_id", TARGET_COLUMN),
        "feature table",
    )
    rows: list[dict[str, object]] = []
    for fold in folds:
        train = feature_table.loc[
            feature_table["contest_id"].isin(fold.train_contest_ids)
        ].copy()
        test = feature_table.loc[
            feature_table["contest_id"].isin(fold.test_contest_ids)
        ].copy()
        if train.empty or test.empty:
            raise TemporalValidationError(
                f"Fold {fold.fold_id} has empty train or test rows."
            )
        for feature_set_name in FEATURE_SET_ORDER:
            feature_columns = list(feature_sets[feature_set_name])
            predictions = _fit_predict_fold(train, test, feature_columns, seed)
            metrics = compute_regression_metrics(test[TARGET_COLUMN], predictions)
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "feature_set_name": feature_set_name,
                    "model_name": "hist_gradient_boosting_regressor",
                    "train_contest_count": len(fold.train_contest_ids),
                    "test_contest_count": len(fold.test_contest_ids),
                    "train_row_count": int(len(train)),
                    "test_row_count": int(len(test)),
                    "train_start_time_utc": _seconds_to_utc(
                        fold.train_start_time_seconds
                    ),
                    "train_end_time_utc": _seconds_to_utc(
                        fold.train_end_time_seconds
                    ),
                    "test_start_time_utc": _seconds_to_utc(
                        fold.test_start_time_seconds
                    ),
                    "test_end_time_utc": _seconds_to_utc(
                        fold.test_end_time_seconds
                    ),
                    **metrics,
                    "feature_count": len(feature_columns),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["fold_id", "feature_set_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def _quantile(series: pd.Series, q: float) -> float | None:
    """Return a finite quantile or None for empty data."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.quantile(q))


def _mean(series: pd.Series) -> float | None:
    """Return a finite mean or None for empty data."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _median(series: pd.Series) -> float | None:
    """Return a finite median or None for empty data."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.median())


def build_temporal_drift_summary(
    feature_table: pd.DataFrame,
    folds: Sequence[RollingFold],
    *,
    columns: Sequence[str] = DRIFT_CANDIDATE_COLUMNS,
) -> pd.DataFrame:
    """Compute train-vs-test descriptive drift diagnostics by fold."""
    rows: list[dict[str, object]] = []
    for fold in folds:
        train = feature_table.loc[
            feature_table["contest_id"].isin(fold.train_contest_ids)
        ]
        test = feature_table.loc[
            feature_table["contest_id"].isin(fold.test_contest_ids)
        ]
        for column in columns:
            if column not in feature_table.columns:
                continue
            train_series = pd.to_numeric(train[column], errors="coerce").dropna()
            test_series = pd.to_numeric(test[column], errors="coerce").dropna()
            if train_series.empty or test_series.empty:
                continue
            train_mean = _mean(train_series)
            test_mean = _mean(test_series)
            train_median = _median(train_series)
            test_median = _median(test_series)
            train_p90 = _quantile(train_series, 0.90)
            test_p90 = _quantile(test_series, 0.90)
            train_std = float(train_series.std(ddof=0))
            standardized = (
                abs(float(test_mean) - float(train_mean)) / train_std
                if train_std > 0 and train_mean is not None and test_mean is not None
                else None
            )
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "column_name": column,
                    "train_mean": _finite_float(train_mean),
                    "test_mean": _finite_float(test_mean),
                    "mean_difference": _finite_float(
                        None
                        if train_mean is None or test_mean is None
                        else test_mean - train_mean
                    ),
                    "train_median": _finite_float(train_median),
                    "test_median": _finite_float(test_median),
                    "median_difference": _finite_float(
                        None
                        if train_median is None or test_median is None
                        else test_median - train_median
                    ),
                    "train_p90": _finite_float(train_p90),
                    "test_p90": _finite_float(test_p90),
                    "p90_difference": _finite_float(
                        None
                        if train_p90 is None or test_p90 is None
                        else test_p90 - train_p90
                    ),
                    "absolute_standardized_mean_difference": _finite_float(
                        standardized
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["fold_id", "column_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_temporal_validation_summary(
    feature_table: pd.DataFrame,
    folds: Sequence[RollingFold],
    metrics: pd.DataFrame,
    snapshot_time: pd.Timestamp,
) -> dict[str, object]:
    """Create a conservative machine-readable temporal-validation summary."""
    average_mae = (
        metrics.groupby("feature_set_name", as_index=False)["MAE"]
        .mean()
        .sort_values(["MAE", "feature_set_name"], kind="mergesort")
    )
    average_mae_by_feature = {
        row["feature_set_name"]: _finite_float(row["MAE"])
        for _, row in average_mae.iterrows()
    }
    best_row = average_mae.iloc[0]
    metadata_mae = average_mae_by_feature.get("metadata_only")
    full_api_mae = average_mae_by_feature.get("full_api")
    metadata_consistently_worse = False
    if metadata_mae is not None and full_api_mae is not None:
        paired = metrics.pivot_table(
            index="fold_id",
            columns="feature_set_name",
            values="MAE",
            aggfunc="first",
        )
        if {"metadata_only", "full_api"}.issubset(paired.columns):
            metadata_consistently_worse = bool(
                (paired["metadata_only"] > paired["full_api"]).all()
            )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_time_utc": snapshot_time.isoformat(),
        "total_rows": int(len(feature_table)),
        "rolling_fold_count": int(len(folds)),
        "best_feature_setting_by_average_MAE": {
            "feature_set_name": best_row["feature_set_name"],
            "average_MAE": _finite_float(best_row["MAE"]),
        },
        "average_MAE_by_feature_setting": average_mae_by_feature,
        "metadata_only_consistently_worse_than_full_api": (
            metadata_consistently_worse
        ),
        "conservative_interpretation_notes": [
            (
                "Rolling-window validation tests temporal stability but does "
                "not prove future performance."
            ),
            "Solved-count features remain post-publication signals.",
            (
                "Age-normalized exposure features are simple proxies, not full "
                "solve-curve models."
            ),
            "Concept drift diagnostics are descriptive, not causal proof.",
        ],
    }


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Persist a matplotlib figure with stable output settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "cf-diff-temporal-validation"},
    )
    plt.close(fig)


def _empty_figure(title: str) -> plt.Figure:
    """Create a readable placeholder figure for empty inputs."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title(title)
    ax.set_axis_off()
    return fig


def plot_rolling_metric(metrics: pd.DataFrame, metric: str, path: Path) -> None:
    """Plot one rolling-window metric by fold and feature setting."""
    title = f"Rolling-window {metric}"
    if metrics.empty:
        _save_figure(_empty_figure(title), path)
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for feature_set in FEATURE_SET_ORDER:
        subset = metrics.loc[metrics["feature_set_name"].eq(feature_set)]
        if subset.empty:
            continue
        subset = subset.sort_values("fold_id", kind="mergesort")
        ax.plot(
            subset["fold_id"],
            pd.to_numeric(subset[metric], errors="coerce"),
            marker="o",
            linewidth=2,
            label=FEATURE_SET_LABELS.get(feature_set, feature_set),
        )
    ax.set_title(title)
    ax.set_xlabel("Rolling fold")
    ax.set_ylabel(metric)
    ax.set_xticks(sorted(metrics["fold_id"].unique()))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, path)


def plot_temporal_drift_summary(drift_summary: pd.DataFrame, path: Path) -> None:
    """Plot standardized mean differences for selected drift columns."""
    title = "Temporal drift diagnostics"
    if drift_summary.empty:
        _save_figure(_empty_figure(title), path)
        return
    ranked_columns = (
        drift_summary.groupby("column_name")[
            "absolute_standardized_mean_difference"
        ]
        .mean()
        .sort_values(ascending=False)
        .dropna()
        .head(8)
        .index.tolist()
    )
    if not ranked_columns:
        _save_figure(_empty_figure(title), path)
        return
    subset = drift_summary.loc[drift_summary["column_name"].isin(ranked_columns)]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for column in ranked_columns:
        column_frame = subset.loc[subset["column_name"].eq(column)].sort_values(
            "fold_id",
            kind="mergesort",
        )
        ax.plot(
            column_frame["fold_id"],
            column_frame["absolute_standardized_mean_difference"],
            marker="o",
            linewidth=2,
            label=column,
        )
    ax.set_title(title)
    ax.set_xlabel("Rolling fold")
    ax.set_ylabel("Absolute standardized mean difference")
    ax.set_xticks(sorted(subset["fold_id"].unique()))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save_figure(fig, path)


def run_temporal_validation(
    *,
    config_path: Path,
    feature_path: Path,
    feature_columns_path: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Path]:
    """Run rolling temporal validation and write tables, figures, summary."""
    logger = configure_logger(log_path)
    try:
        config = load_baseline_config(config_path)
        feature_table = pd.read_parquet(feature_path, engine="pyarrow")
        feature_columns = load_feature_columns(feature_columns_path)
        snapshot_time = load_snapshot_time(DEFAULT_RAW_MANIFEST_PATH, feature_table)
        feature_table = add_age_normalized_features(feature_table, snapshot_time)
        contest_times = build_contest_time_table(feature_table)
        folds = build_rolling_folds(contest_times)
        feature_groups = build_feature_groups(feature_table, feature_columns)
        feature_sets = build_feature_sets(feature_groups)
        metrics = evaluate_rolling_windows(
            feature_table,
            folds,
            feature_sets,
            seed=config.random_seed,
        )
        drift_summary = build_temporal_drift_summary(feature_table, folds)
        summary = build_temporal_validation_summary(
            feature_table,
            folds,
            metrics,
            snapshot_time,
        )

        output_dir = output_dir.resolve()
        summary_dir = output_dir / "summary"
        tables_dir = output_dir / "tables"
        figures_dir = output_dir / "figures"
        for directory in (summary_dir, tables_dir, figures_dir):
            directory.mkdir(parents=True, exist_ok=True)

        paths = {
            "temporal_validation_summary": (
                summary_dir / "temporal_validation_summary.json"
            ),
            "rolling_window_metrics": tables_dir / "rolling_window_metrics.csv",
            "temporal_drift_summary": tables_dir / "temporal_drift_summary.csv",
            "rolling_window_mae": figures_dir / "rolling_window_mae.png",
            "rolling_window_within_200": (
                figures_dir / "rolling_window_within_200.png"
            ),
            "temporal_drift_summary_figure": (
                figures_dir / "temporal_drift_summary.png"
            ),
        }
        write_json(paths["temporal_validation_summary"], summary)
        metrics.to_csv(paths["rolling_window_metrics"], index=False)
        drift_summary.to_csv(paths["temporal_drift_summary"], index=False)
        plot_rolling_metric(metrics, "MAE", paths["rolling_window_mae"])
        plot_rolling_metric(
            metrics,
            "within_200",
            paths["rolling_window_within_200"],
        )
        plot_temporal_drift_summary(
            drift_summary,
            paths["temporal_drift_summary_figure"],
        )

        logger.info(
            "Completed Codeforces rolling temporal validation",
            extra={
                "event": "temporal_validation_completed",
                "details": {
                    "output_dir": output_dir.as_posix(),
                    "random_seed": config.random_seed,
                    "snapshot_time_utc": snapshot_time.isoformat(),
                    "row_count": len(feature_table),
                    "fold_count": len(folds),
                },
            },
        )
        return paths
    except Exception:
        logger.exception(
            "Codeforces rolling temporal validation failed",
            extra={"event": "temporal_validation_failed"},
        )
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the temporal-validation command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run Codeforces rolling-window temporal validation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument(
        "--feature-columns-path",
        type=Path,
        default=DEFAULT_FEATURE_COLUMNS_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the temporal-validation CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        paths = run_temporal_validation(
            config_path=args.config,
            feature_path=args.feature_path,
            feature_columns_path=args.feature_columns_path,
            output_dir=args.output_dir,
            log_path=args.log_path,
        )
    except (TemporalValidationError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote temporal validation summary: {paths['temporal_validation_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
