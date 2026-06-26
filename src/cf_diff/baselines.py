"""Train and evaluate baseline regressors for Codeforces rating prediction."""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cf_diff import RANDOM_SEED
from cf_diff.features import parse_simple_yaml, write_json

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
DEFAULT_OUTPUT_DIR: Final[Path] = Path("outputs/baselines")
DEFAULT_LOG_PATH: Final[Path] = Path("outputs/logs/baselines.log")
TARGET_COLUMN: Final[str] = "rating"
ACTUAL_COLUMN: Final[str] = "actual_rating"
PREDICTED_COLUMN: Final[str] = "predicted_rating"
IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (
    "contest_id",
    "index",
    "name",
    "start_time_seconds",
)
SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "valid", "test")
INDEX_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "index_letter",
    "index_number",
    "index_rank",
)
SOLVED_COUNT_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "solved_count",
    "log_solved_count",
    "solved_count_missing",
)


class BaselineError(RuntimeError):
    """Raised when baseline training cannot proceed safely."""


class JsonLogFormatter(logging.Formatter):
    """Format model training logs as JSON Lines."""

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
class BaselineConfig:
    """Store reproducibility settings for the baseline layer."""

    random_seed: int = RANDOM_SEED


@dataclass(frozen=True)
class ModelSpec:
    """Describe one model experiment."""

    name: str
    feature_selector: Callable[[pd.DataFrame, Sequence[str]], list[str]]
    estimator_factory: Callable[[int], object]


def configure_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """Create the dedicated structured baseline logger."""
    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.baselines")
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


def _mapping(value: object, key: str) -> Mapping[str, object]:
    """Require a mapping-valued config section."""
    if not isinstance(value, dict):
        raise BaselineError(f"Config key {key!r} must be a mapping.")
    return value


def load_baseline_config(path: Path = DEFAULT_CONFIG_PATH) -> BaselineConfig:
    """Load the random seed from JSON or the project YAML subset."""
    if not path.exists():
        return BaselineConfig()
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise BaselineError("Config must contain a JSON object.")
    else:
        parsed = parse_simple_yaml(text)

    seed = parsed.get("random_seed")
    project = parsed.get("project")
    if seed is None and isinstance(project, dict):
        seed = project.get("random_seed")
    return BaselineConfig(
        random_seed=int(seed if seed is not None else RANDOM_SEED)
    )


def load_feature_columns(path: Path) -> list[str]:
    """Read feature column metadata written by the feature pipeline."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BaselineError("Feature column metadata must be a JSON object.")
    raw_columns = payload.get("feature_columns")
    if not isinstance(raw_columns, list) or not all(
        isinstance(column, str) for column in raw_columns
    ):
        raise BaselineError("feature_columns.json lacks string feature_columns.")
    return list(raw_columns)


def _available_columns(
    frame: pd.DataFrame,
    requested_columns: Sequence[str],
) -> list[str]:
    """Return requested columns that exist in the model table."""
    return [column for column in requested_columns if column in frame.columns]


def select_index_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> list[str]:
    """Select only index-derived features."""
    del feature_columns
    columns = _available_columns(frame, INDEX_FEATURE_COLUMNS)
    if not columns:
        raise BaselineError("No index-derived feature columns are available.")
    return columns


def select_tag_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> list[str]:
    """Select tag count and tag one-hot columns only."""
    del feature_columns
    columns = [
        column
        for column in frame.columns
        if column == "tag_count" or column.startswith("tag__")
    ]
    if not columns:
        raise BaselineError("No tag feature columns are available.")
    return sorted(columns, key=lambda column: (column != "tag_count", column))


def select_solved_count_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> list[str]:
    """Select solved-count-only features."""
    del feature_columns
    columns = _available_columns(frame, SOLVED_COUNT_FEATURE_COLUMNS)
    if not columns:
        raise BaselineError("No solved-count feature columns are available.")
    return columns


def select_full_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> list[str]:
    """Select all modeling features from metadata, excluding identifiers."""
    if feature_columns:
        columns = _available_columns(frame, feature_columns)
    else:
        excluded = {TARGET_COLUMN, *IDENTIFIER_COLUMNS}
        columns = [column for column in frame.columns if column not in excluded]
    if not columns:
        raise BaselineError("No full-model feature columns are available.")
    return columns


def select_no_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> list[str]:
    """Select no predictors for a mean-only baseline."""
    del frame, feature_columns
    return []


def compute_regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict[str, float]:
    """Compute deterministic baseline regression metrics."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise BaselineError(
            f"Metric arrays must have matching shapes; got "
            f"{actual.shape} and {predicted.shape}."
        )
    if actual.size == 0:
        raise BaselineError("Cannot compute metrics for an empty split.")
    errors = actual - predicted
    absolute_errors = np.abs(errors)
    mae = float(np.mean(absolute_errors))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    r2 = float(r2_score(actual, predicted)) if actual.size >= 2 else float("nan")
    return {
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "R2": round(r2, 6) if math.isfinite(r2) else r2,
        "within_100": round(float(np.mean(absolute_errors <= 100.0)), 6),
        "within_200": round(float(np.mean(absolute_errors <= 200.0)), 6),
    }


def _safe_name(value: str) -> str:
    """Create a filesystem-safe model artifact stem."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _constant_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a one-column matrix for predictors that ignore features."""
    return pd.DataFrame({"constant": np.ones(len(frame), dtype=float)})


def _coerce_feature_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a feature matrix with stable column order."""
    if not columns:
        return _constant_frame(frame)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise BaselineError(f"Model table lacks selected features: {missing}")
    return frame.loc[:, list(columns)].copy()


def _categorical_columns(frame: pd.DataFrame) -> list[str]:
    """Identify categorical predictors requiring one-hot encoding."""
    return [
        column
        for column in frame.columns
        if (
            pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
            or isinstance(frame[column].dtype, pd.CategoricalDtype)
        )
    ]


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Identify numeric predictors for imputation and scaling."""
    categorical = set(_categorical_columns(frame))
    return [column for column in frame.columns if column not in categorical]


def make_preprocessed_estimator(estimator: object, sample: pd.DataFrame) -> Pipeline:
    """Wrap an estimator with deterministic tabular preprocessing."""
    numeric_columns = _numeric_columns(sample)
    categorical_columns = _categorical_columns(sample)
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__missing__",
                            ),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        transformers.append(
            (
                "constant",
                "passthrough",
                list(sample.columns),
            )
        )
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    transformers=transformers,
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
            ("model", estimator),
        ]
    )


def _mean_estimator(seed: int) -> DummyRegressor:
    """Build the mean-only baseline estimator."""
    del seed
    return DummyRegressor(strategy="mean")


def _ridge_estimator(seed: int) -> Ridge:
    """Build a deterministic ridge regressor."""
    del seed
    return Ridge(alpha=1.0)


def _random_forest_estimator(seed: int) -> RandomForestRegressor:
    """Build a deterministic random forest regressor."""
    return RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )


def _hist_gradient_boosting_estimator(
    seed: int,
) -> HistGradientBoostingRegressor:
    """Build a deterministic histogram gradient boosting regressor."""
    return HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=200,
        l2_regularization=0.01,
        random_state=seed,
    )


def build_model_specs() -> list[ModelSpec]:
    """Return all required baseline experiment specifications."""
    return [
        ModelSpec("mean_baseline", select_no_features, _mean_estimator),
        ModelSpec("index_only_baseline", select_index_features, _ridge_estimator),
        ModelSpec("tag_only_baseline", select_tag_features, _ridge_estimator),
        ModelSpec(
            "solved_count_only_baseline",
            select_solved_count_features,
            _ridge_estimator,
        ),
        ModelSpec("ridge_regression", select_full_features, _ridge_estimator),
        ModelSpec(
            "random_forest_regressor",
            select_full_features,
            _random_forest_estimator,
        ),
        ModelSpec(
            "hist_gradient_boosting_regressor",
            select_full_features,
            _hist_gradient_boosting_estimator,
        ),
    ]


def _validate_model_table(frame: pd.DataFrame) -> None:
    """Validate identifiers and target availability in the feature table."""
    required = (*IDENTIFIER_COLUMNS, TARGET_COLUMN)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise BaselineError(f"Model table lacks required columns: {missing}")
    if frame[[TARGET_COLUMN, "contest_id", "index"]].isna().any().any():
        raise BaselineError("Model table has null rating or split identifiers.")
    duplicated = frame.duplicated(["contest_id", "index"], keep=False)
    if duplicated.any():
        raise BaselineError(
            f"Model table has {int(duplicated.sum())} duplicate identifiers."
        )


def join_split_assignments(
    model_table: pd.DataFrame,
    split_assignment: pd.DataFrame,
) -> pd.DataFrame:
    """Join model rows with one split-assignment table."""
    _validate_model_table(model_table)
    required_split_columns = ("contest_id", "index", "split_name")
    missing = [
        column
        for column in required_split_columns
        if column not in split_assignment.columns
    ]
    if missing:
        raise BaselineError(f"Split assignment lacks columns: {missing}")
    joined = model_table.merge(
        split_assignment.loc[:, list(required_split_columns)],
        on=["contest_id", "index"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(model_table):
        raise BaselineError(
            "Split assignment did not match every model-table row: "
            f"{len(joined)} of {len(model_table)} rows matched."
        )
    unknown_splits = set(joined["split_name"].dropna()) - set(SPLIT_NAMES)
    if unknown_splits:
        raise BaselineError(f"Unexpected split names: {sorted(unknown_splits)}")
    missing_splits = [
        split_name
        for split_name in SPLIT_NAMES
        if not joined["split_name"].eq(split_name).any()
    ]
    if missing_splits:
        raise BaselineError(f"Split assignment has empty splits: {missing_splits}")
    return joined.sort_values(
        ["split_name", "contest_id", "index"],
        kind="mergesort",
    ).reset_index(drop=True)


def _fit_model(
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    spec: ModelSpec,
    seed: int,
) -> object:
    """Fit one model specification on training rows."""
    estimator = spec.estimator_factory(seed)
    train_x = _coerce_feature_frame(train_frame, feature_columns)
    train_y = pd.to_numeric(train_frame[TARGET_COLUMN], errors="raise").to_numpy(
        dtype=float
    )
    if isinstance(estimator, DummyRegressor):
        model = estimator.fit(_constant_frame(train_frame), train_y)
    else:
        model = make_preprocessed_estimator(estimator, train_x)
        model.fit(train_x, train_y)
    return model


def _predict_model(
    model: object,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    """Predict ratings for rows using a fitted model."""
    if isinstance(model, DummyRegressor):
        return np.asarray(model.predict(_constant_frame(frame)), dtype=float)
    return np.asarray(
        model.predict(_coerce_feature_frame(frame, feature_columns)),
        dtype=float,
    )


def _save_model_artifact(
    model: object,
    path: Path,
    *,
    model_name: str,
    strategy: str,
    feature_columns: Sequence[str],
    seed: int,
) -> None:
    """Persist a fitted model and its minimal provenance metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "model_name": model_name,
        "strategy": strategy,
        "feature_columns": list(feature_columns),
        "target_column": TARGET_COLUMN,
        "random_seed": seed,
    }
    with path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate_models_for_strategy(
    model_table: pd.DataFrame,
    split_assignment: pd.DataFrame,
    *,
    strategy: str,
    feature_columns: Sequence[str],
    seed: int,
    model_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train all baseline models for one split strategy."""
    joined = join_split_assignments(model_table, split_assignment)
    train_frame = joined.loc[joined["split_name"].eq("train")].copy()
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for spec in build_model_specs():
        selected_columns = spec.feature_selector(joined, feature_columns)
        model = _fit_model(train_frame, selected_columns, spec, seed)
        predictions = _predict_model(model, joined, selected_columns)
        model_predictions = joined.loc[
            :,
            [*IDENTIFIER_COLUMNS, TARGET_COLUMN, "split_name"],
        ].copy()
        model_predictions = model_predictions.rename(
            columns={TARGET_COLUMN: ACTUAL_COLUMN}
        )
        model_predictions[PREDICTED_COLUMN] = predictions
        model_predictions["model_name"] = spec.name
        model_predictions = model_predictions.loc[
            :,
            [
                *IDENTIFIER_COLUMNS,
                ACTUAL_COLUMN,
                PREDICTED_COLUMN,
                "model_name",
                "split_name",
            ],
        ]
        prediction_frames.append(model_predictions)

        for split_name in SPLIT_NAMES:
            mask = model_predictions["split_name"].eq(split_name)
            split_metrics = compute_regression_metrics(
                model_predictions.loc[mask, ACTUAL_COLUMN],
                model_predictions.loc[mask, PREDICTED_COLUMN],
            )
            metrics_rows.append(
                {
                    "strategy": strategy,
                    "model_name": spec.name,
                    "split_name": split_name,
                    "row_count": int(mask.sum()),
                    "feature_count": len(selected_columns),
                    **split_metrics,
                }
            )

        _save_model_artifact(
            model,
            model_dir / strategy / f"{_safe_name(spec.name)}.pkl",
            model_name=spec.name,
            strategy=strategy,
            feature_columns=selected_columns,
            seed=seed,
        )

    metrics = pd.DataFrame(metrics_rows).sort_values(
        ["model_name", "split_name"],
        kind="mergesort",
    ).reset_index(drop=True)
    predictions_output = pd.concat(
        prediction_frames,
        ignore_index=True,
    ).sort_values(
        ["model_name", "split_name", "contest_id", "index"],
        kind="mergesort",
    ).reset_index(drop=True)
    return metrics, predictions_output


def _metrics_json_payload(strategy: str, metrics: pd.DataFrame) -> dict[str, object]:
    """Build machine-readable JSON metrics output."""
    return {
        "strategy": strategy,
        "metrics": metrics.to_dict(orient="records"),
    }


def run_baselines(
    *,
    config_path: Path,
    feature_path: Path,
    feature_columns_path: Path,
    contest_split_path: Path,
    time_split_path: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Path]:
    """Run all baseline experiments and write metrics, predictions, models."""
    logger = configure_logger(log_path)
    try:
        config = load_baseline_config(config_path)
        model_table = pd.read_parquet(feature_path, engine="pyarrow")
        feature_columns = load_feature_columns(feature_columns_path)
        contest_split = pd.read_parquet(contest_split_path, engine="pyarrow")
        time_split = pd.read_parquet(time_split_path, engine="pyarrow")

        output_dir = output_dir.resolve()
        metrics_dir = output_dir / "metrics"
        predictions_dir = output_dir / "predictions"
        tables_dir = output_dir / "tables"
        model_dir = output_dir / "models"
        for directory in (metrics_dir, predictions_dir, tables_dir, model_dir):
            directory.mkdir(parents=True, exist_ok=True)

        contest_metrics, contest_predictions = evaluate_models_for_strategy(
            model_table,
            contest_split,
            strategy="contest_grouped",
            feature_columns=feature_columns,
            seed=config.random_seed,
            model_dir=model_dir,
        )
        time_metrics, time_predictions = evaluate_models_for_strategy(
            model_table,
            time_split,
            strategy="forward_time",
            feature_columns=feature_columns,
            seed=config.random_seed,
            model_dir=model_dir,
        )

        paths = {
            "contest_grouped_metrics_csv": (
                metrics_dir / "contest_grouped_metrics.csv"
            ),
            "forward_time_metrics_csv": metrics_dir / "forward_time_metrics.csv",
            "contest_grouped_metrics_json": (
                metrics_dir / "contest_grouped_metrics.json"
            ),
            "forward_time_metrics_json": metrics_dir / "forward_time_metrics.json",
            "contest_grouped_predictions": (
                predictions_dir / "contest_grouped_predictions.parquet"
            ),
            "forward_time_predictions": (
                predictions_dir / "forward_time_predictions.parquet"
            ),
            "main_results_table": tables_dir / "main_results_table.csv",
        }

        contest_metrics.to_csv(paths["contest_grouped_metrics_csv"], index=False)
        time_metrics.to_csv(paths["forward_time_metrics_csv"], index=False)
        write_json(
            paths["contest_grouped_metrics_json"],
            _metrics_json_payload("contest_grouped", contest_metrics),
        )
        write_json(
            paths["forward_time_metrics_json"],
            _metrics_json_payload("forward_time", time_metrics),
        )
        contest_predictions.to_parquet(
            paths["contest_grouped_predictions"],
            engine="pyarrow",
            index=False,
        )
        time_predictions.to_parquet(
            paths["forward_time_predictions"],
            engine="pyarrow",
            index=False,
        )
        main_results = pd.concat(
            [contest_metrics, time_metrics],
            ignore_index=True,
        ).sort_values(
            ["strategy", "split_name", "model_name"],
            kind="mergesort",
        )
        main_results.to_csv(paths["main_results_table"], index=False)

        logger.info(
            "Completed Codeforces baseline training",
            extra={
                "event": "baselines_completed",
                "details": {
                    "random_seed": config.random_seed,
                    "model_count": len(build_model_specs()),
                    "rows": len(model_table),
                    "output_dir": output_dir.as_posix(),
                },
            },
        )
        return paths
    except Exception:
        logger.exception(
            "Codeforces baseline training failed",
            extra={"event": "baselines_failed"},
        )
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the baseline command-line parser."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate Codeforces baseline regressors."
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
    """Run the baseline training CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        paths = run_baselines(
            config_path=args.config,
            feature_path=args.feature_path,
            feature_columns_path=args.feature_columns_path,
            contest_split_path=args.contest_split_path,
            time_split_path=args.time_split_path,
            output_dir=args.output_dir,
            log_path=args.log_path,
        )
    except (BaselineError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote main results table: {paths['main_results_table']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
