"""v6 TF-IDF semantic statement text cold-start experiments.

This module evaluates classical bag-of-words statement text features on top of
the existing processed Codeforces dataset. It reads local artifacts only and
does not modify v5 outputs or save trained model files. The full_api_reference
setting in this module uses the same ridge-based comparison setup as the TF-IDF
experiment and should not replace the historical v5 full API benchmark.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cf_diff import RANDOM_SEED
from cf_diff.model_selection import (
    DEFAULT_METRIC_COLUMNS,
    build_validation_ranked_report,
    select_rank_one,
)
from cf_diff.statement_cold_start import has_solved_leakage
from cf_diff.statement_features import STATEMENT_FEATURE_COLUMNS


DEFAULT_FEATURE_PATH: Final[Path] = Path("data/processed/features/model_table.parquet")
DEFAULT_STATEMENT_FEATURE_PATH: Final[Path] = Path(
    "data/processed/statement_features/statement_features.parquet"
)
DEFAULT_STATEMENT_TEXT_PATH: Final[Path] = Path(
    "data/processed/statement_text/statement_text.parquet"
)
DEFAULT_CONTEST_SPLIT_PATH: Final[Path] = Path(
    "data/processed/splits/contest_grouped_split.parquet"
)
DEFAULT_TIME_SPLIT_PATH: Final[Path] = Path(
    "data/processed/splits/forward_time_split.parquet"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path("outputs/semantic_tfidf")
DEFAULT_LOG_PATH: Final[Path] = Path("outputs/logs/semantic_tfidf.log")

TARGET_COLUMN: Final[str] = "rating"
TEXT_COLUMN_DEFAULT: Final[str] = "combined_text"
IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (
    "contest_id",
    "index",
    "name",
    "start_time_seconds",
)
SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "valid", "test")
STRATEGIES: Final[tuple[str, str]] = ("contest_grouped", "forward_time")
FEATURE_SETTINGS: Final[tuple[str, ...]] = (
    "metadata_only",
    "text_light_only",
    "tfidf_text_only",
    "metadata_plus_text_light",
    "metadata_plus_tfidf",
    "metadata_plus_text_light_plus_tfidf",
    "full_api_reference",
)
TFIDF_SETTINGS: Final[set[str]] = {
    "tfidf_text_only",
    "metadata_plus_tfidf",
    "metadata_plus_text_light_plus_tfidf",
}
TEXT_STATUS_COLUMNS: Final[set[str]] = {
    "text_extract_status",
    "text_extract_error",
    "html_cache_found",
    "statement_text_available",
    "title_text",
    "statement_text",
    "input_text",
    "output_text",
    "note_text",
    "examples_text",
    "combined_text",
}
STATEMENT_STATUS_COLUMNS: Final[set[str]] = {
    "url",
    "statement_fetch_status",
    "statement_parse_status",
    "statement_error",
}
TFIDF_DEFAULTS: Final[dict[str, object]] = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_df": 0.85,
    "max_features": 20000,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}
CONSERVATIVE_NOTES: Final[tuple[str, ...]] = (
    "TF-IDF is a classical bag-of-words semantic baseline, not deep language understanding.",
    "Statement text extraction is approximate.",
    "Cold-start settings exclude solved-count behavior.",
    "Tags may still be post-contest metadata, so this is metadata/statement cold-start rather than strict pre-contest prediction.",
    "The full_api_reference setting in this module uses the same ridge-based comparison setup as the TF-IDF experiment and should not replace the historical v5 full API benchmark.",
    "This module does not modify v5 results.",
)


class SemanticTfidfError(RuntimeError):
    """Raised when semantic TF-IDF experiments cannot proceed safely."""


class JsonLogFormatter(logging.Formatter):
    """Format semantic TF-IDF logs as JSON Lines."""

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
class TfidfConfig:
    """TF-IDF vectorizer configuration."""

    lowercase: bool = True
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 3
    max_df: float = 0.85
    max_features: int = 20000
    sublinear_tf: bool = True
    strip_accents: str = "unicode"


def configure_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """Create the dedicated structured semantic TF-IDF logger."""

    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.semantic_tfidf")
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
    """Flush and close all handlers attached to the semantic TF-IDF logger."""

    for handler in tuple(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _normalize_key(value: object) -> str:
    """Normalize contest/index keys for robust joins."""

    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], table_name: str) -> None:
    """Raise a clear error when required columns are absent."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SemanticTfidfError(f"{table_name} lacks columns: {missing}")


def _dedupe_preserve_order(columns: Sequence[str]) -> list[str]:
    """Return unique columns while preserving first appearance."""

    seen: set[str] = set()
    output: list[str] = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            output.append(column)
    return output


def load_model_feature_columns(feature_path: Path) -> list[str]:
    """Load sibling ``feature_columns.json`` if present."""

    path = feature_path.parent / "feature_columns.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SemanticTfidfError("feature_columns.json must contain a JSON object.")
    columns = payload.get("feature_columns", [])
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        raise SemanticTfidfError("Invalid feature_columns.json schema.")
    return list(columns)


def _with_join_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized join keys."""

    output = frame.copy()
    _require_columns(output, ("contest_id", "index"), "table")
    output["_contest_key"] = output["contest_id"].map(_normalize_key)
    output["_index_key"] = output["index"].map(_normalize_key)
    return output


def _join_auxiliary_table(
    base: pd.DataFrame,
    auxiliary: pd.DataFrame,
    *,
    columns: Sequence[str],
    suffix: str,
) -> tuple[pd.DataFrame, int]:
    """Left-join selected auxiliary columns by contest/index."""

    base_keys = _with_join_keys(base)
    aux = _with_join_keys(auxiliary)
    selected = [
        column
        for column in columns
        if column in aux.columns and column not in {"contest_id", "index"}
    ]
    aux = aux.loc[:, ["_contest_key", "_index_key", *selected]].drop_duplicates(
        ["_contest_key", "_index_key"],
        keep="first",
    )
    matched_keys = base_keys.loc[:, ["_contest_key", "_index_key"]].merge(
        aux.loc[:, ["_contest_key", "_index_key"]],
        on=["_contest_key", "_index_key"],
        how="inner",
    )
    joined = base_keys.merge(
        aux,
        on=["_contest_key", "_index_key"],
        how="left",
        validate="one_to_one",
        suffixes=("", suffix),
    ).drop(columns=["_contest_key", "_index_key"])
    return joined, int(len(matched_keys))


def load_experiment_table(
    *,
    feature_path: Path,
    statement_feature_path: Path,
    statement_text_path: Path,
    text_column: str,
) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    """Load and join model features, text-light features, and statement text."""

    if not feature_path.exists():
        raise SemanticTfidfError(f"Feature table does not exist: {feature_path}")
    if not statement_feature_path.exists():
        raise SemanticTfidfError(
            f"Statement feature table does not exist: {statement_feature_path}"
        )
    if not statement_text_path.exists():
        raise SemanticTfidfError(f"Statement text table does not exist: {statement_text_path}")

    model_table = pd.read_parquet(feature_path, engine="pyarrow")
    _require_columns(model_table, ("contest_id", "index", TARGET_COLUMN), "model table")
    model_feature_columns = load_model_feature_columns(feature_path)

    statement_features = pd.read_parquet(statement_feature_path, engine="pyarrow")
    statement_feature_columns = [
        column
        for column in (
            "contest_id",
            "index",
            *STATEMENT_STATUS_COLUMNS,
            *STATEMENT_FEATURE_COLUMNS,
        )
        if column in statement_features.columns
    ]
    table, matched_statement_feature_rows = _join_auxiliary_table(
        model_table,
        statement_features,
        columns=statement_feature_columns,
        suffix="_statement_feature",
    )

    statement_text = pd.read_parquet(statement_text_path, engine="pyarrow")
    if text_column not in statement_text.columns:
        raise SemanticTfidfError(f"Statement text table lacks text column: {text_column}")
    statement_text_columns = [
        column
        for column in ("contest_id", "index", *TEXT_STATUS_COLUMNS)
        if column in statement_text.columns
    ]
    table, matched_statement_text_rows = _join_auxiliary_table(
        table,
        statement_text,
        columns=statement_text_columns,
        suffix="_statement_text",
    )
    table = prepare_text_column(table, text_column)
    counts = {
        "input_model_table_rows": int(len(model_table)),
        "matched_statement_feature_rows": matched_statement_feature_rows,
        "matched_statement_text_rows": matched_statement_text_rows,
    }
    return table, counts, model_feature_columns


def prepare_text_column(frame: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Fill missing statement text and normalize text availability flags."""

    output = frame.copy()
    if text_column not in output.columns:
        output[text_column] = ""
    output[text_column] = output[text_column].fillna("").astype(str)
    if "statement_text_available" in output.columns:
        output["statement_text_available"] = (
            output["statement_text_available"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
    else:
        output["statement_text_available"] = output[text_column].str.strip().ne("")
    return output


def text_availability_summary(frame: pd.DataFrame, text_column: str) -> dict[str, object]:
    """Return text availability counts for summary reporting."""

    prepared = prepare_text_column(frame, text_column)
    available = prepared[text_column].str.strip().ne("")
    return {
        "text_available_count": int(available.sum()),
        "text_available_rate": round(float(available.mean()) if len(available) else 0.0, 6),
    }


def _candidate_model_columns(
    frame: pd.DataFrame,
    model_feature_columns: Sequence[str],
) -> list[str]:
    """Return original model-table feature candidates."""

    if model_feature_columns:
        return [column for column in model_feature_columns if column in frame.columns]
    excluded = {
        TARGET_COLUMN,
        *IDENTIFIER_COLUMNS,
        *TEXT_STATUS_COLUMNS,
        *STATEMENT_STATUS_COLUMNS,
        *STATEMENT_FEATURE_COLUMNS,
    }
    return [
        column
        for column in frame.columns
        if column not in excluded
        and not column.endswith("_statement_feature")
        and not column.endswith("_statement_text")
        and not column.startswith("_")
    ]


def _is_list_like_cell(value: object) -> bool:
    """Return whether a cell is list-like and unsuitable for sklearn features."""

    return isinstance(value, (list, tuple, set, dict, np.ndarray))


def _is_modelable_column(frame: pd.DataFrame, column: str) -> bool:
    """Return whether a selected column is a safe scalar feature."""

    if column not in frame.columns:
        return False
    non_null = frame[column].dropna()
    if non_null.empty:
        return True
    return not non_null.map(_is_list_like_cell).any()


def select_metadata_features(
    frame: pd.DataFrame,
    model_feature_columns: Sequence[str],
) -> list[str]:
    """Select cold-start metadata columns with solved-behavior leakage removed."""

    columns = [
        column
        for column in _candidate_model_columns(frame, model_feature_columns)
        if not has_solved_leakage(column) and _is_modelable_column(frame, column)
    ]
    if not columns:
        raise SemanticTfidfError("No metadata-only features were selected.")
    return _dedupe_preserve_order(columns)


def select_text_light_features(frame: pd.DataFrame) -> list[str]:
    """Select numeric statement text-light features."""

    columns: list[str] = []
    for column in STATEMENT_FEATURE_COLUMNS:
        if column not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any() and not has_solved_leakage(column):
            columns.append(column)
    if not columns:
        raise SemanticTfidfError("No statement text-light features were selected.")
    return sorted(_dedupe_preserve_order(columns))


def select_full_api_reference_features(
    frame: pd.DataFrame,
    model_feature_columns: Sequence[str],
) -> list[str]:
    """Select original API features, allowing solved-count behavior."""

    columns = [
        column
        for column in _candidate_model_columns(frame, model_feature_columns)
        if _is_modelable_column(frame, column)
    ]
    if not columns:
        raise SemanticTfidfError("No full API reference features were selected.")
    return _dedupe_preserve_order(columns)


def build_feature_sets(
    frame: pd.DataFrame,
    model_feature_columns: Sequence[str],
) -> dict[str, list[str]]:
    """Build explicit feature columns for all v6 settings."""

    metadata = select_metadata_features(frame, model_feature_columns)
    text_light = select_text_light_features(frame)
    full_api = select_full_api_reference_features(frame, model_feature_columns)
    feature_sets = {
        "metadata_only": metadata,
        "text_light_only": text_light,
        "tfidf_text_only": [],
        "metadata_plus_text_light": _dedupe_preserve_order([*metadata, *text_light]),
        "metadata_plus_tfidf": metadata,
        "metadata_plus_text_light_plus_tfidf": _dedupe_preserve_order(
            [*metadata, *text_light]
        ),
        "full_api_reference": full_api,
    }
    for setting, columns in feature_sets.items():
        if setting != "full_api_reference":
            leakage = [column for column in columns if has_solved_leakage(column)]
            if leakage:
                raise SemanticTfidfError(
                    f"Cold-start setting {setting} selected leakage columns: {leakage}"
                )
    return feature_sets


def load_split_assignment(path: Path, strategy: str) -> pd.DataFrame:
    """Load one saved split assignment."""

    if not path.exists():
        raise SemanticTfidfError(f"Split file does not exist: {path}")
    split = pd.read_parquet(path, engine="pyarrow")
    _require_columns(split, ("contest_id", "index", "split_name"), "split assignment")
    if "strategy" in split.columns:
        split = split.loc[split["strategy"].eq(strategy)].copy()
    missing = [name for name in SPLIT_NAMES if not split["split_name"].eq(name).any()]
    if missing:
        raise SemanticTfidfError(f"Split {strategy} has empty partitions: {missing}")
    return split


def join_split_assignments(frame: pd.DataFrame, split_assignment: pd.DataFrame) -> pd.DataFrame:
    """Join row-level split assignments to experiment rows."""

    base = _with_join_keys(frame)
    split = _with_join_keys(split_assignment)
    columns = [
        column
        for column in ("_contest_key", "_index_key", "split_name", "fold", "strategy")
        if column in split.columns
    ]
    split = split.loc[:, columns].drop_duplicates(["_contest_key", "_index_key"])
    joined = base.merge(
        split,
        on=["_contest_key", "_index_key"],
        how="inner",
        validate="one_to_one",
    ).drop(columns=["_contest_key", "_index_key"])
    if len(joined) != len(frame):
        raise SemanticTfidfError(
            f"Split assignment matched {len(joined)} of {len(frame)} rows."
        )
    return joined


def compute_regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict[str, float]:
    """Compute deterministic regression metrics."""

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise SemanticTfidfError(
            f"Metric arrays must have matching shapes; got {actual.shape} and {predicted.shape}."
        )
    if actual.size == 0:
        raise SemanticTfidfError("Cannot compute metrics for an empty split.")
    errors = actual - predicted
    absolute_errors = np.abs(errors)
    r2 = float(r2_score(actual, predicted)) if actual.size >= 2 else float("nan")
    return {
        "MAE": round(float(np.mean(absolute_errors)), 6),
        "RMSE": round(float(np.sqrt(np.mean(np.square(errors)))), 6),
        "R2": round(r2, 6) if math.isfinite(r2) else r2,
        "within_100": round(float(np.mean(absolute_errors <= 100.0)), 6),
        "within_200": round(float(np.mean(absolute_errors <= 200.0)), 6),
    }


def _categorical_columns(frame: pd.DataFrame) -> list[str]:
    """Identify categorical scalar predictors."""

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
    """Identify numeric predictors."""

    categorical = set(_categorical_columns(frame))
    return [column for column in frame.columns if column not in categorical]


def build_tfidf_vectorizer(config: TfidfConfig) -> TfidfVectorizer:
    """Build the configured TF-IDF vectorizer."""

    return TfidfVectorizer(
        lowercase=config.lowercase,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.max_features,
        sublinear_tf=config.sublinear_tf,
        strip_accents=config.strip_accents,
    )


def build_semantic_pipeline(
    sample: pd.DataFrame,
    *,
    text_column: str,
    use_tfidf: bool,
    tfidf_config: TfidfConfig,
) -> Pipeline:
    """Build a Ridge pipeline for dense and optional sparse TF-IDF features."""

    transformers: list[tuple[str, object, object]] = []
    if use_tfidf:
        transformers.append(("tfidf", build_tfidf_vectorizer(tfidf_config), text_column))
    structured_columns = [column for column in sample.columns if column != text_column]
    structured = sample.loc[:, structured_columns] if structured_columns else pd.DataFrame()
    numeric_columns = _numeric_columns(structured) if not structured.empty else []
    categorical_columns = _categorical_columns(structured) if not structured.empty else []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
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
                            SimpleImputer(strategy="constant", fill_value="__missing__"),
                        ),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise SemanticTfidfError("No model inputs were provided to the pipeline.")
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    transformers=transformers,
                    remainder="drop",
                    sparse_threshold=1.0,
                    verbose_feature_names_out=False,
                ),
            ),
            ("model", Ridge(alpha=1.0, solver="lsqr")),
        ]
    )


def _model_input_columns(
    feature_columns: Sequence[str],
    *,
    use_tfidf: bool,
    text_column: str,
) -> list[str]:
    """Return ordered model input columns."""

    columns = list(feature_columns)
    if use_tfidf:
        columns = [text_column, *columns]
    return _dedupe_preserve_order(columns)


def fit_predict_setting(
    joined: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    setting: str,
    text_column: str,
    tfidf_config: TfidfConfig,
) -> pd.Series:
    """Fit on train rows and predict all rows for one feature setting."""

    use_tfidf = setting in TFIDF_SETTINGS
    input_columns = _model_input_columns(
        feature_columns,
        use_tfidf=use_tfidf,
        text_column=text_column,
    )
    train = joined.loc[joined["split_name"].eq("train")].copy()
    if train.empty:
        raise SemanticTfidfError("Train split is empty.")
    train_x = train.loc[:, input_columns]
    train_y = pd.to_numeric(train[TARGET_COLUMN], errors="raise").to_numpy(float)
    model = build_semantic_pipeline(
        train_x,
        text_column=text_column,
        use_tfidf=use_tfidf,
        tfidf_config=tfidf_config,
    )
    model.fit(train_x, train_y)
    predictions = model.predict(joined.loc[:, input_columns])
    return pd.Series(predictions, index=joined.index, dtype=float)


def evaluate_strategy(
    experiment_table: pd.DataFrame,
    split_assignment: pd.DataFrame,
    *,
    strategy: str,
    feature_sets: Mapping[str, Sequence[str]],
    text_column: str,
    tfidf_config: TfidfConfig,
    prediction_dir: Path | None = None,
) -> pd.DataFrame:
    """Evaluate every v6 feature setting for one split strategy."""

    joined = join_split_assignments(experiment_table, split_assignment)
    rows: list[dict[str, object]] = []
    for setting in FEATURE_SETTINGS:
        feature_columns = list(feature_sets[setting])
        predictions = fit_predict_setting(
            joined,
            feature_columns=feature_columns,
            setting=setting,
            text_column=text_column,
            tfidf_config=tfidf_config,
        )
        for split_name in ("train", "valid", "test"):
            split_mask = joined["split_name"].eq(split_name)
            metrics = compute_regression_metrics(
                joined.loc[split_mask, TARGET_COLUMN],
                predictions.loc[split_mask],
            )
            rows.append(
                {
                    "strategy": strategy,
                    "split_name": split_name,
                    "feature_setting": setting,
                    "model_name": "ridge_regression",
                    "feature_count": int(len(feature_columns)),
                    "uses_tfidf": bool(setting in TFIDF_SETTINGS),
                    "row_count": int(split_mask.sum()),
                    **metrics,
                }
            )
        if prediction_dir is not None:
            write_predictions(
                joined,
                predictions,
                prediction_dir / f"{strategy}_{setting}_predictions.csv",
                strategy=strategy,
                setting=setting,
            )
    return pd.DataFrame(rows)


def write_predictions(
    joined: pd.DataFrame,
    predictions: pd.Series,
    path: Path,
    *,
    strategy: str,
    setting: str,
) -> None:
    """Write row-level predictions for one strategy/setting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(
        {
            "strategy": strategy,
            "feature_setting": setting,
            "contest_id": joined["contest_id"],
            "index": joined["index"],
            "name": joined["name"] if "name" in joined.columns else "",
            "split_name": joined["split_name"],
            "actual_rating": joined[TARGET_COLUMN],
            "predicted_rating": predictions,
        }
    )
    output["error"] = output["actual_rating"] - output["predicted_rating"]
    output["abs_error"] = output["error"].abs()
    output.sort_values(
        ["split_name", "contest_id", "index"],
        kind="mergesort",
    ).to_csv(path, index=False)


def best_by_strategy(metrics: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Select a feature setting on validation and report its test metrics."""
    ranked = build_validation_ranked_report(
        metrics,
        group_columns=("strategy",),
        candidate_columns=("feature_setting", "model_name"),
        metric_columns=DEFAULT_METRIC_COLUMNS,
    )
    selected = select_rank_one(ranked)
    result: dict[str, dict[str, object]] = {}
    for strategy, group in selected.groupby("strategy", sort=True):
        best = group.iloc[0]
        result[str(strategy)] = {
            "feature_setting": str(best["feature_setting"]),
            "model_name": str(best["model_name"]),
            "selection_split": "valid",
            "validation_MAE": float(best["validation_MAE"]),
            "report_split": "test",
            "MAE": float(best["MAE"]),
            "RMSE": float(best["RMSE"]),
            "within_200": float(best["within_200"]),
        }
    return result


def build_best_by_setting_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach validation evidence to each pre-specified setting's test row."""
    ranked = build_validation_ranked_report(
        metrics,
        group_columns=("strategy", "feature_setting"),
        candidate_columns=("model_name",),
        metric_columns=DEFAULT_METRIC_COLUMNS,
    )
    output = select_rank_one(ranked).sort_values(
        ["strategy", "feature_setting"],
        kind="mergesort",
    )
    return output.reset_index(drop=True)


def improvement_table(
    metrics: pd.DataFrame,
    *,
    baseline_setting: str,
    comparison_setting: str,
) -> list[dict[str, object]]:
    """Compute MAE improvements for one comparison across strategies."""

    rows: list[dict[str, object]] = []
    for strategy, group in metrics.groupby("strategy", sort=True):
        by_setting = group.set_index("feature_setting")
        if baseline_setting not in by_setting.index or comparison_setting not in by_setting.index:
            continue
        baseline_mae = float(by_setting.loc[baseline_setting, "MAE"])
        comparison_mae = float(by_setting.loc[comparison_setting, "MAE"])
        absolute = baseline_mae - comparison_mae
        rows.append(
            {
                "strategy": strategy,
                "baseline_setting": baseline_setting,
                "comparison_setting": comparison_setting,
                "baseline_MAE": round(baseline_mae, 6),
                "comparison_MAE": round(comparison_mae, 6),
                "absolute_MAE_improvement": round(absolute, 6),
                "percent_MAE_improvement": round(
                    absolute / baseline_mae * 100.0 if baseline_mae else 0.0,
                    6,
                ),
            }
        )
    return rows


def build_summary(
    *,
    experiment_table: pd.DataFrame,
    join_counts: Mapping[str, int],
    metrics: pd.DataFrame,
    tfidf_config: TfidfConfig,
    text_column: str,
) -> dict[str, object]:
    """Build the machine-readable experiment summary."""

    availability = text_availability_summary(experiment_table, text_column)
    locked_test_report = build_best_by_setting_table(metrics)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_model_table_rows": int(join_counts["input_model_table_rows"]),
        "matched_statement_text_rows": int(join_counts["matched_statement_text_rows"]),
        "matched_statement_feature_rows": int(
            join_counts["matched_statement_feature_rows"]
        ),
        **availability,
        "tfidf_max_features": int(tfidf_config.max_features),
        "tfidf_ngram_range": list(tfidf_config.ngram_range),
        "strategies": list(STRATEGIES),
        "settings": list(FEATURE_SETTINGS),
        "validation_selected_setting_test_report": best_by_strategy(metrics),
        "improvement_of_metadata_plus_tfidf_over_metadata_only": improvement_table(
            locked_test_report,
            baseline_setting="metadata_only",
            comparison_setting="metadata_plus_tfidf",
        ),
        "improvement_of_metadata_plus_text_light_plus_tfidf_over_metadata_plus_text_light": improvement_table(
            locked_test_report,
            baseline_setting="metadata_plus_text_light",
            comparison_setting="metadata_plus_text_light_plus_tfidf",
        ),
        "conservative_notes": list(CONSERVATIVE_NOTES),
    }


def write_json(path: Path, payload: object) -> None:
    """Write pretty UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def plot_mae_by_setting(metrics: pd.DataFrame, path: Path) -> None:
    """Create a bar chart comparing test MAE by setting and strategy."""

    path.parent.mkdir(parents=True, exist_ok=True)
    test_metrics = metrics.loc[metrics["split_name"].eq("test")]
    pivot = test_metrics.pivot(
        index="feature_setting",
        columns="strategy",
        values="MAE",
    ).loc[list(FEATURE_SETTINGS)]
    ax = pivot.plot(kind="bar", figsize=(13, 6))
    ax.set_title("Semantic TF-IDF cold-start test MAE by setting")
    ax.set_xlabel("Feature setting")
    ax.set_ylabel("Test MAE")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Split strategy")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_semantic_tfidf_experiment(
    *,
    feature_path: Path,
    statement_feature_path: Path,
    statement_text_path: Path,
    contest_split_path: Path,
    time_split_path: Path,
    output_dir: Path,
    log_path: Path,
    text_column: str = TEXT_COLUMN_DEFAULT,
    tfidf_config: TfidfConfig | None = None,
) -> dict[str, Path]:
    """Run the full v6 semantic TF-IDF experiment."""

    config = tfidf_config or TfidfConfig()
    logger = configure_logger(log_path)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_dir = output_dir / "summary"
        table_dir = output_dir / "tables"
        prediction_dir = output_dir / "predictions"
        figure_dir = output_dir / "figures"
        for directory in (summary_dir, table_dir, prediction_dir, figure_dir):
            directory.mkdir(parents=True, exist_ok=True)

        experiment_table, join_counts, model_feature_columns = load_experiment_table(
            feature_path=feature_path,
            statement_feature_path=statement_feature_path,
            statement_text_path=statement_text_path,
            text_column=text_column,
        )
        feature_sets = build_feature_sets(experiment_table, model_feature_columns)
        contest_split = load_split_assignment(contest_split_path, "contest_grouped")
        time_split = load_split_assignment(time_split_path, "forward_time")
        metrics = pd.concat(
            [
                evaluate_strategy(
                    experiment_table,
                    contest_split,
                    strategy="contest_grouped",
                    feature_sets=feature_sets,
                    text_column=text_column,
                    tfidf_config=config,
                    prediction_dir=prediction_dir,
                ),
                evaluate_strategy(
                    experiment_table,
                    time_split,
                    strategy="forward_time",
                    feature_sets=feature_sets,
                    text_column=text_column,
                    tfidf_config=config,
                    prediction_dir=prediction_dir,
                ),
            ],
            ignore_index=True,
        ).sort_values(["strategy", "feature_setting"], kind="mergesort")
        best_table = build_best_by_setting_table(metrics)
        summary = build_summary(
            experiment_table=experiment_table,
            join_counts=join_counts,
            metrics=metrics,
            tfidf_config=config,
            text_column=text_column,
        )

        paths = {
            "summary": summary_dir / "semantic_tfidf_summary.json",
            "metrics": table_dir / "semantic_tfidf_metrics.csv",
            "best_by_setting": table_dir / "semantic_tfidf_best_by_setting.csv",
            "figure": figure_dir / "semantic_tfidf_mae_by_setting.png",
        }
        write_json(paths["summary"], summary)
        metrics.to_csv(paths["metrics"], index=False)
        best_table.to_csv(paths["best_by_setting"], index=False)
        plot_mae_by_setting(metrics, paths["figure"])
        logger.info(
            "Completed semantic TF-IDF experiment",
            extra={
                "event": "semantic_tfidf_completed",
                "details": {
                    "input_rows": len(experiment_table),
                    "metrics_rows": len(metrics),
                    "output_dir": output_dir.as_posix(),
                },
            },
        )
        return paths
    except Exception:
        logger.exception(
            "Semantic TF-IDF experiment failed",
            extra={"event": "semantic_tfidf_failed"},
        )
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run v6 semantic TF-IDF statement text cold-start experiments."
    )
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument(
        "--statement-feature-path",
        type=Path,
        default=DEFAULT_STATEMENT_FEATURE_PATH,
    )
    parser.add_argument(
        "--statement-text-path",
        type=Path,
        default=DEFAULT_STATEMENT_TEXT_PATH,
    )
    parser.add_argument("--contest-split-path", type=Path, default=DEFAULT_CONTEST_SPLIT_PATH)
    parser.add_argument("--time-split-path", type=Path, default=DEFAULT_TIME_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--text-column", type=str, default=TEXT_COLUMN_DEFAULT)
    parser.add_argument("--tfidf-max-features", type=int, default=20000)
    parser.add_argument("--tfidf-min-df", type=int, default=3)
    parser.add_argument("--tfidf-max-df", type=float, default=0.85)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the semantic TF-IDF CLI."""

    args = _build_argument_parser().parse_args(argv)
    try:
        config = TfidfConfig(
            min_df=args.tfidf_min_df,
            max_df=args.tfidf_max_df,
            max_features=args.tfidf_max_features,
        )
        paths = run_semantic_tfidf_experiment(
            feature_path=args.feature_path,
            statement_feature_path=args.statement_feature_path,
            statement_text_path=args.statement_text_path,
            contest_split_path=args.contest_split_path,
            time_split_path=args.time_split_path,
            output_dir=args.output_dir,
            log_path=args.log_path,
            text_column=args.text_column,
            tfidf_config=config,
        )
    except (SemanticTfidfError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote semantic TF-IDF metrics: {paths['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
