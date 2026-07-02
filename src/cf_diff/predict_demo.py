"""Local v5.1 prediction demo for Codeforces difficulty.

This module is an application/demo layer on top of the existing research
pipeline. It trains lightweight models at runtime from the processed feature
table and does not save model artifacts or create new research results.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from cf_diff import RANDOM_SEED
from cf_diff.baselines import make_preprocessed_estimator
from cf_diff.statement_cold_start import (
    StatementColdStartError,
    has_solved_leakage,
    join_statement_features,
    select_full_api_reference_features,
    select_metadata_features,
)
from cf_diff.statement_features import STATEMENT_FEATURE_COLUMNS


TARGET_COLUMN: Final[str] = "rating"
STANDARD_CONTEST_COLUMN: Final[str] = "contest_id"
STANDARD_INDEX_COLUMN: Final[str] = "index"
STANDARD_NAME_COLUMN: Final[str] = "name"

SCENARIO_ORDER: Final[tuple[str, ...]] = (
    "post_publication_reference",
    "cold_start_metadata",
    "cold_start_metadata_plus_text_light",
)
SCENARIO_LABELS: Final[dict[str, str]] = {
    "post_publication_reference": "Post-publication reference",
    "cold_start_metadata": "Cold-start metadata",
    "cold_start_metadata_plus_text_light": "Cold-start metadata + text-light",
}
MODEL_CHOICES: Final[tuple[str, str, str]] = ("hgb", "rf", "ridge")
TAG_SYNONYMS: Final[dict[str, str]] = {
    "binary search": "binary_search",
    "binary searches": "binary_search",
    "shortest path": "shortest_paths",
    "shortest paths": "shortest_paths",
    "dynamic programming": "dp",
    "graph": "graphs",
    "graphs": "graphs",
    "string": "strings",
    "strings": "strings",
    "constructive algorithm": "constructive_algorithms",
    "constructive algorithms": "constructive_algorithms",
    "data structure": "data_structures",
    "data structures": "data_structures",
    "two pointers": "two_pointers",
    "number theory": "number_theory",
    "divide and conquer": "divide_and_conquer",
    "dfs and similar": "dfs_and_similar",
    "2 sat": "2_sat",
    "2-sat": "2_sat",
}
MANUAL_STATEMENT_ARGUMENTS: Final[dict[str, str]] = {
    "statement_char_len": "statement_char_len",
    "statement_word_count": "statement_word_count",
    "statement_line_count": "statement_line_count",
    "statement_paragraph_count": "statement_paragraph_count",
    "sample_count": "sample_count",
    "number_count": "number_count",
    "math_symbol_count": "math_symbol_count",
    "constraint_keyword_count": "constraint_keyword_count",
    "time_limit_ms": "time_limit_ms",
    "memory_limit_mb": "memory_limit_mb",
}


class PredictionDemoError(RuntimeError):
    """Raised when the local prediction demo cannot proceed safely."""


@dataclass(frozen=True)
class ProblemIdentity:
    """Problem identity shown in prediction outputs."""

    problem_id: str
    contest_id: object | None
    index: str
    name: str | None
    tags: list[str]
    official_rating: float | None


@dataclass(frozen=True)
class ResidualSummary:
    """Empirical residual statistics used for prediction ranges."""

    residual_mae_validation: float | None
    residual_q80_validation: float | None
    range_method: str


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    """Return unique strings while preserving first appearance."""

    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _column_token(column: str) -> str:
    """Normalize a column name for robust matching."""

    return re.sub(r"[^a-z0-9]+", "", column.lower())


def find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Find a column by exact name or normalized candidate name."""

    exact = [candidate for candidate in candidates if candidate in frame.columns]
    if exact:
        return exact[0]
    normalized = {_column_token(column): column for column in frame.columns}
    for candidate in candidates:
        match = normalized.get(_column_token(candidate))
        if match is not None:
            return match
    return None


def standardize_core_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with core columns renamed to project-standard names."""

    output = frame.copy()
    mappings = {
        STANDARD_CONTEST_COLUMN: (
            "contest_id",
            "contestId",
            "contestid",
            "cf_contest_id",
        ),
        STANDARD_INDEX_COLUMN: ("index", "problem_index", "problemIndex"),
        STANDARD_NAME_COLUMN: ("name", "problem_name", "problemName"),
        TARGET_COLUMN: ("rating", "problem_rating", "target_rating"),
    }
    rename: dict[str, str] = {}
    for standard, candidates in mappings.items():
        if standard in output.columns:
            continue
        found = find_column(output, candidates)
        if found is not None and found not in rename:
            rename[found] = standard
    if rename:
        output = output.rename(columns=rename)
    return output


def _key_value(value: object) -> str:
    """Normalize problem key values for stable comparisons."""

    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def normalize_tag_name(tag: str) -> str:
    """Normalize a Codeforces-style tag to the project one-hot suffix."""

    cleaned = tag.strip().lower()
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"[^a-z0-9+* ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = TAG_SYNONYMS.get(cleaned, cleaned)
    cleaned = cleaned.replace(" ", "_").replace("+", "plus").replace("*", "special")
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise PredictionDemoError("Tag names must not be empty.")
    return cleaned


def normalize_tag_to_column(tag: str) -> str:
    """Normalize user tag input to a model-table tag column name."""

    return f"tag__{normalize_tag_name(tag)}"


def parse_problem_index(problem_index: str) -> tuple[str, float | None, float | None]:
    """Extract index letter, numeric suffix, and rank from a Codeforces index."""

    raw = str(problem_index).strip()
    match = re.match(r"^([A-Za-z]+)(\d*)", raw)
    if not match:
        return "", None, None
    letter = match.group(1).upper()
    number = float(match.group(2)) if match.group(2) else None
    rank = 0
    for character in letter:
        if "A" <= character <= "Z":
            rank = rank * 26 + (ord(character) - ord("A") + 1)
    return letter, number, float(rank) if rank else None


def load_model_feature_columns(feature_path: Path) -> list[str]:
    """Load sibling ``feature_columns.json`` when present."""

    metadata_path = feature_path.parent / "feature_columns.json"
    if not metadata_path.exists():
        return []
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    columns = payload.get("feature_columns", []) if isinstance(payload, dict) else []
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        raise PredictionDemoError(f"Invalid feature column metadata: {metadata_path}")
    return list(columns)


def load_demo_frame(
    feature_path: Path,
    statement_feature_path: Path | None = None,
) -> tuple[pd.DataFrame, list[str], bool]:
    """Load model features and optionally join statement text-light features."""

    if not feature_path.exists():
        raise PredictionDemoError(f"Feature table does not exist: {feature_path}")
    frame = standardize_core_columns(pd.read_parquet(feature_path, engine="pyarrow"))
    _require_core_columns(frame, (STANDARD_CONTEST_COLUMN, STANDARD_INDEX_COLUMN, TARGET_COLUMN))
    model_feature_columns = load_model_feature_columns(feature_path)
    statement_features_loaded = False
    if statement_feature_path is not None:
        if not statement_feature_path.exists():
            raise PredictionDemoError(
                f"Statement feature table does not exist: {statement_feature_path}"
            )
        statement = standardize_core_columns(
            pd.read_parquet(statement_feature_path, engine="pyarrow")
        )
        try:
            frame, _ = join_statement_features(frame, statement)
        except StatementColdStartError as error:
            raise PredictionDemoError(str(error)) from error
        statement_features_loaded = True
    return frame, model_feature_columns, statement_features_loaded


def _require_core_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a clear error when required columns are absent."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise PredictionDemoError(f"Feature table lacks required columns: {missing}")


def select_known_problem(
    frame: pd.DataFrame,
    contest_id: object,
    index: str,
) -> pd.Series:
    """Select one known problem by contest id and index."""

    _require_core_columns(frame, (STANDARD_CONTEST_COLUMN, STANDARD_INDEX_COLUMN))
    mask = (
        frame[STANDARD_CONTEST_COLUMN].map(_key_value) == _key_value(contest_id)
    ) & (frame[STANDARD_INDEX_COLUMN].map(_key_value).str.upper() == str(index).upper())
    matches = frame.loc[mask]
    if matches.empty:
        raise PredictionDemoError(
            f"Known problem not found: contest_id={contest_id}, index={index}"
        )
    return matches.iloc[0].copy()


def exclude_known_problem_from_training(
    frame: pd.DataFrame,
    contest_id: object,
    index: str,
) -> pd.DataFrame:
    """Return training rows with the selected known problem removed."""

    mask = (
        frame[STANDARD_CONTEST_COLUMN].map(_key_value) == _key_value(contest_id)
    ) & (frame[STANDARD_INDEX_COLUMN].map(_key_value).str.upper() == str(index).upper())
    return frame.loc[~mask].copy()


def _is_list_like_cell(value: object) -> bool:
    """Return whether a cell is a list-like object unsuitable for sklearn."""

    return isinstance(value, (list, tuple, set, dict, np.ndarray))


def _is_modelable_column(frame: pd.DataFrame, column: str) -> bool:
    """Return whether a selected column is safe for tabular sklearn pipelines."""

    if column not in frame.columns:
        return False
    series = frame[column]
    non_null = series.dropna()
    if non_null.empty:
        return True
    return not non_null.map(_is_list_like_cell).any()


def _filter_modelable_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Drop columns that are not safe scalar model inputs."""

    return [
        column
        for column in columns
        if _is_modelable_column(frame, column) and column != TARGET_COLUMN
    ]


def _available_text_light_features(frame: pd.DataFrame) -> list[str]:
    """Select numeric statement text-light columns present in a frame."""

    columns: list[str] = []
    for column in STATEMENT_FEATURE_COLUMNS:
        if column not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            columns.append(column)
    return sorted(_dedupe_preserve_order(columns))


def select_features_for_scenario(
    frame: pd.DataFrame,
    scenario: str,
    model_feature_columns: Sequence[str] | None = None,
) -> list[str]:
    """Select feature columns for one prediction scenario with leakage controls."""

    model_feature_columns = list(model_feature_columns or [])
    try:
        if scenario == "post_publication_reference":
            columns = select_full_api_reference_features(frame, model_feature_columns)
        elif scenario == "cold_start_metadata":
            columns = select_metadata_features(frame, model_feature_columns)
        elif scenario == "cold_start_metadata_plus_text_light":
            metadata = select_metadata_features(frame, model_feature_columns)
            text_light = _available_text_light_features(frame)
            if not text_light:
                raise PredictionDemoError(
                    "Statement text-light features are unavailable for this scenario."
                )
            columns = _dedupe_preserve_order([*metadata, *text_light])
        else:
            raise PredictionDemoError(f"Unknown prediction scenario: {scenario}")
    except StatementColdStartError as error:
        raise PredictionDemoError(str(error)) from error

    selected = _filter_modelable_columns(frame, columns)
    if scenario != "post_publication_reference":
        leakage_columns = [column for column in selected if has_solved_leakage(column)]
        if leakage_columns:
            raise PredictionDemoError(
                f"Cold-start scenario selected solved-behavior columns: {leakage_columns}"
            )
    if not selected:
        raise PredictionDemoError(f"No usable feature columns for scenario: {scenario}")
    return selected


def _zero_value_for_dtype(dtype: object) -> object:
    """Return a safe default value for a pandas dtype."""

    if pd.api.types.is_bool_dtype(dtype):
        return False
    if pd.api.types.is_numeric_dtype(dtype):
        return 0.0
    return ""


def _empty_manual_row(template: pd.DataFrame) -> dict[str, object]:
    """Create an empty synthetic row with all template columns present."""

    row: dict[str, object] = {}
    for column in template.columns:
        row[column] = _zero_value_for_dtype(template[column].dtype)
    row[TARGET_COLUMN] = np.nan
    row[STANDARD_CONTEST_COLUMN] = np.nan
    row[STANDARD_NAME_COLUMN] = "manual hypothetical problem"
    return row


def _set_if_present(row: dict[str, object], column: str, value: object) -> None:
    """Set a value only when the destination column exists."""

    if column in row:
        row[column] = value


def build_manual_feature_row(
    template: pd.DataFrame,
    *,
    problem_index: str,
    tags: Sequence[str],
    points: float | None = None,
    solved_count: float | None = None,
    statement_values: dict[str, float | int | None] | None = None,
) -> pd.Series:
    """Build one synthetic feature row from manual CLI inputs."""

    if not tags:
        raise PredictionDemoError("Manual mode requires at least one tag.")
    frame = standardize_core_columns(template)
    _require_core_columns(frame, (STANDARD_INDEX_COLUMN, TARGET_COLUMN))
    row = _empty_manual_row(frame)
    row[STANDARD_INDEX_COLUMN] = str(problem_index).strip().upper()

    index_letter, index_number, index_rank = parse_problem_index(problem_index)
    _set_if_present(row, "index_letter", index_letter)
    _set_if_present(row, "index_number", index_number if index_number is not None else 0.0)
    _set_if_present(row, "index_rank", index_rank if index_rank is not None else 0.0)

    if points is not None:
        _set_if_present(row, "points", float(points))
        _set_if_present(row, "has_points", 1)
    else:
        _set_if_present(row, "points", 0.0)
        _set_if_present(row, "has_points", 0)

    tag_columns = [normalize_tag_to_column(tag) for tag in tags]
    missing_tag_columns = [column for column in tag_columns if column not in frame.columns]
    if missing_tag_columns:
        raise PredictionDemoError(
            "Requested tag(s) are not available in the model table: "
            f"{missing_tag_columns}"
        )
    for column in [column for column in frame.columns if column.startswith("tag__")]:
        row[column] = 0
    for column in tag_columns:
        row[column] = 1
    _set_if_present(row, "tag_count", float(len(_dedupe_preserve_order(tag_columns))))

    if solved_count is not None:
        if solved_count < 0:
            raise PredictionDemoError("--solved-count must be non-negative.")
        _set_if_present(row, "solved_count", float(solved_count))
        _set_if_present(row, "log_solved_count", float(np.log1p(solved_count)))
        _set_if_present(row, "solved_count_missing", 0)
    else:
        _set_if_present(row, "solved_count", 0.0)
        _set_if_present(row, "log_solved_count", 0.0)
        _set_if_present(row, "solved_count_missing", 1)

    for column in STATEMENT_FEATURE_COLUMNS:
        if column in row:
            row[column] = 0.0
    if statement_values:
        for source_name, value in statement_values.items():
            column = MANUAL_STATEMENT_ARGUMENTS.get(source_name, source_name)
            if column in row and value is not None:
                row[column] = float(value)
    if "statement_available" in row and statement_values:
        has_any_statement_value = any(value is not None for value in statement_values.values())
        row["statement_available"] = int(has_any_statement_value)
    if "has_time_limit" in row:
        row["has_time_limit"] = int(row.get("time_limit_ms") not in (None, 0, 0.0, ""))
    if "has_memory_limit" in row:
        row["has_memory_limit"] = int(
            row.get("memory_limit_mb") not in (None, 0, 0.0, "")
        )

    return pd.Series(row)


def _extract_known_tags(row: pd.Series) -> list[str]:
    """Extract display tags from raw tags or one-hot tag columns."""

    raw_tags = row.get("tags")
    if isinstance(raw_tags, list):
        return [str(tag) for tag in raw_tags]
    if isinstance(raw_tags, str) and raw_tags.strip():
        return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
    tags: list[str] = []
    for column, value in row.items():
        if column.startswith("tag__") and pd.notna(value) and float(value) > 0:
            tags.append(column.removeprefix("tag__").replace("_", " "))
    return tags


def build_known_problem_identity(row: pd.Series) -> ProblemIdentity:
    """Create display identity from a known feature-table row."""

    contest_id = row.get(STANDARD_CONTEST_COLUMN)
    index = str(row.get(STANDARD_INDEX_COLUMN, ""))
    name = row.get(STANDARD_NAME_COLUMN)
    official_rating = _finite_float(row.get(TARGET_COLUMN))
    return ProblemIdentity(
        problem_id=f"{_key_value(contest_id)}{index}",
        contest_id=contest_id,
        index=index,
        name=str(name) if name is not None and not pd.isna(name) else None,
        tags=_extract_known_tags(row),
        official_rating=official_rating,
    )


def build_manual_problem_identity(row: pd.Series, tags: Sequence[str]) -> ProblemIdentity:
    """Create display identity for a synthetic manual problem."""

    index = str(row.get(STANDARD_INDEX_COLUMN, ""))
    return ProblemIdentity(
        problem_id=f"manual-{index}",
        contest_id=None,
        index=index,
        name="manual hypothetical problem",
        tags=list(tags),
        official_rating=None,
    )


def build_estimator(model_name: str, seed: int) -> object:
    """Create one deterministic demo estimator."""

    if model_name == "hgb":
        return HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=120,
            l2_regularization=0.01,
            random_state=seed,
        )
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=80,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "ridge":
        del seed
        return Ridge(alpha=1.0)
    raise PredictionDemoError(f"Unknown model: {model_name}")


def estimate_residual_summary(
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    model_name: str,
    seed: int,
) -> ResidualSummary:
    """Estimate validation residuals for a transparent prediction range."""

    eligible = train_frame.dropna(subset=[TARGET_COLUMN]).copy()
    if len(eligible) < 6:
        return ResidualSummary(None, None, "fallback_pm250_not_enough_validation_rows")
    val_size = max(2, int(math.ceil(len(eligible) * 0.2)))
    if len(eligible) - val_size < 3:
        return ResidualSummary(None, None, "fallback_pm250_not_enough_training_rows")
    try:
        fit_rows, validation_rows = train_test_split(
            eligible,
            test_size=val_size,
            random_state=seed,
            shuffle=True,
        )
        estimator = make_preprocessed_estimator(
            build_estimator(model_name, seed),
            fit_rows.loc[:, list(feature_columns)],
        )
        estimator.fit(
            fit_rows.loc[:, list(feature_columns)],
            fit_rows[TARGET_COLUMN].astype(float),
        )
        predictions = estimator.predict(validation_rows.loc[:, list(feature_columns)])
        residuals = np.abs(validation_rows[TARGET_COLUMN].astype(float).to_numpy() - predictions)
        if residuals.size == 0 or not np.isfinite(residuals).all():
            return ResidualSummary(None, None, "fallback_pm250_invalid_residuals")
        return ResidualSummary(
            residual_mae_validation=round(float(np.mean(residuals)), 6),
            residual_q80_validation=round(float(np.quantile(residuals, 0.8)), 6),
            range_method="validation_residual_q80",
        )
    except Exception:
        return ResidualSummary(None, None, "fallback_pm250_residual_estimation_failed")


def rounded_to_nearest_100(value: float) -> int:
    """Round a rating value to the nearest 100."""

    return int(round(float(value) / 100.0) * 100)


def prediction_range_from_residual(
    predicted_rating: float,
    residual_summary: ResidualSummary,
) -> tuple[int, int, str]:
    """Build a rounded prediction range from validation residuals."""

    if (
        residual_summary.residual_q80_validation is None
        or not math.isfinite(float(residual_summary.residual_q80_validation))
    ):
        half_width = 250.0
        method = residual_summary.range_method
    else:
        half_width = max(50.0, float(residual_summary.residual_q80_validation))
        method = residual_summary.range_method
    low = rounded_to_nearest_100(max(800.0, predicted_rating - half_width))
    high = rounded_to_nearest_100(min(3500.0, predicted_rating + half_width))
    if low > high:
        low, high = high, low
    return low, high, method


def _finite_float(value: object) -> float | None:
    """Return a finite float or ``None``."""

    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _prepare_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a usable rating target."""

    training = frame.dropna(subset=[TARGET_COLUMN]).copy()
    if len(training) < 4:
        raise PredictionDemoError("Not enough training rows with known ratings.")
    return training


def predict_one_scenario(
    *,
    scenario: str,
    train_frame: pd.DataFrame,
    prediction_row: pd.Series,
    identity: ProblemIdentity,
    model_feature_columns: Sequence[str],
    model_name: str,
    seed: int,
) -> dict[str, object]:
    """Train one runtime demo model and return one prediction record."""

    feature_columns = select_features_for_scenario(
        pd.concat([train_frame, prediction_row.to_frame().T], ignore_index=True),
        scenario,
        model_feature_columns,
    )
    training = _prepare_training_frame(train_frame)
    missing = [column for column in feature_columns if column not in prediction_row.index]
    if missing:
        raise PredictionDemoError(f"Prediction row lacks selected features: {missing}")

    residual_summary = estimate_residual_summary(
        training,
        feature_columns,
        model_name=model_name,
        seed=seed,
    )
    estimator = make_preprocessed_estimator(
        build_estimator(model_name, seed),
        training.loc[:, list(feature_columns)],
    )
    estimator.fit(training.loc[:, list(feature_columns)], training[TARGET_COLUMN].astype(float))
    predicted_rating = float(
        estimator.predict(prediction_row.to_frame().T.loc[:, list(feature_columns)])[0]
    )
    predicted_rating_rounded = rounded_to_nearest_100(predicted_rating)
    range_low, range_high, range_method = prediction_range_from_residual(
        predicted_rating,
        residual_summary,
    )
    official_rating = identity.official_rating
    absolute_error = (
        round(abs(predicted_rating - official_rating), 6)
        if official_rating is not None
        else None
    )
    is_cold_start = scenario != "post_publication_reference"
    uses_solved_behavior = scenario == "post_publication_reference"
    statement_features_used = scenario == "cold_start_metadata_plus_text_light"
    notes = _scenario_note(scenario)
    return {
        "scenario": scenario,
        "is_cold_start": is_cold_start,
        "uses_solved_behavior": uses_solved_behavior,
        "model": model_name,
        "predicted_rating": round(predicted_rating, 6),
        "predicted_rating_rounded": predicted_rating_rounded,
        "prediction_range_low": range_low,
        "prediction_range_high": range_high,
        "range_method": range_method,
        "residual_mae_validation": residual_summary.residual_mae_validation,
        "residual_q80_validation": residual_summary.residual_q80_validation,
        "official_rating": official_rating,
        "absolute_error": absolute_error,
        "problem_id": identity.problem_id,
        "contest_id": identity.contest_id,
        "index": identity.index,
        "name": identity.name,
        "tags": identity.tags,
        "statement_features_used": statement_features_used,
        "notes": notes,
    }


def _scenario_note(scenario: str) -> str:
    """Return one concise scenario explanation."""

    if scenario == "post_publication_reference":
        return (
            "Uses full API features including solved-count behavior; this is a "
            "post-publication reference, not cold-start."
        )
    if scenario == "cold_start_metadata":
        return (
            "Uses API metadata only and excludes solved-count behavior. This is "
            "metadata cold-start, not strict pre-contest cold-start."
        )
    if scenario == "cold_start_metadata_plus_text_light":
        return (
            "Uses metadata plus lightweight statement-structure features and "
            "excludes solved-count behavior."
        )
    return ""


def scenarios_for_request(
    *,
    manual: bool,
    solved_count: float | None,
    has_statement_features: bool,
) -> list[str]:
    """Determine which scenarios can be run for known or manual inputs."""

    scenarios: list[str] = []
    if not manual or solved_count is not None:
        scenarios.append("post_publication_reference")
    scenarios.append("cold_start_metadata")
    if has_statement_features:
        scenarios.append("cold_start_metadata_plus_text_light")
    return scenarios


def run_predictions(
    *,
    frame: pd.DataFrame,
    model_feature_columns: Sequence[str],
    prediction_row: pd.Series,
    train_frame: pd.DataFrame,
    identity: ProblemIdentity,
    scenarios: Sequence[str],
    model_name: str,
    seed: int,
) -> list[dict[str, object]]:
    """Run all requested prediction scenarios."""

    records: list[dict[str, object]] = []
    for scenario in scenarios:
        records.append(
            predict_one_scenario(
                scenario=scenario,
                train_frame=train_frame,
                prediction_row=prediction_row,
                identity=identity,
                model_feature_columns=model_feature_columns,
                model_name=model_name,
                seed=seed,
            )
        )
    return records


def _manual_statement_values_from_args(args: argparse.Namespace) -> dict[str, float | None]:
    """Collect optional manual statement feature values from CLI args."""

    values: dict[str, float | None] = {}
    for argument_name in MANUAL_STATEMENT_ARGUMENTS:
        values[argument_name] = getattr(args, argument_name)
    return values


def _manual_mode_inputs(args: argparse.Namespace) -> None:
    """Validate required manual mode arguments."""

    if not args.problem_index:
        raise PredictionDemoError("--manual requires --problem-index.")
    if not args.tags:
        raise PredictionDemoError("--manual requires --tags with at least one tag.")
    if args.points is not None and args.points < 0:
        raise PredictionDemoError("--points must be non-negative when provided.")


def _known_mode_inputs(args: argparse.Namespace) -> None:
    """Validate required known problem mode arguments."""

    if args.contest_id is None or args.index is None:
        raise PredictionDemoError("Known problem mode requires --contest-id and --index.")


def build_records_from_args(args: argparse.Namespace) -> list[dict[str, object]]:
    """Load data, build the prediction row, and return prediction records."""

    frame, model_feature_columns, statement_features_loaded = load_demo_frame(
        args.feature_path,
        args.statement_feature_path,
    )
    if args.manual:
        _manual_mode_inputs(args)
        statement_values = _manual_statement_values_from_args(args)
        prediction_row = build_manual_feature_row(
            frame,
            problem_index=args.problem_index,
            tags=args.tags,
            points=args.points,
            solved_count=args.solved_count,
            statement_values=statement_values,
        )
        identity = build_manual_problem_identity(prediction_row, args.tags)
        train_frame = frame.copy()
        has_statement_features = bool(_available_text_light_features(frame))
        scenarios = scenarios_for_request(
            manual=True,
            solved_count=args.solved_count,
            has_statement_features=has_statement_features and statement_features_loaded,
        )
    else:
        _known_mode_inputs(args)
        prediction_row = select_known_problem(frame, args.contest_id, args.index)
        identity = build_known_problem_identity(prediction_row)
        train_frame = exclude_known_problem_from_training(frame, args.contest_id, args.index)
        has_statement_features = bool(_available_text_light_features(frame))
        scenarios = scenarios_for_request(
            manual=False,
            solved_count=None,
            has_statement_features=has_statement_features and statement_features_loaded,
        )

    return run_predictions(
        frame=frame,
        model_feature_columns=model_feature_columns,
        prediction_row=prediction_row,
        train_frame=train_frame,
        identity=identity,
        scenarios=scenarios,
        model_name=args.model,
        seed=args.random_seed,
    )


def _json_safe(value: object) -> object:
    """Convert numpy/pandas scalars into JSON-safe values."""

    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    return value


def records_to_frame(records: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Convert prediction records to a table-ready DataFrame."""

    return pd.DataFrame([{key: _json_safe(value) for key, value in record.items()} for record in records])


def write_records(records: Sequence[dict[str, object]], output_path: Path, output_format: str) -> None:
    """Write prediction records as CSV or JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    use_json = suffix == ".json" or (suffix != ".csv" and output_format == "json")
    frame = records_to_frame(records)
    if use_json:
        output_path.write_text(
            json.dumps(
                [{key: _json_safe(value) for key, value in record.items()} for record in records],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        frame.to_csv(output_path, index=False)


def format_console_report(records: Sequence[dict[str, object]], output_format: str) -> str:
    """Format records for console display."""

    if output_format == "json":
        return json.dumps(
            [{key: _json_safe(value) for key, value in record.items()} for record in records],
            ensure_ascii=False,
            indent=2,
        )
    frame = records_to_frame(records)
    display_columns = [
        "scenario",
        "model",
        "predicted_rating_rounded",
        "prediction_range_low",
        "prediction_range_high",
        "official_rating",
        "absolute_error",
        "uses_solved_behavior",
        "statement_features_used",
    ]
    available = [column for column in display_columns if column in frame.columns]
    lines = [
        "Codeforces difficulty prediction demo",
        "",
        "Scenario comparison:",
        frame.loc[:, available].to_string(index=False),
        "",
        "Truthfulness note:",
        (
            "This is a local demo that trains lightweight models at runtime from "
            "the processed dataset. It is not an official Codeforces rating "
            "system and does not create new paper results."
        ),
        (
            "Cold-start here means metadata/statement cold-start; official tags "
            "may not always be available before or during contests."
        ),
    ]
    return "\n".join(lines)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run a local v5.1 Codeforces difficulty prediction demo."
    )
    parser.add_argument("--feature-path", type=Path, required=True)
    parser.add_argument("--statement-feature-path", type=Path, default=None)
    parser.add_argument("--contest-id", type=str, default=None)
    parser.add_argument("--index", type=str, default=None)
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--problem-index", type=str, default=None)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--points", type=float, default=None)
    parser.add_argument("--solved-count", type=float, default=None)
    parser.add_argument("--statement-char-len", type=float, default=None)
    parser.add_argument("--statement-word-count", type=float, default=None)
    parser.add_argument("--statement-line-count", type=float, default=None)
    parser.add_argument("--statement-paragraph-count", type=float, default=None)
    parser.add_argument("--sample-count", type=float, default=None)
    parser.add_argument("--number-count", type=float, default=None)
    parser.add_argument("--math-symbol-count", type=float, default=None)
    parser.add_argument("--constraint-keyword-count", type=float, default=None)
    parser.add_argument("--time-limit-ms", type=float, default=None)
    parser.add_argument("--memory-limit-mb", type=float, default=None)
    parser.add_argument("--model", choices=MODEL_CHOICES, default="hgb")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local prediction demo CLI."""

    args = _build_argument_parser().parse_args(argv)
    try:
        records = build_records_from_args(args)
        if args.output_path is not None:
            write_records(records, args.output_path, args.format)
        print(format_console_report(records, args.format))
    except (
        PredictionDemoError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        ImportError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
