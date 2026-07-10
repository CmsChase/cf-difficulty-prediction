"""Freeze and run the protocol-locked prospective prediction models.

The model artifact is deliberately plain JSON.  Freezing is the only operation
that fits estimators; prediction verifies the frozen protocol, manifest, and
artifact hashes before evaluating the recorded coefficients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


DEFAULT_PROTOCOL_PATH: Final[Path] = Path("configs/prospective_protocol_v1.json")
DEFAULT_MODEL_PATH: Final[Path] = Path("prospective/model_bundle_v1.json")
DEFAULT_MANIFEST_PATH: Final[Path] = Path("prospective/model_freeze_manifest_v1.json")

PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "contest_id",
    "index",
    "contest_start_utc",
    "primary_prediction",
    "comparator_prediction",
    "feature_row_sha256",
    "protocol_id",
    "model_bundle_id",
    "model_artifact_sha256",
    "prediction_created_at_utc",
)

_DERIVED_INDEX_COLUMNS: Final[frozenset[str]] = frozenset(
    {"index_rank", "index_number"}
)
_EXPLICIT_FORBIDDEN_COLUMN_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "rating",
        "ratings",
        "point",
        "points",
        "tag",
        "tags",
        "solve",
        "solved",
        "submission",
        "submissions",
        "accept",
        "accepted",
        "acceptance",
        "attempt",
        "attempted",
        "attempts",
        "participant",
        "participants",
        "verdict",
        "verdicts",
    }
)


class ProspectiveModelError(RuntimeError):
    """Raised when model freezing or prediction would violate the protocol."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ProspectiveModelError(f"Required file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: object, *, pretty: bool = False) -> bytes:
    options: dict[str, object] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(payload, **options) + "\n").encode("utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(payload, pretty=True))


def _read_json_mapping(path: Path, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ProspectiveModelError(f"{description} does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ProspectiveModelError(f"{description} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ProspectiveModelError(f"{description} must be a JSON object: {path}")
    return payload


def _require_mapping(
    payload: Mapping[str, object], key: str, description: str
) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ProspectiveModelError(f"{description}.{key} must be an object.")
    return value


def _require_string(
    payload: Mapping[str, object], key: str, description: str
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProspectiveModelError(f"{description}.{key} must be a non-empty string.")
    return value.strip()


def _require_feature_columns(
    bundle: Mapping[str, object], key: str
) -> list[str]:
    value = bundle.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(column, str) or not column for column in value)
    ):
        raise ProspectiveModelError(
            f"protocol.model_bundle.{key} must be a non-empty string list."
        )
    columns = list(value)
    if len(columns) != len(set(columns)):
        raise ProspectiveModelError(
            f"protocol.model_bundle.{key} contains duplicate columns."
        )
    return columns


def _parse_utc(value: str | datetime, description: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise ProspectiveModelError(
                f"{description} must be an ISO-8601 timestamp."
            ) from error
    else:
        raise ProspectiveModelError(f"{description} must be an ISO-8601 timestamp.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProspectiveModelError(f"{description} must include a UTC offset.")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    normalized = value.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.isoformat().replace("+00:00", "Z")


def load_frozen_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> dict[str, object]:
    """Load and validate the immutable fields used by model code."""
    protocol = _read_json_mapping(path, "Prospective protocol")
    if protocol.get("schema_version") != 1:
        raise ProspectiveModelError("Protocol schema_version must equal 1.")
    if protocol.get("status") != "frozen":
        raise ProspectiveModelError("Protocol status must be 'frozen'.")
    _require_string(protocol, "protocol_id", "protocol")

    cohort = _require_mapping(protocol, "cohort", "protocol")
    start = _parse_utc(
        _require_string(cohort, "eligibility_start_utc", "protocol.cohort"),
        "protocol.cohort.eligibility_start_utc",
    )
    end = _parse_utc(
        _require_string(cohort, "eligibility_end_utc", "protocol.cohort"),
        "protocol.cohort.eligibility_end_utc",
    )
    if start >= end:
        raise ProspectiveModelError("Protocol cohort start must precede its end.")

    bundle = _require_mapping(protocol, "model_bundle", "protocol")
    _require_string(bundle, "bundle_id", "protocol.model_bundle")
    if bundle.get("estimator") != "Ridge regression":
        raise ProspectiveModelError("Protocol estimator must be 'Ridge regression'.")
    alpha = bundle.get("alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ProspectiveModelError("Protocol Ridge alpha must be numeric.")
    if not math.isclose(float(alpha), 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ProspectiveModelError("Protocol Ridge alpha must equal 1.0.")
    if bundle.get("randomness") != "none":
        raise ProspectiveModelError("Protocol model randomness must be 'none'.")
    expected_preprocessing = (
        "Per-feature median imputation, then population-variance standardization; "
        "constants use scale 1."
    )
    if bundle.get("numeric_preprocessing") != expected_preprocessing:
        raise ProspectiveModelError(
            "Protocol numeric_preprocessing does not match the implemented frozen "
            "pipeline."
        )
    primary = _require_feature_columns(bundle, "primary_feature_columns")
    comparator = _require_feature_columns(bundle, "comparator_feature_columns")
    if not set(comparator).issubset(primary):
        raise ProspectiveModelError(
            "Comparator feature columns must be a subset of primary columns."
        )
    if not _DERIVED_INDEX_COLUMNS.issubset(primary):
        raise ProspectiveModelError(
            "Protocol primary features must include index_rank and index_number."
        )
    return protocol


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ProspectiveModelError(f"Input table does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(
            path,
            dtype={"contest_id": "string", "index": "string"},
        )
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, engine="pyarrow")
    raise ProspectiveModelError(f"Input table must be CSV or Parquet: {path}")


def _normalize_contest_id(value: object) -> str:
    if value is None or pd.isna(value):
        raise ProspectiveModelError("contest_id contains a missing value.")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    normalized = str(value).strip()
    if not normalized:
        raise ProspectiveModelError("contest_id contains an empty value.")
    return normalized


def _normalize_index(value: object) -> str:
    if value is None or pd.isna(value):
        raise ProspectiveModelError("index contains a missing value.")
    normalized = str(value).strip().upper()
    if not normalized:
        raise ProspectiveModelError("index contains an empty value.")
    return normalized


def _normalize_keys(frame: pd.DataFrame, description: str) -> pd.DataFrame:
    missing = sorted({"contest_id", "index"} - set(frame.columns))
    if missing:
        raise ProspectiveModelError(f"{description} lacks key columns: {missing}")
    result = frame.copy()
    result["contest_id"] = result["contest_id"].map(_normalize_contest_id)
    result["index"] = result["index"].map(_normalize_index)
    duplicate = result.duplicated(["contest_id", "index"], keep=False)
    if duplicate.any():
        keys = result.loc[duplicate, ["contest_id", "index"]].head(5)
        raise ProspectiveModelError(
            f"{description} has duplicate normalized keys: "
            f"{keys.to_dict(orient='records')}"
        )
    return result


def _derive_index_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    indices = result["index"].map(_normalize_index)
    letters = indices.str.extract(r"^([A-Z]+)", expand=False)
    suffixes = indices.str.extract(r"(\d+)$", expand=False)
    rank = letters.str[0].map(
        {letter: number for number, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 1)}
    )
    # Numeric-only problem indices occur in the historical corpus.  Match the
    # canonical feature pipeline by encoding a missing alphabetic prefix as 0.
    result["index_rank"] = rank.fillna(0.0).astype(float)
    result["index_number"] = pd.to_numeric(suffixes, errors="coerce").fillna(0.0)
    return result


def _training_frame(
    *,
    model_table: pd.DataFrame,
    statement_features: pd.DataFrame,
    feature_columns: Sequence[str],
    cutoff: datetime,
) -> pd.DataFrame:
    required_model = {"contest_id", "index", "rating", "start_time_seconds"}
    missing_model = sorted(required_model - set(model_table.columns))
    if missing_model:
        raise ProspectiveModelError(
            f"Historical model table lacks required columns: {missing_model}"
        )
    model = _normalize_keys(model_table, "Historical model table")
    starts = pd.to_numeric(model["start_time_seconds"], errors="coerce")
    if starts.isna().any() or np.isinf(starts.to_numpy(dtype=float)).any():
        raise ProspectiveModelError(
            "Historical start_time_seconds must be finite for every row."
        )
    cutoff_seconds = cutoff.timestamp()
    model = model.loc[starts < cutoff_seconds].copy()
    if model.empty:
        raise ProspectiveModelError("No historical rows occur before the cohort cutoff.")
    ratings = pd.to_numeric(model["rating"], errors="coerce")
    if ratings.isna().any() or np.isinf(ratings.to_numpy(dtype=float)).any():
        raise ProspectiveModelError(
            "Every historical training row must have a finite numeric rating."
        )
    model["rating"] = ratings.astype(float)
    model = _derive_index_features(model)

    statements = _normalize_keys(statement_features, "Statement feature table")
    nonderived = [
        column for column in feature_columns if column not in _DERIVED_INDEX_COLUMNS
    ]
    ambiguous = sorted(
        column
        for column in nonderived
        if column in model.columns and column in statements.columns
    )
    if ambiguous:
        raise ProspectiveModelError(
            f"Protocol features occur ambiguously in both input tables: {ambiguous}"
        )
    missing_features = sorted(
        column
        for column in nonderived
        if column not in model.columns and column not in statements.columns
    )
    if missing_features:
        raise ProspectiveModelError(
            f"Historical inputs lack protocol feature columns: {missing_features}"
        )

    from_model = [column for column in nonderived if column in model.columns]
    from_statements = [column for column in nonderived if column in statements.columns]
    selected = model.loc[
        :, ["contest_id", "index", "rating", "index_rank", "index_number", *from_model]
    ].copy()
    if from_statements:
        statement_selected = statements.loc[
            :, ["contest_id", "index", *from_statements]
        ]
        selected = selected.merge(
            statement_selected,
            on=["contest_id", "index"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if not selected["_merge"].eq("both").all():
            missing_keys = selected.loc[
                selected["_merge"].ne("both"), ["contest_id", "index"]
            ].head(5)
            raise ProspectiveModelError(
                "Statement features are missing for historical training keys: "
                f"{missing_keys.to_dict(orient='records')}"
            )
        selected = selected.drop(columns="_merge")
    return selected.loc[:, ["contest_id", "index", "rating", *feature_columns]]


def _numeric_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for column in feature_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=float,
            copy=True,
        )
        values[np.isinf(values)] = np.nan
        arrays.append(values)
    return np.column_stack(arrays)


def _fit_locked_ridge(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> dict[str, object]:
    raw = _numeric_matrix(frame, feature_columns)
    all_missing = np.isnan(raw).all(axis=0)
    if all_missing.any():
        missing = [
            column for column, flag in zip(feature_columns, all_missing, strict=True) if flag
        ]
        raise ProspectiveModelError(
            f"Cannot freeze features that are entirely missing: {missing}"
        )
    medians = np.nanmedian(raw, axis=0)
    imputed = np.where(np.isnan(raw), medians, raw)
    means = imputed.mean(axis=0)
    scales = imputed.std(axis=0, ddof=0)
    scales = np.where(scales == 0.0, 1.0, scales)
    standardized = (imputed - means) / scales
    target = frame["rating"].to_numpy(dtype=float)
    estimator = Ridge(alpha=1.0, fit_intercept=True)
    estimator.fit(standardized, target)
    return {
        "feature_columns": list(feature_columns),
        "median_imputation": [float(value) for value in medians],
        "population_mean": [float(value) for value in means],
        "population_scale": [float(value) for value in scales],
        "coefficients": [float(value) for value in estimator.coef_],
        "intercept": float(estimator.intercept_),
    }


def _default_source_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProspectiveModelError(
            "source_commit was not supplied and the Git commit could not be read."
        ) from error
    return completed.stdout.strip()


def freeze_prospective_model(
    *,
    protocol_path: Path,
    model_table_path: Path,
    statement_features_path: Path,
    model_path: Path,
    manifest_path: Path,
    source_commit: str | None = None,
    frozen_at_utc: str | datetime | None = None,
) -> dict[str, Path]:
    """Fit both protocol-locked Ridge models and write JSON freeze artifacts."""
    protocol = load_frozen_protocol(protocol_path)
    protocol_sha256 = _sha256_file(protocol_path)
    protocol_id = _require_string(protocol, "protocol_id", "protocol")
    cohort = _require_mapping(protocol, "cohort", "protocol")
    cutoff = _parse_utc(
        _require_string(cohort, "eligibility_start_utc", "protocol.cohort"),
        "protocol.cohort.eligibility_start_utc",
    )
    bundle = _require_mapping(protocol, "model_bundle", "protocol")
    model_bundle_id = _require_string(
        bundle, "bundle_id", "protocol.model_bundle"
    )
    primary_columns = _require_feature_columns(bundle, "primary_feature_columns")
    comparator_columns = _require_feature_columns(
        bundle, "comparator_feature_columns"
    )

    model_table = _read_table(model_table_path)
    statement_features = _read_table(statement_features_path)
    training = _training_frame(
        model_table=model_table,
        statement_features=statement_features,
        feature_columns=primary_columns,
        cutoff=cutoff,
    )
    artifact = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "model_bundle_id": model_bundle_id,
        "estimator": {"name": "Ridge", "alpha": 1.0},
        "preprocessing": {
            "imputation": "per-feature median",
            "standardization": "population mean and standard deviation",
            "constant_scale": 1.0,
        },
        "primary_model": _fit_locked_ridge(training, primary_columns),
        "comparator_model": _fit_locked_ridge(training, comparator_columns),
    }
    _write_json(model_path, artifact)
    model_artifact_sha256 = _sha256_file(model_path)

    commit = (source_commit or _default_source_commit()).strip()
    if not commit or any(character.isspace() for character in commit):
        raise ProspectiveModelError("source_commit must be a non-empty token.")
    frozen_at = (
        datetime.now(timezone.utc)
        if frozen_at_utc is None
        else _parse_utc(frozen_at_utc, "frozen_at_utc")
    )
    if frozen_at >= cutoff:
        raise ProspectiveModelError(
            "The model bundle must be frozen before the prospective cohort starts."
        )
    starts = pd.to_numeric(model_table["start_time_seconds"], errors="coerce")
    eligible_starts = starts.loc[starts < cutoff.timestamp()]
    manifest = {
        "schema_version": 1,
        "model_bundle_id": model_bundle_id,
        "model_artifact_sha256": model_artifact_sha256,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "source_commit": commit,
        "frozen_at_utc": _format_utc(frozen_at),
        "input_sha256": {
            "model_table": _sha256_file(model_table_path),
            "statement_features": _sha256_file(statement_features_path),
        },
        "training_cutoff_utc": _format_utc(cutoff),
        "training_row_count": int(len(training)),
        "training_start_time_seconds_min": float(eligible_starts.min()),
        "training_start_time_seconds_max": float(eligible_starts.max()),
    }
    _write_json(manifest_path, manifest)
    return {"model": model_path, "manifest": manifest_path}


def _validate_model_record(
    record: object,
    expected_columns: Sequence[str],
    description: str,
) -> dict[str, object]:
    if not isinstance(record, dict):
        raise ProspectiveModelError(f"{description} must be an object.")
    if record.get("feature_columns") != list(expected_columns):
        raise ProspectiveModelError(
            f"{description} feature columns do not match the frozen protocol."
        )
    width = len(expected_columns)
    for key in (
        "median_imputation",
        "population_mean",
        "population_scale",
        "coefficients",
    ):
        values = record.get(key)
        if not isinstance(values, list) or len(values) != width:
            raise ProspectiveModelError(f"{description}.{key} has the wrong width.")
        try:
            numeric = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise ProspectiveModelError(
                f"{description}.{key} must be numeric."
            ) from error
        if not np.isfinite(numeric).all():
            raise ProspectiveModelError(f"{description}.{key} must be finite.")
        if key == "population_scale" and (numeric <= 0).any():
            raise ProspectiveModelError(
                f"{description}.population_scale must be positive."
            )
    intercept = record.get("intercept")
    if isinstance(intercept, bool) or not isinstance(intercept, (int, float)):
        raise ProspectiveModelError(f"{description}.intercept must be numeric.")
    if not math.isfinite(float(intercept)):
        raise ProspectiveModelError(f"{description}.intercept must be finite.")
    return record


def _load_verified_bundle(
    *,
    protocol_path: Path,
    model_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    protocol = load_frozen_protocol(protocol_path)
    model = _read_json_mapping(model_path, "Model artifact")
    manifest = _read_json_mapping(manifest_path, "Freeze manifest")
    bundle = _require_mapping(protocol, "model_bundle", "protocol")
    protocol_id = _require_string(protocol, "protocol_id", "protocol")
    model_bundle_id = _require_string(bundle, "bundle_id", "protocol.model_bundle")
    protocol_sha256 = _sha256_file(protocol_path)
    model_sha256 = _sha256_file(model_path)

    expected = {
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "model_bundle_id": model_bundle_id,
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise ProspectiveModelError(
                f"Model artifact {key} does not match the frozen protocol."
            )
        if manifest.get(key) != value:
            raise ProspectiveModelError(
                f"Freeze manifest {key} does not match the frozen protocol."
            )
    if manifest.get("model_artifact_sha256") != model_sha256:
        raise ProspectiveModelError("Model artifact SHA-256 does not match the manifest.")
    if model.get("schema_version") != 1 or manifest.get("schema_version") != 1:
        raise ProspectiveModelError("Model and manifest schema_version must equal 1.")
    if model.get("estimator") != {"name": "Ridge", "alpha": 1.0}:
        raise ProspectiveModelError("Model artifact does not describe locked Ridge(alpha=1).")

    primary_columns = _require_feature_columns(bundle, "primary_feature_columns")
    comparator_columns = _require_feature_columns(bundle, "comparator_feature_columns")
    _validate_model_record(model.get("primary_model"), primary_columns, "primary_model")
    _validate_model_record(
        model.get("comparator_model"), comparator_columns, "comparator_model"
    )
    return protocol, model, manifest


def _column_tokens(column: object) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(column).strip().lower())
        if token
    }


def _reject_forbidden_prediction_columns(columns: Sequence[object]) -> None:
    forbidden: list[str] = []
    for column in columns:
        tokens = _column_tokens(column)
        if tokens & _EXPLICIT_FORBIDDEN_COLUMN_TOKENS:
            forbidden.append(str(column))
            continue
        # Catch common compounds such as solvedCount and submissionCount.
        compact = re.sub(r"[^a-z0-9]", "", str(column).lower())
        if any(
            marker in compact
            for marker in (
                "rating",
                "points",
                "tags",
                "solved",
                "submission",
                "accepted",
                "acceptance",
                "attempted",
                "attemptcount",
                "participantcount",
                "verdict",
            )
        ):
            forbidden.append(str(column))
    if forbidden:
        raise ProspectiveModelError(
            "Prediction input contains forbidden label, metadata, or behavioral "
            f"columns: {sorted(forbidden)}"
        )


def _prediction_feature_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    _reject_forbidden_prediction_columns(list(frame.columns))
    normalized = _normalize_keys(frame, "Prediction input")
    normalized = _derive_index_features(normalized)
    required_input = [
        column for column in feature_columns if column not in _DERIVED_INDEX_COLUMNS
    ]
    missing = sorted(set(required_input) - set(normalized.columns))
    if missing:
        raise ProspectiveModelError(
            f"Prediction input lacks protocol feature columns: {missing}"
        )
    return normalized


def _predict_record(
    frame: pd.DataFrame,
    record: Mapping[str, object],
) -> np.ndarray:
    columns = list(record["feature_columns"])
    raw = _numeric_matrix(frame, columns)
    medians = np.asarray(record["median_imputation"], dtype=float)
    means = np.asarray(record["population_mean"], dtype=float)
    scales = np.asarray(record["population_scale"], dtype=float)
    coefficients = np.asarray(record["coefficients"], dtype=float)
    imputed = np.where(np.isnan(raw), medians, raw)
    standardized = (imputed - means) / scales
    return standardized @ coefficients + float(record["intercept"])


def _json_scalar(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def _feature_row_hash(
    row: pd.Series,
    feature_columns: Sequence[str],
) -> str:
    payload = {
        "contest_id": str(row["contest_id"]),
        "index": str(row["index"]),
        "features": {
            column: _json_scalar(row[column]) for column in feature_columns
        },
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def predict_prospective(
    *,
    protocol_path: Path,
    model_path: Path,
    manifest_path: Path,
    input_path: Path,
    output_path: Path,
    contest_start_utc: str | datetime,
    prediction_created_at_utc: str | datetime | None = None,
) -> Path:
    """Predict without fitting after verifying all frozen-artifact hashes."""
    protocol, model, manifest = _load_verified_bundle(
        protocol_path=protocol_path,
        model_path=model_path,
        manifest_path=manifest_path,
    )
    cohort = _require_mapping(protocol, "cohort", "protocol")
    eligibility_start = _parse_utc(
        _require_string(cohort, "eligibility_start_utc", "protocol.cohort"),
        "protocol.cohort.eligibility_start_utc",
    )
    eligibility_end = _parse_utc(
        _require_string(cohort, "eligibility_end_utc", "protocol.cohort"),
        "protocol.cohort.eligibility_end_utc",
    )
    contest_start = _parse_utc(contest_start_utc, "contest_start_utc")
    if not eligibility_start <= contest_start <= eligibility_end:
        raise ProspectiveModelError(
            "contest_start_utc is outside the frozen protocol cohort window."
        )
    created_at = (
        datetime.now(timezone.utc)
        if prediction_created_at_utc is None
        else _parse_utc(prediction_created_at_utc, "prediction_created_at_utc")
    )
    if created_at < contest_start:
        raise ProspectiveModelError(
            "prediction_created_at_utc must be on or after contest_start_utc."
        )
    timepoint = _require_mapping(protocol, "prediction_timepoint", "protocol")
    deadline_minutes = timepoint.get("lock_deadline_minutes_after_contest_start")
    if not isinstance(deadline_minutes, int) or deadline_minutes < 1:
        raise ProspectiveModelError(
            "Protocol prediction lock deadline must be a positive integer."
        )
    if created_at > contest_start + timedelta(minutes=deadline_minutes):
        raise ProspectiveModelError(
            "prediction_created_at_utc is after the frozen lock deadline."
        )

    bundle = _require_mapping(protocol, "model_bundle", "protocol")
    primary_columns = _require_feature_columns(bundle, "primary_feature_columns")
    comparator_columns = _require_feature_columns(bundle, "comparator_feature_columns")
    raw_input = _read_table(input_path)
    features = _prediction_feature_frame(raw_input, primary_columns)
    primary = _predict_record(features, model["primary_model"])
    comparator = _predict_record(features, model["comparator_model"])
    row_hashes = [
        _feature_row_hash(row, primary_columns)
        for _, row in features.iterrows()
    ]
    contest_start_text = _format_utc(contest_start)
    created_at_text = _format_utc(created_at)
    result = pd.DataFrame(
        {
            "contest_id": features["contest_id"].astype(str),
            "index": features["index"].astype(str),
            "contest_start_utc": contest_start_text,
            "primary_prediction": primary.astype(float),
            "comparator_prediction": comparator.astype(float),
            "feature_row_sha256": row_hashes,
            "protocol_id": protocol["protocol_id"],
            "model_bundle_id": model["model_bundle_id"],
            "model_artifact_sha256": manifest["model_artifact_sha256"],
            "prediction_created_at_utc": created_at_text,
        },
        columns=list(PREDICTION_COLUMNS),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        result.to_csv(output_path, index=False)
    elif suffix in {".parquet", ".pq"}:
        result.to_parquet(output_path, engine="pyarrow", index=False)
    else:
        raise ProspectiveModelError(
            "Prediction output path must have a .csv or .parquet suffix."
        )
    return output_path


# Descriptive aliases make the API easy to discover without changing semantics.
freeze_model_bundle = freeze_prospective_model
predict_with_frozen_model = predict_prospective


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or run the protocol-locked prospective Ridge models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="Fit and freeze the JSON model bundle.")
    freeze.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    freeze.add_argument("--model-table", type=Path, required=True)
    freeze.add_argument("--statement-features", type=Path, required=True)
    freeze.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    freeze.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_PATH)
    freeze.add_argument("--source-commit", default=None)
    freeze.add_argument("--frozen-at-utc", default=None)

    predict = subparsers.add_parser(
        "predict", help="Run the verified frozen bundle without fitting."
    )
    predict.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    predict.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--contest-start-utc", required=True)
    predict.add_argument("--prediction-created-at-utc", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the prospective model command-line interface."""
    args = _build_argument_parser().parse_args(argv)
    try:
        if args.command == "freeze":
            paths = freeze_prospective_model(
                protocol_path=args.protocol,
                model_table_path=args.model_table,
                statement_features_path=args.statement_features,
                model_path=args.model_output,
                manifest_path=args.manifest_output,
                source_commit=args.source_commit,
                frozen_at_utc=args.frozen_at_utc,
            )
            print(f"Wrote frozen model: {paths['model']}")
            print(f"Wrote freeze manifest: {paths['manifest']}")
        else:
            output = predict_prospective(
                protocol_path=args.protocol,
                model_path=args.model,
                manifest_path=args.manifest,
                input_path=args.input,
                output_path=args.output,
                contest_start_utc=args.contest_start_utc,
                prediction_created_at_utc=args.prediction_created_at_utc,
            )
            print(f"Wrote prospective predictions: {output}")
    except (ProspectiveModelError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
