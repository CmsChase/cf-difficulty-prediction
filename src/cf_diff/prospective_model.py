"""Freeze and run the protocol-locked prospective prediction models.

The model artifact is deliberately plain JSON.  Freezing is the only operation
that fits estimators; prediction verifies the frozen protocol, manifest, and
artifact hashes before evaluating the recorded coefficients.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Final, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import scipy
import sklearn
from sklearn.linear_model import Ridge

from cf_diff.statement_features import (
    _decode_bytes,
    build_statement_feature_values,
    parse_problem_statement,
)


DEFAULT_PROTOCOL_PATH: Final[Path] = Path("configs/prospective_protocol_v2.json")
DEFAULT_MODEL_PATH: Final[Path] = Path("prospective/model_bundle_v2.json")
DEFAULT_MANIFEST_PATH: Final[Path] = Path("prospective/model_freeze_manifest_v2.json")
DEFAULT_REQUIREMENTS_PATH: Final[Path] = Path("requirements.txt")

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
    "freeze_manifest_sha256",
    "input_file_sha256",
    "capture_sidecar_sha256",
    "prediction_created_at_utc",
)

_DERIVED_INDEX_COLUMNS: Final[frozenset[str]] = frozenset(
    {"index_rank", "index_number"}
)
_DECODE_POLICY: Final[str] = "utf-8_errors_replace"
_PROTOCOL_DECODE_POLICY: Final[str] = (
    "Decode raw statement bytes as UTF-8 with replacement for invalid byte "
    "sequences; response Content-Type and final redirect URL are audit-only fields."
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


def _sha256_lf_text_file(path: Path) -> str:
    """Hash text with platform line endings normalized to LF."""
    if not path.is_file():
        raise ProspectiveModelError(f"Required text file does not exist: {path}")
    normalized = path.read_bytes().replace(b"\r\n", b"\n")
    return _sha256_bytes(normalized)


def _frozen_source_sha256() -> dict[str, str]:
    """Hash every operational control and golden test frozen with v2."""

    project_root = Path(__file__).resolve().parents[2]
    paths = {
        "prospective_model": Path(__file__),
        "prospective_input": Path(__file__).with_name("prospective_input.py"),
        "prospective_ledger": Path(__file__).with_name("prospective_ledger.py"),
        "prospective_snapshot": Path(__file__).with_name("prospective_snapshot.py"),
        "prospective_cohort": Path(__file__).with_name("prospective_cohort.py"),
        "prospective_analysis": Path(__file__).with_name("prospective_analysis.py"),
        "statement_features": Path(__file__).with_name("statement_features.py"),
        "witness_workflow": project_root
        / ".github"
        / "workflows"
        / "prospective-witness.yml",
        "tests_workflow": project_root / ".github" / "workflows" / "tests.yml",
        "test_prospective_protocol": project_root
        / "tests"
        / "test_prospective_protocol.py",
        "test_prospective_input": project_root
        / "tests"
        / "test_prospective_input.py",
        "test_prospective_model": project_root
        / "tests"
        / "test_prospective_model.py",
        "test_prospective_ledger": project_root
        / "tests"
        / "test_prospective_ledger.py",
        "test_prospective_snapshot": project_root
        / "tests"
        / "test_prospective_snapshot.py",
        "test_prospective_cohort": project_root
        / "tests"
        / "test_prospective_cohort.py",
        "test_prospective_analysis": project_root
        / "tests"
        / "test_prospective_analysis.py",
    }
    return {key: _sha256_lf_text_file(path) for key, path in paths.items()}


def _validate_source_freeze_contract(
    protocol: Mapping[str, object],
    source_hashes: Mapping[str, str],
) -> None:
    analysis = _require_mapping(
        protocol, "confirmatory_analysis", "protocol"
    )
    implementation = _require_mapping(
        analysis, "implementation_freeze", "protocol.confirmatory_analysis"
    )
    if implementation.get("manifest_path") != DEFAULT_MANIFEST_PATH.as_posix():
        raise ProspectiveModelError(
            "Protocol analysis freeze manifest path is not canonical."
        )
    required_keys = implementation.get("required_source_sha256_keys")
    if (
        not isinstance(required_keys, list)
        or any(not isinstance(key, str) for key in required_keys)
        or len(required_keys) != len(set(required_keys))
        or set(required_keys) != set(source_hashes)
    ):
        raise ProspectiveModelError(
            "Protocol analysis source-hash keys do not match the freeze manifest."
        )


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
    created = False
    try:
        with path.open("xb") as handle:
            created = True
            handle.write(_canonical_json_bytes(payload, pretty=True))
    except FileExistsError as error:
        raise ProspectiveModelError(
            f"Frozen artifact already exists and will not be overwritten: {path}"
        ) from error
    except Exception:
        if created and path.exists():
            path.unlink()
        raise


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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_frozen_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> dict[str, object]:
    """Load and validate the immutable fields used by model code."""
    protocol = _read_json_mapping(path, "Prospective protocol")
    if protocol.get("schema_version") != 1:
        raise ProspectiveModelError("Protocol schema_version must equal 1.")
    if protocol.get("status") != "frozen":
        raise ProspectiveModelError("Protocol status must be 'frozen'.")
    _require_string(protocol, "protocol_id", "protocol")
    protocol_frozen_at = _parse_utc(
        _require_string(protocol, "protocol_frozen_at_utc", "protocol"),
        "protocol.protocol_frozen_at_utc",
    )

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
    if protocol_frozen_at >= start:
        raise ProspectiveModelError(
            "Protocol must be frozen before the prospective cohort starts."
        )

    bundle = _require_mapping(protocol, "model_bundle", "protocol")
    _require_string(bundle, "bundle_id", "protocol.model_bundle")
    if bundle.get("estimator") != "Ridge regression":
        raise ProspectiveModelError("Protocol estimator must be 'Ridge regression'.")
    alpha = bundle.get("alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ProspectiveModelError("Protocol Ridge alpha must be numeric.")
    if not math.isclose(float(alpha), 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ProspectiveModelError("Protocol Ridge alpha must equal 1.0.")
    if bundle.get("fit_intercept") is not True:
        raise ProspectiveModelError("Protocol fit_intercept must be true.")
    if bundle.get("solver") != "svd":
        raise ProspectiveModelError("Protocol Ridge solver must be 'svd'.")
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
    capture = _require_mapping(protocol, "input_capture", "protocol")
    if capture.get("decode_policy") != _PROTOCOL_DECODE_POLICY:
        raise ProspectiveModelError(
            "Protocol input capture decode policy does not match the implementation."
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


def _prediction_table_columns(path: Path) -> list[str]:
    """Inspect only table metadata before any prospective values are loaded."""
    if not path.is_file():
        raise ProspectiveModelError(f"Prediction input does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                header_line = handle.readline()
            if not header_line:
                raise ProspectiveModelError("Prediction input CSV is empty.")
            header = next(csv.reader([header_line], strict=True))
        except (OSError, UnicodeError, csv.Error) as error:
            raise ProspectiveModelError(
                f"Prediction input CSV header cannot be read: {path}"
            ) from error
        columns = [str(column) for column in header]
    elif suffix in {".parquet", ".pq"}:
        try:
            columns = list(pq.ParquetFile(path).schema_arrow.names)
        except (OSError, ValueError) as error:
            raise ProspectiveModelError(
                f"Prediction input Parquet schema cannot be read: {path}"
            ) from error
    else:
        raise ProspectiveModelError(
            f"Prediction input must be CSV or Parquet: {path}"
        )
    if not columns or any(not column for column in columns):
        raise ProspectiveModelError("Prediction input has an empty column name.")
    duplicates = sorted(
        {column for column in columns if columns.count(column) > 1}
    )
    if duplicates:
        raise ProspectiveModelError(
            f"Prediction input has duplicate columns: {duplicates}"
        )
    return columns


def _expected_prediction_columns(feature_columns: Sequence[str]) -> list[str]:
    return [
        "contest_id",
        "index",
        *(
            column
            for column in feature_columns
            if column not in _DERIVED_INDEX_COLUMNS
        ),
    ]


def _validate_prediction_schema(
    path: Path,
    feature_columns: Sequence[str],
) -> list[str]:
    """Reject unsafe schemas before reading any prospective row values."""
    columns = _prediction_table_columns(path)
    _reject_forbidden_prediction_columns(columns)
    expected = _expected_prediction_columns(feature_columns)
    missing = sorted(set(expected) - set(columns))
    unexpected = sorted(set(columns) - set(expected))
    if missing or unexpected or columns != expected:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        if not missing and not unexpected and columns != expected:
            details.append("column order does not match the frozen allowlist")
        raise ProspectiveModelError(
            "Prediction input must have the exact frozen schema (" + "; ".join(details) + ")."
        )
    return expected


def _read_prediction_table(
    path: Path,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    expected = _validate_prediction_schema(path, feature_columns)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path,
            usecols=expected,
            dtype={"contest_id": "string", "index": "string"},
            keep_default_na=False,
        )
    return pd.read_parquet(path, engine="pyarrow", columns=expected)


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
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        blank_string = original.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        malformed = numeric.isna() & ~(original.isna() | blank_string)
        if malformed.any():
            examples = original.loc[malformed].astype(str).head(3).tolist()
            raise ProspectiveModelError(
                f"Feature {column!r} contains malformed numeric values: {examples}"
            )
        values = numeric.to_numpy(dtype=float, copy=True)
        if np.isinf(values).any():
            raise ProspectiveModelError(
                f"Feature {column!r} contains infinite values."
            )
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
    estimator = Ridge(
        alpha=1.0,
        fit_intercept=True,
        solver="svd",
        copy_X=True,
        positive=False,
    )
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
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=normal",
                "--",
                "src/cf_diff",
                "configs",
                "requirements.txt",
                ".gitattributes",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout.strip():
            raise ProspectiveModelError(
                "Freeze-relevant source files are dirty; commit them before "
                "creating a model bundle."
            )
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProspectiveModelError(
            "source_commit was not supplied and the Git commit could not be read."
        ) from error
    return completed.stdout.strip()


def _runtime_provenance() -> dict[str, str]:
    config = getattr(np.__config__, "CONFIG", {})
    dependencies = (
        config.get("Build Dependencies", {})
        if isinstance(config, dict)
        else {}
    )
    blas = dependencies.get("blas", {}) if isinstance(dependencies, dict) else {}
    lapack = (
        dependencies.get("lapack", {})
        if isinstance(dependencies, dict)
        else {}
    )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "machine": platform.machine() or "unknown",
        "blas": str(blas.get("name", "unknown")),
        "blas_version": str(blas.get("version", "unknown")),
        "lapack": str(lapack.get("name", "unknown")),
        "lapack_version": str(lapack.get("version", "unknown")),
    }


def freeze_prospective_model(
    *,
    protocol_path: Path,
    model_table_path: Path,
    statement_features_path: Path,
    model_path: Path,
    manifest_path: Path,
    requirements_path: Path = DEFAULT_REQUIREMENTS_PATH,
) -> dict[str, Path]:
    """Fit both protocol-locked Ridge models and write JSON freeze artifacts."""
    if model_path.resolve() == manifest_path.resolve():
        raise ProspectiveModelError("Model and manifest paths must be different.")
    existing = [path for path in (model_path, manifest_path) if path.exists()]
    if existing:
        raise ProspectiveModelError(
            "Frozen artifacts already exist and will not be overwritten: "
            f"{existing}"
        )
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
        "estimator": {
            "name": "Ridge",
            "alpha": 1.0,
            "fit_intercept": True,
            "solver": "svd",
            "copy_X": True,
            "positive": False,
        },
        "preprocessing": {
            "imputation": "per-feature median",
            "standardization": "population mean and standard deviation",
            "constant_scale": 1.0,
        },
        "primary_model": _fit_locked_ridge(training, primary_columns),
        "comparator_model": _fit_locked_ridge(training, comparator_columns),
    }
    model_artifact_sha256 = _sha256_bytes(
        _canonical_json_bytes(artifact, pretty=True)
    )

    commit = _default_source_commit().strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ProspectiveModelError(
            "source_commit must be a complete 40-character Git commit SHA."
        )
    frozen_at = _parse_utc(_utc_now(), "model freeze clock")
    if frozen_at >= cutoff:
        raise ProspectiveModelError(
            "The model bundle must be frozen before the prospective cohort starts."
        )
    protocol_frozen_at = _parse_utc(
        _require_string(protocol, "protocol_frozen_at_utc", "protocol"),
        "protocol.protocol_frozen_at_utc",
    )
    if frozen_at < protocol_frozen_at:
        raise ProspectiveModelError(
            "The model bundle cannot predate the protocol freeze timestamp."
        )
    starts = pd.to_numeric(model_table["start_time_seconds"], errors="coerce")
    eligible_starts = starts.loc[starts < cutoff.timestamp()]
    source_sha256 = _frozen_source_sha256()
    _validate_source_freeze_contract(protocol, source_sha256)
    external_timestamp = _require_mapping(
        protocol, "external_timestamp", "protocol"
    )
    if external_timestamp.get("workflow_file_sha256") != source_sha256.get(
        "witness_workflow"
    ):
        raise ProspectiveModelError(
            "Protocol workflow SHA-256 does not match the frozen witness workflow."
        )
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
        "runtime": _runtime_provenance(),
        "dependency_spec": {
            "path": requirements_path.as_posix(),
            "sha256": _sha256_lf_text_file(requirements_path),
        },
        "source_sha256": source_sha256,
        "training_cutoff_utc": _format_utc(cutoff),
        "training_row_count": int(len(training)),
        "training_start_time_seconds_min": float(eligible_starts.min()),
        "training_start_time_seconds_max": float(eligible_starts.max()),
    }
    created_model = False
    try:
        _write_json(model_path, artifact)
        created_model = True
        _write_json(manifest_path, manifest)
    except Exception:
        if created_model and model_path.exists() and not manifest_path.exists():
            model_path.unlink()
        raise
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


def _require_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ProspectiveModelError(
            f"{description} must be a lowercase SHA-256 digest."
        )
    return value


def _validate_freeze_manifest(
    manifest: Mapping[str, object],
    protocol: Mapping[str, object],
) -> None:
    cohort = _require_mapping(protocol, "cohort", "protocol")
    cutoff = _parse_utc(
        _require_string(cohort, "eligibility_start_utc", "protocol.cohort"),
        "protocol.cohort.eligibility_start_utc",
    )
    protocol_frozen_at = _parse_utc(
        _require_string(protocol, "protocol_frozen_at_utc", "protocol"),
        "protocol.protocol_frozen_at_utc",
    )
    manifest_frozen_at = _parse_utc(
        _require_string(manifest, "frozen_at_utc", "manifest"),
        "manifest.frozen_at_utc",
    )
    if not protocol_frozen_at <= manifest_frozen_at < cutoff:
        raise ProspectiveModelError(
            "Freeze manifest timestamp must fall between protocol freeze and cohort start."
        )
    if manifest.get("training_cutoff_utc") != _format_utc(cutoff):
        raise ProspectiveModelError(
            "Freeze manifest training cutoff does not match the protocol cohort start."
        )
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", source_commit
    ):
        raise ProspectiveModelError(
            "Freeze manifest source_commit must be a complete Git SHA."
        )
    _require_sha256(
        manifest.get("model_artifact_sha256"),
        "manifest.model_artifact_sha256",
    )
    input_hashes = _require_mapping(manifest, "input_sha256", "manifest")
    if set(input_hashes) != {"model_table", "statement_features"}:
        raise ProspectiveModelError(
            "Freeze manifest input_sha256 must name exactly the two training inputs."
        )
    for key, value in input_hashes.items():
        _require_sha256(value, f"manifest.input_sha256.{key}")
    runtime = _require_mapping(manifest, "runtime", "manifest")
    runtime_keys = {
        "python",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit_learn",
        "platform",
        "machine",
        "blas",
        "blas_version",
        "lapack",
        "lapack_version",
    }
    if set(runtime) != runtime_keys or any(
        not isinstance(runtime.get(key), str) or not str(runtime[key]).strip()
        for key in runtime_keys
    ):
        raise ProspectiveModelError(
            "Freeze manifest runtime versions are incomplete."
        )
    dependency = _require_mapping(manifest, "dependency_spec", "manifest")
    _require_string(dependency, "path", "manifest.dependency_spec")
    _require_sha256(
        dependency.get("sha256"),
        "manifest.dependency_spec.sha256",
    )
    source_hashes = _require_mapping(manifest, "source_sha256", "manifest")
    expected_source_keys = set(_frozen_source_sha256())
    if set(source_hashes) != expected_source_keys:
        raise ProspectiveModelError(
            "Freeze manifest source_sha256 fields are incomplete."
        )
    for key, value in source_hashes.items():
        _require_sha256(value, f"manifest.source_sha256.{key}")
    _validate_source_freeze_contract(protocol, source_hashes)
    row_count = manifest.get("training_row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
        raise ProspectiveModelError(
            "Freeze manifest training_row_count must be a positive integer."
        )
    minimum = manifest.get("training_start_time_seconds_min")
    maximum = manifest.get("training_start_time_seconds_max")
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, (int, float))
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(minimum))
        or not math.isfinite(float(maximum))
        or float(minimum) > float(maximum)
        or float(maximum) >= cutoff.timestamp()
    ):
        raise ProspectiveModelError(
            "Freeze manifest training time range is invalid."
        )


def _load_verified_bundle(
    *,
    protocol_path: Path,
    model_path: Path,
    manifest_path: Path,
    enforce_runtime: bool = False,
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
    _validate_freeze_manifest(manifest, protocol)
    manifest_runtime = _require_mapping(manifest, "runtime", "manifest")
    if enforce_runtime and manifest_runtime != _runtime_provenance():
        raise ProspectiveModelError(
            "Current numerical runtime does not match the frozen manifest."
        )
    manifest_source_hashes = _require_mapping(
        manifest,
        "source_sha256",
        "manifest",
    )
    current_source_hashes = _frozen_source_sha256()
    if manifest_source_hashes != current_source_hashes:
        raise ProspectiveModelError(
            "Current prospective source files do not match the frozen manifest."
        )
    expected_estimator = {
        "name": "Ridge",
        "alpha": 1.0,
        "fit_intercept": True,
        "solver": "svd",
        "copy_X": True,
        "positive": False,
    }
    if model.get("estimator") != expected_estimator:
        raise ProspectiveModelError(
            "Model artifact does not describe the fully locked Ridge estimator."
        )

    primary_columns = _require_feature_columns(bundle, "primary_feature_columns")
    comparator_columns = _require_feature_columns(bundle, "comparator_feature_columns")
    _validate_model_record(model.get("primary_model"), primary_columns, "primary_model")
    _validate_model_record(
        model.get("comparator_model"), comparator_columns, "comparator_model"
    )
    return protocol, model, manifest


def verify_frozen_model(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, object]:
    """Verify the committed protocol, JSON model, and freeze-manifest hashes."""
    protocol, model, manifest = _load_verified_bundle(
        protocol_path=protocol_path,
        model_path=model_path,
        manifest_path=manifest_path,
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "model_bundle_id": model["model_bundle_id"],
        "model_artifact_sha256": manifest["model_artifact_sha256"],
        "source_commit": manifest.get("source_commit"),
        "training_cutoff_utc": manifest.get("training_cutoff_utc"),
        "training_row_count": manifest.get("training_row_count"),
    }


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
    if frame.empty:
        raise ProspectiveModelError("Prediction input must contain at least one row.")
    normalized = _normalize_keys(frame, "Prediction input")
    contest_ids = normalized["contest_id"].drop_duplicates().tolist()
    if len(contest_ids) != 1:
        raise ProspectiveModelError(
            "Prediction input must contain exactly one contest_id; "
            f"found {contest_ids[:5]}."
        )
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


def _verify_capture_sidecar(
    *,
    sidecar_path: Path,
    input_path: Path,
    protocol_path: Path,
    protocol: Mapping[str, object],
    manifest: Mapping[str, object],
    expected_columns: Sequence[str],
    contest_start: datetime,
    deadline: datetime,
) -> tuple[dict[str, object], str, str, datetime, pd.DataFrame]:
    sidecar = _read_json_mapping(sidecar_path, "T0 capture sidecar")
    if sidecar.get("schema_version") != 1 or sidecar.get("status") != "complete":
        raise ProspectiveModelError(
            "T0 capture sidecar must be a complete schema-version 1 record."
        )
    if sidecar.get("protocol_id") != protocol.get("protocol_id"):
        raise ProspectiveModelError(
            "T0 capture sidecar protocol_id does not match the frozen protocol."
        )
    if sidecar.get("protocol_sha256") != _sha256_file(protocol_path):
        raise ProspectiveModelError(
            "T0 capture sidecar protocol SHA-256 does not match."
        )
    sidecar_start = _parse_utc(
        _require_string(sidecar, "contest_start_utc", "capture_sidecar"),
        "capture_sidecar.contest_start_utc",
    )
    if sidecar_start != contest_start:
        raise ProspectiveModelError(
            "T0 capture sidecar contest start does not match the prediction request."
        )
    capture_started = _parse_utc(
        _require_string(sidecar, "capture_started_at_utc", "capture_sidecar"),
        "capture_sidecar.capture_started_at_utc",
    )
    capture_completed = _parse_utc(
        _require_string(sidecar, "capture_completed_at_utc", "capture_sidecar"),
        "capture_sidecar.capture_completed_at_utc",
    )
    if not contest_start <= capture_started <= capture_completed <= deadline:
        raise ProspectiveModelError(
            "T0 capture sidecar timestamps fall outside the frozen capture window."
        )
    policy = _require_mapping(sidecar, "request_policy", "capture_sidecar")
    if (
        policy.get("metadata_api_used") is not False
        or policy.get("decode_policy") != _DECODE_POLICY
    ):
        raise ProspectiveModelError(
            "T0 capture sidecar request policy does not match the frozen policy."
        )
    extractor_hashes = _require_mapping(
        sidecar,
        "extractor_sha256",
        "capture_sidecar",
    )
    manifest_source_hashes = _require_mapping(
        manifest,
        "source_sha256",
        "manifest",
    )
    expected_extractors = {"prospective_input", "statement_features"}
    if set(extractor_hashes) != expected_extractors or any(
        extractor_hashes.get(key) != manifest_source_hashes.get(key)
        for key in expected_extractors
    ):
        raise ProspectiveModelError(
            "T0 capture extractor hashes do not match the frozen model manifest."
        )
    output = _require_mapping(sidecar, "output", "capture_sidecar")
    if output.get("columns") != list(expected_columns):
        raise ProspectiveModelError(
            "T0 capture sidecar columns do not match the frozen input schema."
        )
    row_count = output.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
        raise ProspectiveModelError(
            "T0 capture sidecar row_count must be a positive integer."
        )
    input_sha256 = _sha256_file(input_path)
    if output.get("sha256") != input_sha256:
        raise ProspectiveModelError(
            "T0 input SHA-256 does not match the capture sidecar."
        )
    requested_indices = sidecar.get("requested_indices")
    problems = sidecar.get("problems")
    sidecar_contest = sidecar.get("contest_id")
    raw_capture_dir_text = sidecar.get("raw_capture_dir")
    if (
        not isinstance(requested_indices, list)
        or any(
            not isinstance(index, str)
            or not re.fullmatch(r"[A-Z0-9]+", index)
            for index in requested_indices
        )
        or len(requested_indices) != row_count
        or len(set(requested_indices)) != row_count
        or not isinstance(problems, list)
        or len(problems) != row_count
        or not isinstance(sidecar_contest, str)
        or not re.fullmatch(r"[1-9][0-9]*", sidecar_contest)
        or not isinstance(raw_capture_dir_text, str)
        or not raw_capture_dir_text.strip()
    ):
        raise ProspectiveModelError(
            "T0 capture sidecar does not account for every requested problem."
        )
    reconstructed_records: list[dict[str, object]] = []
    statement_columns = list(expected_columns[2:])
    for index, problem in zip(requested_indices, problems, strict=True):
        expected_url = (
            "https://codeforces.com/problemset/problem/"
            f"{sidecar_contest}/{index}"
        )
        if (
            not isinstance(problem, dict)
            or problem.get("index") != index
            or problem.get("url") != expected_url
            or problem.get("fetch_status") != "fetched"
            or problem.get("parse_status") != "parsed"
            or problem.get("http_status") != 200
            or problem.get("error") not in {"", None}
            or not isinstance(problem.get("raw_path"), str)
            or not str(problem["raw_path"]).strip()
            or not isinstance(problem.get("raw_html_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(problem["raw_html_sha256"]))
            or not isinstance(problem.get("decoded_html_sha256"), str)
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(problem["decoded_html_sha256"]),
            )
            or not isinstance(problem.get("final_url"), str)
            or not str(problem["final_url"]).startswith(("https://", "http://"))
            or (
                problem.get("response_content_type") is not None
                and not isinstance(problem.get("response_content_type"), str)
            )
        ):
            raise ProspectiveModelError(
                "T0 capture sidecar contains an incomplete problem record."
            )
        raw_path = Path(str(problem["raw_path"]))
        expected_raw_path = (
            Path(raw_capture_dir_text) / f"{sidecar_contest}_{index}.html"
        )
        if (
            raw_path.resolve() != expected_raw_path.resolve()
            or not raw_path.is_file()
            or _sha256_file(raw_path) != problem.get("raw_html_sha256")
        ):
            raise ProspectiveModelError(
                "T0 raw statement file is missing, misplaced, or hash-mismatched."
            )
        decoded_html = _decode_bytes(
            raw_path.read_bytes(),
            "text/html; charset=utf-8",
        )
        if _sha256_bytes(decoded_html.encode("utf-8")) != problem.get(
            "decoded_html_sha256"
        ):
            raise ProspectiveModelError(
                "T0 decoded statement hash does not match the capture sidecar."
            )
        parsed = parse_problem_statement(decoded_html)
        if parsed.status != "parsed":
            raise ProspectiveModelError(
                "T0 raw statement no longer parses under the frozen extractor."
            )
        values = build_statement_feature_values(parsed)
        reconstructed_records.append(
            {
                "contest_id": sidecar_contest,
                "index": index,
                **{column: values[column] for column in statement_columns},
            }
        )
        fetch_started = _parse_utc(
            _require_string(problem, "fetch_started_at_utc", "capture problem"),
            "capture problem.fetch_started_at_utc",
        )
        fetch_completed = _parse_utc(
            _require_string(problem, "fetch_completed_at_utc", "capture problem"),
            "capture problem.fetch_completed_at_utc",
        )
        if not capture_started <= fetch_started <= fetch_completed <= capture_completed:
            raise ProspectiveModelError(
                "T0 capture sidecar problem timestamps are inconsistent."
            )
    reconstructed = pd.DataFrame(
        reconstructed_records,
        columns=list(expected_columns),
    )
    return (
        sidecar,
        input_sha256,
        _sha256_file(sidecar_path),
        capture_completed,
        reconstructed,
    )


def _require_capture_feature_match(
    actual: pd.DataFrame,
    reconstructed: pd.DataFrame,
    feature_columns: Sequence[str],
) -> None:
    expected = _prediction_feature_frame(reconstructed, feature_columns)
    key_columns = ["contest_id", "index"]
    if actual.loc[:, key_columns].astype(str).to_dict("records") != expected.loc[
        :, key_columns
    ].astype(str).to_dict("records"):
        raise ProspectiveModelError(
            "Prediction input keys do not match the verified raw statements."
        )
    statement_columns = [
        column
        for column in feature_columns
        if column not in _DERIVED_INDEX_COLUMNS
    ]
    actual_matrix = _numeric_matrix(actual, statement_columns)
    expected_matrix = _numeric_matrix(expected, statement_columns)
    if not np.array_equal(actual_matrix, expected_matrix, equal_nan=True):
        raise ProspectiveModelError(
            "Prediction input features do not match features reconstructed from "
            "the verified raw statements."
        )


def predict_prospective(
    *,
    protocol_path: Path,
    model_path: Path,
    manifest_path: Path,
    input_path: Path,
    capture_sidecar_path: Path,
    output_path: Path,
    contest_start_utc: str | datetime,
    _clock: Callable[[], datetime] | None = None,
) -> Path:
    """Predict without fitting after verifying all frozen-artifact hashes."""
    output_suffix = output_path.suffix.lower()
    if output_suffix not in {".csv", ".parquet", ".pq"}:
        raise ProspectiveModelError(
            "Prediction output path must have a .csv or .parquet suffix."
        )
    if output_path.exists():
        raise ProspectiveModelError(
            f"Prediction output already exists and will not be overwritten: {output_path}"
        )
    protocol, model, manifest = _load_verified_bundle(
        protocol_path=protocol_path,
        model_path=model_path,
        manifest_path=manifest_path,
        enforce_runtime=True,
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
    clock = _clock or _utc_now
    prediction_started = _parse_utc(clock(), "prediction clock")
    if prediction_started < contest_start:
        raise ProspectiveModelError(
            "Prediction clock must be on or after contest_start_utc."
        )
    timepoint = _require_mapping(protocol, "prediction_timepoint", "protocol")
    deadline_minutes = timepoint.get("lock_deadline_minutes_after_contest_start")
    if not isinstance(deadline_minutes, int) or deadline_minutes < 1:
        raise ProspectiveModelError(
            "Protocol prediction lock deadline must be a positive integer."
        )
    deadline = contest_start + timedelta(minutes=deadline_minutes)
    if prediction_started > deadline:
        raise ProspectiveModelError(
            "Prediction clock is after the frozen lock deadline."
        )

    bundle = _require_mapping(protocol, "model_bundle", "protocol")
    primary_columns = _require_feature_columns(bundle, "primary_feature_columns")
    comparator_columns = _require_feature_columns(bundle, "comparator_feature_columns")
    expected_input_columns = _validate_prediction_schema(input_path, primary_columns)
    (
        sidecar,
        input_sha256,
        sidecar_sha256,
        capture_completed,
        reconstructed_input,
    ) = (
        _verify_capture_sidecar(
            sidecar_path=capture_sidecar_path,
            input_path=input_path,
            protocol_path=protocol_path,
            protocol=protocol,
            manifest=manifest,
            expected_columns=expected_input_columns,
            contest_start=contest_start,
            deadline=deadline,
        )
    )
    raw_input = _read_prediction_table(input_path, primary_columns)
    features = _prediction_feature_frame(raw_input, primary_columns)
    _require_capture_feature_match(
        features,
        reconstructed_input,
        primary_columns,
    )
    sidecar_contest = str(sidecar.get("contest_id", "")).strip()
    if sidecar_contest != str(features["contest_id"].iloc[0]):
        raise ProspectiveModelError(
            "T0 capture sidecar contest_id does not match the input rows."
        )
    if [str(index) for index in sidecar["requested_indices"]] != features[
        "index"
    ].astype(str).tolist():
        raise ProspectiveModelError(
            "T0 capture sidecar problem order does not match the input rows."
        )
    primary = _predict_record(features, model["primary_model"])
    comparator = _predict_record(features, model["comparator_model"])
    row_hashes = [
        _feature_row_hash(row, primary_columns)
        for _, row in features.iterrows()
    ]
    created_at = _parse_utc(clock(), "prediction creation clock")
    if created_at < capture_completed:
        raise ProspectiveModelError(
            "Prediction creation cannot predate T0 capture completion."
        )
    if created_at > deadline:
        raise ProspectiveModelError(
            "Prediction creation crossed the frozen lock deadline."
        )
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
            "freeze_manifest_sha256": _sha256_file(manifest_path),
            "input_file_sha256": input_sha256,
            "capture_sidecar_sha256": sidecar_sha256,
            "prediction_created_at_utc": created_at_text,
        },
        columns=list(PREDICTION_COLUMNS),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_created = False
    try:
        if output_suffix == ".csv":
            with output_path.open("x", encoding="utf-8", newline="") as handle:
                output_created = True
                result.to_csv(handle, index=False)
        else:
            with output_path.open("xb") as handle:
                output_created = True
                result.to_parquet(handle, engine="pyarrow", index=False)
    except Exception:
        if output_created and output_path.exists():
            output_path.unlink()
        raise
    published_at = _parse_utc(clock(), "prediction publication clock")
    if published_at > deadline:
        output_path.unlink()
        raise ProspectiveModelError(
            "Prediction publication crossed the frozen lock deadline; output removed."
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
    freeze.add_argument(
        "--requirements",
        type=Path,
        default=DEFAULT_REQUIREMENTS_PATH,
    )

    predict = subparsers.add_parser(
        "predict", help="Run the verified frozen bundle without fitting."
    )
    predict.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    predict.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--capture-sidecar", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--contest-start-utc", required=True)

    verify = subparsers.add_parser(
        "verify",
        help="verify protocol, model, and manifest hashes",
    )
    verify.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    verify.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
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
                requirements_path=args.requirements,
            )
            print(f"Wrote frozen model: {paths['model']}")
            print(f"Wrote freeze manifest: {paths['manifest']}")
        elif args.command == "predict":
            output = predict_prospective(
                protocol_path=args.protocol,
                model_path=args.model,
                manifest_path=args.manifest,
                input_path=args.input,
                capture_sidecar_path=args.capture_sidecar,
                output_path=args.output,
                contest_start_utc=args.contest_start_utc,
            )
            print(f"Wrote prospective predictions: {output}")
        else:
            print(
                json.dumps(
                    verify_frozen_model(
                        protocol_path=args.protocol,
                        model_path=args.model,
                        manifest_path=args.manifest,
                    ),
                    sort_keys=True,
                )
            )
    except (ProspectiveModelError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
