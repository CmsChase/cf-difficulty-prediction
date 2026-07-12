"""Run the frozen forward-time historical statement-only backtest.

The workflow is deliberately split in two.  ``select`` reconstructs statement
features from the checked cache and selects Ridge alphas on validation data.
It never evaluates the test partition.  ``test`` verifies the sealed selection
artifacts, refits the two locked models on train plus validation data, and then
evaluates each model exactly once on the test partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cf_diff.features import ExperimentConfig, SplitRatios
from cf_diff.splits import SplitError, build_forward_time_split
from cf_diff.statement_archive import ArchiveError, verify_archive
from cf_diff.statement_features import (
    STATEMENT_FEATURE_COLUMNS,
    build_statement_feature_values,
    cache_path_for_problem,
    parse_problem_statement,
)

DEFAULT_CONFIG_PATH: Final[Path] = Path("configs/historical_statement_backtest.json")
DEFAULT_MODEL_TABLE_PATH: Final[Path] = Path(
    "data/processed/features/model_table.parquet"
)
DEFAULT_CACHE_MANIFEST_PATH: Final[Path] = Path(
    "data/manifests/historical_statement_cache_v1.csv"
)
DEFAULT_SELECTION_DIR: Final[Path] = Path(
    "outputs/historical_statement_backtest/selection"
)
DEFAULT_TEST_DIR: Final[Path] = Path("outputs/historical_statement_backtest/test")

MODEL_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "contest_id",
    "index",
    "rating",
    "start_time_seconds",
    "index_rank",
    "index_number",
)
PREPARED_ID_COLUMNS: Final[tuple[str, ...]] = MODEL_TABLE_COLUMNS[:4]
EXPECTED_COMPARATOR: Final[tuple[str, ...]] = ("index_rank", "index_number")
EXPECTED_PRIMARY: Final[tuple[str, ...]] = (
    *EXPECTED_COMPARATOR,
    *STATEMENT_FEATURE_COLUMNS,
)
SETTING_NAMES: Final[tuple[str, str]] = ("comparator", "primary")

PREPARED_FILENAME: Final[str] = "prepared_dataset.parquet"
SOURCE_MANIFEST_FILENAME: Final[str] = "source_manifest.csv"
SPLIT_FILENAME: Final[str] = "split_assignment.parquet"
VALIDATION_METRICS_FILENAME: Final[str] = "validation_metrics.csv"
SELECTION_LOCK_FILENAME: Final[str] = "selection_lock.json"
SELECTION_HASH_FILENAME: Final[str] = "selection_lock.sha256"

TEST_METRICS_FILENAME: Final[str] = "test_metrics.json"
TEST_PREDICTIONS_FILENAME: Final[str] = "test_predictions.csv"
BOOTSTRAP_FILENAME: Final[str] = "paired_bootstrap.json"
TOP10_FILENAME: Final[str] = "primary_top10.csv"


class HistoricalBacktestError(RuntimeError):
    """Raised when the frozen backtest contract cannot be followed safely."""


@dataclass(frozen=True)
class HistoricalBacktestConfig:
    """Validated values needed from the frozen JSON protocol."""

    path: Path
    sha256: str
    raw: Mapping[str, Any]
    random_seed: int
    target: str
    cache_dir: Path
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    comparator_features: tuple[str, ...]
    primary_features: tuple[str, ...]
    alpha_candidates: tuple[float, ...]
    bootstrap_resamples: int
    bootstrap_seed: int
    confidence_level: float
    top_count: int


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of the bytes currently stored at ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise HistoricalBacktestError(f"{label} must be a JSON object.")
    return value


def _require_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HistoricalBacktestError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise HistoricalBacktestError(f"{label} must be finite.")
    return result


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HistoricalBacktestError(f"{label} must be a positive integer.")
    return value


def load_backtest_config(path: Path = DEFAULT_CONFIG_PATH) -> HistoricalBacktestConfig:
    """Load and strictly validate the checked historical-backtest protocol."""
    if not path.is_file():
        raise HistoricalBacktestError(f"Backtest config does not exist: {path}")
    try:
        raw_value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HistoricalBacktestError(f"Cannot read backtest config: {error}") from error
    raw = _require_mapping(raw_value, "Backtest config")

    if raw.get("schema_version") != 1:
        raise HistoricalBacktestError("Backtest config schema_version must be 1.")
    if raw.get("design_status") != "frozen_before_rerun":
        raise HistoricalBacktestError("The backtest design must be frozen before rerun.")
    if raw.get("study_type") != "forward_time_retrospective":
        raise HistoricalBacktestError("The study must remain forward-time retrospective.")
    if raw.get("target") != "rating":
        raise HistoricalBacktestError("The frozen target must be 'rating'.")

    split = _require_mapping(raw.get("split"), "split")
    if split.get("unit") != "contest_start_time_bucket":
        raise HistoricalBacktestError(
            "The frozen split unit must be contest_start_time_bucket."
        )
    train_fraction = _require_number(split.get("train_fraction"), "train_fraction")
    validation_fraction = _require_number(
        split.get("validation_fraction"), "validation_fraction"
    )
    test_fraction = _require_number(split.get("test_fraction"), "test_fraction")
    if (train_fraction, validation_fraction, test_fraction) != (0.7, 0.1, 0.2):
        raise HistoricalBacktestError("The frozen split fractions must be 0.70/0.10/0.20.")

    feature_sets = _require_mapping(raw.get("feature_sets"), "feature_sets")
    comparator = tuple(feature_sets.get("comparator", ()))
    primary = tuple(feature_sets.get("primary", ()))
    if comparator != EXPECTED_COMPARATOR:
        raise HistoricalBacktestError(
            f"Comparator must be the exact two-column allowlist {EXPECTED_COMPARATOR}."
        )
    if primary != EXPECTED_PRIMARY or len(primary) != 43:
        raise HistoricalBacktestError(
            "Primary must be index_rank, index_number, and the exact 41 checked "
            "statement features in their frozen order."
        )
    feature_policy = _require_mapping(raw.get("feature_policy"), "feature_policy")
    if feature_policy.get("exact_allowlist_only") is not True:
        raise HistoricalBacktestError("feature_policy.exact_allowlist_only must be true.")

    model = _require_mapping(raw.get("model"), "model")
    if model.get("family") != "ridge":
        raise HistoricalBacktestError("The frozen model family must be Ridge.")
    if (
        model.get("numeric_missing_values") != "training_partition_median"
        or model.get("numeric_scaling") != "training_partition_standard_scaler"
        or model.get("selection_metric") != "validation_mae"
        or model.get("tie_break") != "lowest_alpha"
        or model.get("final_fit") != "refit_selected_alpha_on_train_plus_validation"
    ):
        raise HistoricalBacktestError("The frozen Ridge fitting and selection rules changed.")
    alpha_values = model.get("alpha_candidates")
    if not isinstance(alpha_values, list) or not alpha_values:
        raise HistoricalBacktestError("model.alpha_candidates must be a non-empty list.")
    alphas = tuple(_require_number(value, "alpha candidate") for value in alpha_values)
    if any(value <= 0 for value in alphas) or len(set(alphas)) != len(alphas):
        raise HistoricalBacktestError("Ridge alpha candidates must be positive and unique.")

    uncertainty = _require_mapping(raw.get("uncertainty"), "uncertainty")
    if uncertainty.get("method") != "paired_contest_cluster_bootstrap":
        raise HistoricalBacktestError("The frozen uncertainty method is not available.")
    if uncertainty.get("rng") != "numpy_pcg64":
        raise HistoricalBacktestError("The frozen bootstrap RNG must be numpy_pcg64.")
    if uncertainty.get("quantile_method") != "linear":
        raise HistoricalBacktestError("The frozen quantile method must be linear.")
    if (
        uncertainty.get("cluster") != "contest_id"
        or uncertainty.get("statistic")
        != "primary_mae_minus_comparator_mae"
        or uncertainty.get("interval") != "percentile"
    ):
        raise HistoricalBacktestError("The frozen bootstrap contract changed.")

    test_policy = _require_mapping(raw.get("test_policy"), "test_policy")
    if (
        test_policy.get("evaluate_once_after_all_selection") is not True
        or test_policy.get("primary_metric") != "mae"
        or test_policy.get("primary_contrast")
        != "primary_mae_minus_comparator_mae"
    ):
        raise HistoricalBacktestError("The frozen test policy changed.")

    artifact = _require_mapping(raw.get("statement_artifact"), "statement_artifact")
    cache_value = artifact.get("path")
    if not isinstance(cache_value, str) or not cache_value.strip():
        raise HistoricalBacktestError("statement_artifact.path must be a path string.")
    error_analysis = _require_mapping(raw.get("error_analysis"), "error_analysis")
    if error_analysis.get("model") != "primary" or error_analysis.get(
        "manual_selection"
    ) is not False:
        raise HistoricalBacktestError("Error analysis must be automatic and primary-only.")
    if error_analysis.get("ranking") != [
        "absolute_error_descending",
        "start_time_seconds_ascending",
        "contest_id_ascending",
        "index_ascending",
    ]:
        raise HistoricalBacktestError("The frozen error-ranking rule changed.")

    random_seed = raw.get("random_seed")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise HistoricalBacktestError("random_seed must be an integer.")
    bootstrap_seed = uncertainty.get("random_seed")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise HistoricalBacktestError("uncertainty.random_seed must be an integer.")
    confidence_level = _require_number(
        uncertainty.get("confidence_level"), "confidence_level"
    )
    if not 0.0 < confidence_level < 1.0:
        raise HistoricalBacktestError("confidence_level must lie strictly between 0 and 1.")

    return HistoricalBacktestConfig(
        path=path,
        sha256=sha256_file(path),
        raw=raw,
        random_seed=random_seed,
        target="rating",
        cache_dir=Path(cache_value),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        comparator_features=comparator,
        primary_features=primary,
        alpha_candidates=alphas,
        bootstrap_resamples=_require_positive_int(
            uncertainty.get("resamples"), "uncertainty.resamples"
        ),
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        top_count=_require_positive_int(error_analysis.get("count"), "error_analysis.count"),
    )


def feature_columns(config: HistoricalBacktestConfig, setting: str) -> tuple[str, ...]:
    """Return one exact feature allowlist; never infer columns from a table."""
    if setting == "comparator":
        return config.comparator_features
    if setting == "primary":
        return config.primary_features
    raise HistoricalBacktestError(f"Unknown feature setting: {setting}")


def _normalize_model_table(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in MODEL_TABLE_COLUMNS if column not in frame.columns]
    if missing:
        raise HistoricalBacktestError(f"Model table lacks required columns: {missing}")
    result = frame.loc[:, list(MODEL_TABLE_COLUMNS)].copy()
    result["contest_id"] = pd.to_numeric(result["contest_id"], errors="coerce").astype(
        "Int64"
    )
    result["index"] = result["index"].astype("string")
    for column in ("rating", "start_time_seconds", "index_rank", "index_number"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result.loc[:, list(PREPARED_ID_COLUMNS)].isna().any().any():
        raise HistoricalBacktestError(
            "contest_id, index, rating, and start_time_seconds cannot be missing."
        )
    if result.duplicated(["contest_id", "index"]).any():
        raise HistoricalBacktestError("Model table has duplicate contest_id/index rows.")
    result["contest_id"] = result["contest_id"].astype("int64")
    result["start_time_seconds"] = result["start_time_seconds"].astype("int64")
    return result.sort_values(
        ["start_time_seconds", "contest_id", "index"], kind="mergesort"
    ).reset_index(drop=True)


def _empty_statement_features() -> dict[str, object]:
    values = {column: 0 for column in STATEMENT_FEATURE_COLUMNS}
    values["time_limit_ms"] = None
    values["memory_limit_mb"] = None
    return values


def _statement_row(
    cache_dir: Path, contest_id: object, index: object
) -> tuple[dict[str, object], dict[str, object]]:
    cache_path = cache_path_for_problem(cache_dir, contest_id, index)
    try:
        cache_relpath = cache_path.relative_to(cache_dir).as_posix()
    except ValueError as error:
        raise HistoricalBacktestError("Cache path escaped the selected cache root.") from error
    manifest: dict[str, object] = {
        "contest_id": int(contest_id),
        "index": str(index),
        "cache_relpath": cache_relpath,
        "cache_exists": int(cache_path.is_file()),
        "stored_byte_size": 0,
        "stored_byte_sha256": "",
        "parse_status": "missing_cache",
        "parse_error": "",
    }
    values = _empty_statement_features()
    if not cache_path.is_file():
        return values, manifest
    try:
        raw = cache_path.read_bytes()
    except OSError as error:
        manifest["parse_status"] = "read_failed"
        manifest["parse_error"] = str(error)
        return values, manifest
    manifest["stored_byte_size"] = len(raw)
    manifest["stored_byte_sha256"] = hashlib.sha256(raw).hexdigest()
    if not raw:
        manifest["parse_status"] = "empty_file"
        return values, manifest
    if raw.lstrip().startswith(b"%PDF"):
        manifest["parse_status"] = "pdf_not_html"
        return values, manifest
    try:
        parsed = parse_problem_statement(raw.decode("utf-8", errors="replace"))
        manifest["parse_status"] = parsed.status
        manifest["parse_error"] = parsed.error
        values = build_statement_feature_values(parsed)
    except Exception as error:  # preserve the row and make the failure auditable
        manifest["parse_status"] = "parse_failed"
        manifest["parse_error"] = str(error)
        values = _empty_statement_features()
    if tuple(values) != STATEMENT_FEATURE_COLUMNS:
        raise HistoricalBacktestError(
            "Statement feature builder did not return the frozen 41-column schema."
        )
    if manifest["parse_status"] != "parsed":
        values = _empty_statement_features()
    return values, manifest


def build_prepared_dataset(
    model_table_path: Path,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild all 41 statement columns from current stored cache bytes."""
    if not model_table_path.is_file():
        raise HistoricalBacktestError(f"Model table does not exist: {model_table_path}")
    try:
        # Passing ``columns`` is intentional: post-result and unrelated fields never
        # enter memory in this study, even if the source table contains them.
        raw_table = pd.read_parquet(
            model_table_path, engine="pyarrow", columns=list(MODEL_TABLE_COLUMNS)
        )
    except Exception as error:
        raise HistoricalBacktestError(f"Cannot read model table: {error}") from error
    model_table = _normalize_model_table(raw_table)
    statement_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for row in model_table.itertuples(index=False):
        values, manifest = _statement_row(cache_dir, row.contest_id, row.index)
        statement_rows.append(values)
        manifest_rows.append(manifest)

    statement_frame = pd.DataFrame(statement_rows, columns=list(STATEMENT_FEATURE_COLUMNS))
    prepared = pd.concat(
        [model_table.reset_index(drop=True), statement_frame.reset_index(drop=True)], axis=1
    )
    expected_columns = (*MODEL_TABLE_COLUMNS, *STATEMENT_FEATURE_COLUMNS)
    if tuple(prepared.columns) != expected_columns:
        raise HistoricalBacktestError("Prepared dataset does not match the frozen schema.")
    manifest = pd.DataFrame(
        manifest_rows,
        columns=[
            "contest_id",
            "index",
            "cache_relpath",
            "cache_exists",
            "stored_byte_size",
            "stored_byte_sha256",
            "parse_status",
            "parse_error",
        ],
    )
    return prepared, manifest


def build_frozen_split(
    prepared: pd.DataFrame, config: HistoricalBacktestConfig
) -> pd.DataFrame:
    """Apply the existing complete-timestamp-bucket forward split."""
    experiment_config = ExperimentConfig(
        random_seed=config.random_seed,
        forward_time_split=SplitRatios(
            config.train_fraction,
            config.validation_fraction,
            config.test_fraction,
        ),
    )
    try:
        assignment = build_forward_time_split(
            prepared.loc[:, ["contest_id", "index", "start_time_seconds"]],
            experiment_config,
        )
    except SplitError as error:
        raise HistoricalBacktestError(str(error)) from error
    timestamp_lookup = prepared.set_index(["contest_id", "index"])["start_time_seconds"]
    keys = pd.MultiIndex.from_frame(assignment.loc[:, ["contest_id", "index"]])
    assignment.insert(
        2,
        "start_time_seconds",
        timestamp_lookup.reindex(keys).to_numpy(dtype="int64"),
    )
    validate_split_assignment(prepared, assignment)
    return assignment.sort_values(
        ["start_time_seconds", "contest_id", "index"], kind="mergesort"
    ).reset_index(drop=True)


def validate_split_assignment(prepared: pd.DataFrame, split: pd.DataFrame) -> None:
    """Require a one-to-one, complete, strictly chronological assignment."""
    required = {"contest_id", "index", "start_time_seconds", "split_name"}
    missing = sorted(required - set(split.columns))
    if missing:
        raise HistoricalBacktestError(f"Split assignment lacks columns: {missing}")
    if split.duplicated(["contest_id", "index"]).any():
        raise HistoricalBacktestError("Split assignment contains duplicate identifiers.")
    if prepared.duplicated(["contest_id", "index"]).any():
        raise HistoricalBacktestError("Prepared dataset contains duplicate identifiers.")
    prepared_keys = set(
        prepared.loc[:, ["contest_id", "index"]].itertuples(index=False, name=None)
    )
    split_keys = set(split.loc[:, ["contest_id", "index"]].itertuples(index=False, name=None))
    if prepared_keys != split_keys:
        raise HistoricalBacktestError("Prepared and split identifier sets differ.")
    if set(split["split_name"]) != set(SETTING_PARTITIONS):
        raise HistoricalBacktestError("Split must contain non-empty train, valid, and test rows.")
    time_check = split.loc[
        :, ["contest_id", "index", "start_time_seconds"]
    ].merge(
        prepared.loc[:, ["contest_id", "index", "start_time_seconds"]],
        on=["contest_id", "index"],
        how="inner",
        validate="one_to_one",
        suffixes=("_split", "_prepared"),
    )
    split_times = pd.to_numeric(
        time_check["start_time_seconds_split"], errors="coerce"
    )
    prepared_times = pd.to_numeric(
        time_check["start_time_seconds_prepared"], errors="coerce"
    )
    if split_times.isna().any() or prepared_times.isna().any() or not split_times.equals(
        prepared_times
    ):
        raise HistoricalBacktestError(
            "Split start times do not exactly match the prepared dataset."
        )
    time_sets = {
        name: set(
            pd.to_numeric(
                split.loc[split["split_name"].eq(name), "start_time_seconds"],
                errors="raise",
            ).astype("int64")
        )
        for name in SETTING_PARTITIONS
    }
    if any(time_sets[left] & time_sets[right] for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise HistoricalBacktestError("Equal start-time buckets cross split boundaries.")
    if not (
        max(time_sets["train"]) < min(time_sets["valid"])
        and max(time_sets["valid"]) < min(time_sets["test"])
    ):
        raise HistoricalBacktestError("Split is not strictly train < valid < test.")


SETTING_PARTITIONS: Final[tuple[str, str, str]] = ("train", "valid", "test")


def _joined(prepared: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    assignment = split.loc[:, ["contest_id", "index", "split_name"]]
    joined = prepared.merge(
        assignment,
        on=["contest_id", "index"],
        how="left",
        validate="one_to_one",
    )
    if joined["split_name"].isna().any():
        raise HistoricalBacktestError("Some prepared rows lack split assignments.")
    return joined


def _fit_and_predict(
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    features: Sequence[str],
    alpha: float,
) -> np.ndarray:
    """Fit the only allowed model and predict one held-out frame."""
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    pipeline.fit(train.loc[:, list(features)], train["rating"].to_numpy(dtype=float))
    return np.asarray(pipeline.predict(evaluate.loc[:, list(features)]), dtype=float)


def select_validation_alphas(
    prepared: pd.DataFrame,
    split: pd.DataFrame,
    config: HistoricalBacktestConfig,
    *,
    predictor: Callable[[pd.DataFrame, pd.DataFrame, Sequence[str], float], np.ndarray] = _fit_and_predict,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Select one alpha per setting using validation MAE only."""
    validate_split_assignment(prepared, split)
    joined = _joined(prepared, split)
    train = joined.loc[joined["split_name"].eq("train")].copy()
    valid = joined.loc[joined["split_name"].eq("valid")].copy()
    # Deliberately do not create a test frame or compute any test prediction here.
    rows: list[dict[str, object]] = []
    selected: dict[str, float] = {}
    for setting in SETTING_NAMES:
        columns = feature_columns(config, setting)
        for alpha in config.alpha_candidates:
            prediction = predictor(train, valid, columns, alpha)
            if prediction.shape != (len(valid),):
                raise HistoricalBacktestError("Predictor returned an unexpected shape.")
            mae = float(mean_absolute_error(valid["rating"], prediction))
            rows.append(
                {
                    "setting": setting,
                    "alpha": alpha,
                    "validation_mae": mae,
                    "train_rows": len(train),
                    "validation_rows": len(valid),
                }
            )
        candidates = [row for row in rows if row["setting"] == setting]
        winner = min(candidates, key=lambda row: (float(row["validation_mae"]), float(row["alpha"])))
        selected[setting] = float(winner["alpha"])
    metrics = pd.DataFrame(rows).sort_values(
        ["setting", "alpha"], kind="mergesort"
    ).reset_index(drop=True)
    return metrics, selected


def _temporary_path(target: Path) -> Path:
    return target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp{target.suffix}")


def _publish_exclusive(temp_path: Path, target: Path) -> None:
    try:
        os.link(temp_path, target)
    except FileExistsError as error:
        raise HistoricalBacktestError(f"Refusing to overwrite existing output: {target}") from error
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(path)
    try:
        with temp_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_exclusive(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, encoded)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(path)
    try:
        frame.to_csv(temp_path, index=False, encoding="utf-8", lineterminator="\n")
        _publish_exclusive(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_path(path)
    try:
        frame.to_parquet(temp_path, engine="pyarrow", index=False)
        _publish_exclusive(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _require_absent(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise HistoricalBacktestError(
            "Refusing to overwrite existing output(s): " + ", ".join(existing)
        )


def run_selection(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    model_table_path: Path = DEFAULT_MODEL_TABLE_PATH,
    cache_dir: Path | None = None,
    cache_manifest_path: Path = DEFAULT_CACHE_MANIFEST_PATH,
    output_dir: Path = DEFAULT_SELECTION_DIR,
) -> dict[str, Path]:
    """Prepare inputs and seal validation-only model selection artifacts."""
    paths = {
        "prepared_dataset": output_dir / PREPARED_FILENAME,
        "source_manifest": output_dir / SOURCE_MANIFEST_FILENAME,
        "split_assignment": output_dir / SPLIT_FILENAME,
        "validation_metrics": output_dir / VALIDATION_METRICS_FILENAME,
        "selection_lock": output_dir / SELECTION_LOCK_FILENAME,
        "selection_hash": output_dir / SELECTION_HASH_FILENAME,
    }
    _require_absent(list(paths.values()))
    config = load_backtest_config(config_path)
    selected_cache_dir = cache_dir if cache_dir is not None else config.cache_dir
    try:
        archive_report = verify_archive(selected_cache_dir, cache_manifest_path)
    except ArchiveError as error:
        raise HistoricalBacktestError(f"Cannot verify frozen cache: {error}") from error
    if archive_report.get("ok") is not True:
        raise HistoricalBacktestError(
            "Selected cache does not match the frozen manifest: "
            + json.dumps(archive_report, ensure_ascii=False, sort_keys=True)
        )
    prepared, manifest = build_prepared_dataset(model_table_path, selected_cache_dir)
    split = build_frozen_split(prepared, config)
    validation_metrics, selected = select_validation_alphas(prepared, split, config)

    _atomic_parquet(paths["prepared_dataset"], prepared)
    _atomic_csv(paths["source_manifest"], manifest)
    _atomic_parquet(paths["split_assignment"], split)
    _atomic_csv(paths["validation_metrics"], validation_metrics)
    artifact_hashes = {
        key: {
            "filename": paths[key].name,
            "sha256": sha256_file(paths[key]),
        }
        for key in (
            "prepared_dataset",
            "source_manifest",
            "split_assignment",
            "validation_metrics",
        )
    }
    lock: dict[str, Any] = {
        "schema_version": 1,
        "study_id": config.raw.get("study_id"),
        "config_sha256": config.sha256,
        "selection_scope": "validation_only",
        "test_evaluated": False,
        "model_family": "ridge",
        "target": config.target,
        "inputs": {
            "model_table_sha256": sha256_file(model_table_path),
            "cache_manifest_sha256": sha256_file(cache_manifest_path),
        },
        "feature_sets": {
            setting: {
                "columns": list(feature_columns(config, setting)),
                "selected_alpha": selected[setting],
            }
            for setting in SETTING_NAMES
        },
        "artifacts": artifact_hashes,
    }
    _atomic_json(paths["selection_lock"], lock)
    lock_hash = sha256_file(paths["selection_lock"])
    _atomic_bytes(paths["selection_hash"], f"{lock_hash}  {SELECTION_LOCK_FILENAME}\n".encode("ascii"))
    return paths


def _load_verified_selection(
    config: HistoricalBacktestConfig, selection_dir: Path
) -> tuple[Mapping[str, Any], pd.DataFrame, pd.DataFrame]:
    lock_path = selection_dir / SELECTION_LOCK_FILENAME
    sidecar_path = selection_dir / SELECTION_HASH_FILENAME
    if not lock_path.is_file() or not sidecar_path.is_file():
        raise HistoricalBacktestError("Selection lock or its hash sidecar is missing.")
    sidecar_parts = sidecar_path.read_text(encoding="ascii").strip().split()
    if len(sidecar_parts) != 2 or sidecar_parts[1] != SELECTION_LOCK_FILENAME:
        raise HistoricalBacktestError("Selection hash sidecar has an invalid format.")
    if sidecar_parts[0] != sha256_file(lock_path):
        raise HistoricalBacktestError("Selection lock hash mismatch.")
    try:
        lock_value = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HistoricalBacktestError(f"Cannot read selection lock: {error}") from error
    lock = _require_mapping(lock_value, "selection lock")
    if lock.get("config_sha256") != config.sha256:
        raise HistoricalBacktestError("Backtest config hash does not match the selection lock.")
    if lock.get("selection_scope") != "validation_only" or lock.get("test_evaluated") is not False:
        raise HistoricalBacktestError("Selection lock does not attest validation-only selection.")

    artifacts = _require_mapping(lock.get("artifacts"), "selection artifacts")
    resolved: dict[str, Path] = {}
    for key in (
        "prepared_dataset",
        "source_manifest",
        "split_assignment",
        "validation_metrics",
    ):
        item = _require_mapping(artifacts.get(key), f"selection artifact {key}")
        filename = item.get("filename")
        expected_hash = item.get("sha256")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise HistoricalBacktestError(f"Unsafe artifact filename for {key}.")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise HistoricalBacktestError(f"Invalid artifact hash for {key}.")
        path = selection_dir / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise HistoricalBacktestError(f"Selection artifact hash mismatch: {key}")
        resolved[key] = path

    feature_sets = _require_mapping(lock.get("feature_sets"), "locked feature sets")
    if (
        lock.get("schema_version") != 1
        or lock.get("study_id") != config.raw.get("study_id")
        or lock.get("model_family") != "ridge"
        or lock.get("target") != "rating"
    ):
        raise HistoricalBacktestError("Selection lock identity or model contract changed.")
    for setting in SETTING_NAMES:
        item = _require_mapping(feature_sets.get(setting), f"locked {setting}")
        if tuple(item.get("columns", ())) != feature_columns(config, setting):
            raise HistoricalBacktestError(f"Locked {setting} feature allowlist changed.")
        alpha = _require_number(item.get("selected_alpha"), f"locked {setting} alpha")
        if alpha not in config.alpha_candidates:
            raise HistoricalBacktestError(f"Locked {setting} alpha was not a candidate.")

    try:
        validation_metrics = pd.read_csv(resolved["validation_metrics"])
    except Exception as error:
        raise HistoricalBacktestError(
            f"Cannot read locked validation metrics: {error}"
        ) from error
    expected_metric_columns = (
        "setting",
        "alpha",
        "validation_mae",
        "train_rows",
        "validation_rows",
    )
    if tuple(validation_metrics.columns) != expected_metric_columns:
        raise HistoricalBacktestError("Locked validation metrics schema changed.")
    if (
        set(validation_metrics["setting"]) != set(SETTING_NAMES)
        or len(validation_metrics)
        != len(SETTING_NAMES) * len(config.alpha_candidates)
    ):
        raise HistoricalBacktestError("Locked validation metric settings changed.")
    for setting in SETTING_NAMES:
        setting_metrics = validation_metrics.loc[
            validation_metrics["setting"].eq(setting)
        ].copy()
        setting_metrics["alpha"] = pd.to_numeric(
            setting_metrics["alpha"], errors="coerce"
        )
        setting_metrics["validation_mae"] = pd.to_numeric(
            setting_metrics["validation_mae"], errors="coerce"
        )
        if (
            len(setting_metrics) != len(config.alpha_candidates)
            or setting_metrics["alpha"].duplicated().any()
            or set(setting_metrics["alpha"]) != set(config.alpha_candidates)
            or not np.isfinite(setting_metrics["validation_mae"]).all()
            or (setting_metrics["validation_mae"] < 0).any()
        ):
            raise HistoricalBacktestError(
                f"Locked validation candidates changed for {setting}."
            )
        winner = setting_metrics.sort_values(
            ["validation_mae", "alpha"], kind="mergesort"
        ).iloc[0]
        locked_alpha = float(
            _require_mapping(feature_sets[setting], setting)["selected_alpha"]
        )
        if float(winner["alpha"]) != locked_alpha:
            raise HistoricalBacktestError(
                f"Locked {setting} alpha does not match the validation winner."
            )

    try:
        prepared = pd.read_parquet(resolved["prepared_dataset"], engine="pyarrow")
        split = pd.read_parquet(resolved["split_assignment"], engine="pyarrow")
    except Exception as error:
        raise HistoricalBacktestError(f"Cannot read sealed selection artifact: {error}") from error
    if tuple(prepared.columns) != (*MODEL_TABLE_COLUMNS, *STATEMENT_FEATURE_COLUMNS):
        raise HistoricalBacktestError("Sealed prepared dataset schema changed.")
    validate_split_assignment(prepared, split)
    return lock, prepared, split


def compute_regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    """Compute the three frozen test metrics without emitting non-finite JSON."""
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(mean_squared_error(actual, predicted) ** 0.5)
    r2 = float(r2_score(actual, predicted)) if len(actual) >= 2 else math.nan
    return {"mae": mae, "rmse": rmse, "r2": r2 if math.isfinite(r2) else None}


def evaluate_locked_test(
    prepared: pd.DataFrame,
    split: pd.DataFrame,
    config: HistoricalBacktestConfig,
    locked_alphas: Mapping[str, float],
    *,
    predictor: Callable[[pd.DataFrame, pd.DataFrame, Sequence[str], float], np.ndarray] = _fit_and_predict,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Refit and call test prediction exactly once for each locked setting."""
    validate_split_assignment(prepared, split)
    joined = _joined(prepared, split)
    fit_frame = joined.loc[joined["split_name"].isin(["train", "valid"])].copy()
    test_frame = joined.loc[joined["split_name"].eq("test")].copy()
    predictions: list[pd.DataFrame] = []
    metrics: dict[str, Any] = {"test_rows": len(test_frame), "settings": {}}
    actual = test_frame["rating"].to_numpy(dtype=float)
    for setting in SETTING_NAMES:
        alpha = float(locked_alphas[setting])
        # Exactly one call per setting: no test-set alpha loop is permitted.
        predicted = predictor(fit_frame, test_frame, feature_columns(config, setting), alpha)
        if predicted.shape != (len(test_frame),):
            raise HistoricalBacktestError("Predictor returned an unexpected test shape.")
        setting_metrics = compute_regression_metrics(actual, predicted)
        metrics["settings"][setting] = {"alpha": alpha, **setting_metrics}
        frame = test_frame.loc[
            :, ["contest_id", "index", "start_time_seconds", "rating"]
        ].copy()
        frame.insert(0, "setting", setting)
        frame["prediction"] = predicted
        frame["absolute_error"] = np.abs(actual - predicted)
        predictions.append(frame)
    metrics["primary_mae_minus_comparator_mae"] = (
        metrics["settings"]["primary"]["mae"]
        - metrics["settings"]["comparator"]["mae"]
    )
    result = pd.concat(predictions, ignore_index=True).sort_values(
        ["setting", "start_time_seconds", "contest_id", "index"], kind="mergesort"
    ).reset_index(drop=True)
    return metrics, result


def paired_contest_cluster_bootstrap(
    predictions: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Bootstrap the paired primary-minus-comparator MAE by contest."""
    required = {"setting", "contest_id", "index", "absolute_error"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise HistoricalBacktestError(f"Predictions lack bootstrap columns: {missing}")
    pivot = predictions.pivot(
        index=["contest_id", "index"], columns="setting", values="absolute_error"
    )
    if tuple(sorted(pivot.columns)) != tuple(sorted(SETTING_NAMES)) or pivot.isna().any().any():
        raise HistoricalBacktestError("Bootstrap predictions are not completely paired.")
    pivot = pivot.reset_index().sort_values(["contest_id", "index"], kind="mergesort")
    cluster_rows = []
    for contest_id, group in pivot.groupby("contest_id", sort=True):
        cluster_rows.append(
            (
                contest_id,
                len(group),
                float(group["primary"].sum()),
                float(group["comparator"].sum()),
            )
        )
    if not cluster_rows:
        raise HistoricalBacktestError("Cannot bootstrap an empty test set.")
    sizes = np.asarray([row[1] for row in cluster_rows], dtype=float)
    primary_sums = np.asarray([row[2] for row in cluster_rows], dtype=float)
    comparator_sums = np.asarray([row[3] for row in cluster_rows], dtype=float)
    rng = np.random.Generator(np.random.PCG64(seed))
    statistics = np.empty(resamples, dtype=float)
    cluster_count = len(cluster_rows)
    for offset in range(resamples):
        draw = rng.integers(0, cluster_count, size=cluster_count)
        denominator = float(sizes[draw].sum())
        statistics[offset] = (
            float(primary_sums[draw].sum()) - float(comparator_sums[draw].sum())
        ) / denominator
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(
        statistics, [tail, 1.0 - tail], method="linear"
    ).tolist()
    point = float((primary_sums.sum() - comparator_sums.sum()) / sizes.sum())
    return {
        "method": "paired_contest_cluster_bootstrap",
        "statistic": "primary_mae_minus_comparator_mae",
        "point_estimate": point,
        "confidence_level": confidence_level,
        "confidence_interval": {"lower": float(lower), "upper": float(upper)},
        "resamples": resamples,
        "random_seed": seed,
        "rng": "numpy_pcg64",
        "quantile_method": "linear",
        "cluster_count": cluster_count,
    }


def primary_top_errors(predictions: pd.DataFrame, count: int) -> pd.DataFrame:
    """Return the automatic, deterministically tie-broken primary top errors."""
    primary = predictions.loc[predictions["setting"].eq("primary")].copy()
    primary = primary.sort_values(
        ["absolute_error", "start_time_seconds", "contest_id", "index"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).head(count)
    primary.insert(0, "error_rank", np.arange(1, len(primary) + 1, dtype=int))
    return primary.reset_index(drop=True)


def run_test(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    selection_dir: Path = DEFAULT_SELECTION_DIR,
    output_dir: Path = DEFAULT_TEST_DIR,
) -> dict[str, Path]:
    """Verify selection and perform the single locked test evaluation."""
    paths = {
        "metrics": output_dir / TEST_METRICS_FILENAME,
        "predictions": output_dir / TEST_PREDICTIONS_FILENAME,
        "bootstrap": output_dir / BOOTSTRAP_FILENAME,
        "top10": output_dir / TOP10_FILENAME,
    }
    _require_absent(list(paths.values()))
    config = load_backtest_config(config_path)
    lock, prepared, split = _load_verified_selection(config, selection_dir)
    locked_features = _require_mapping(lock["feature_sets"], "locked feature sets")
    locked_alphas = {
        setting: float(_require_mapping(locked_features[setting], setting)["selected_alpha"])
        for setting in SETTING_NAMES
    }
    metrics, predictions = evaluate_locked_test(
        prepared, split, config, locked_alphas
    )
    bootstrap = paired_contest_cluster_bootstrap(
        predictions,
        resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
        confidence_level=config.confidence_level,
    )
    top10 = primary_top_errors(predictions, config.top_count)
    metrics["config_sha256"] = config.sha256
    metrics["selection_lock_sha256"] = sha256_file(
        selection_dir / SELECTION_LOCK_FILENAME
    )

    _atomic_json(paths["metrics"], metrics)
    _atomic_csv(paths["predictions"], predictions)
    _atomic_json(paths["bootstrap"], bootstrap)
    _atomic_csv(paths["top10"], top10)
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen historical statement-only backtest."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser(
        "select", help="Prepare data and select Ridge alphas on validation only."
    )
    select_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    select_parser.add_argument("--model-table", type=Path, default=DEFAULT_MODEL_TABLE_PATH)
    select_parser.add_argument("--cache-dir", type=Path, default=None)
    select_parser.add_argument(
        "--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST_PATH
    )
    select_parser.add_argument("--output-dir", type=Path, default=DEFAULT_SELECTION_DIR)

    test_parser = subparsers.add_parser(
        "test", help="Verify the selection lock and evaluate the frozen test once."
    )
    test_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    test_parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    test_parser.add_argument("--output-dir", type=Path, default=DEFAULT_TEST_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the two-stage command-line interface."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "select":
            paths = run_selection(
                config_path=args.config,
                model_table_path=args.model_table,
                cache_dir=args.cache_dir,
                cache_manifest_path=args.cache_manifest,
                output_dir=args.output_dir,
            )
            print(f"Sealed validation selection: {paths['selection_lock']}")
        else:
            paths = run_test(
                config_path=args.config,
                selection_dir=args.selection_dir,
                output_dir=args.output_dir,
            )
            print(f"Wrote locked test metrics: {paths['metrics']}")
    except (HistoricalBacktestError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
