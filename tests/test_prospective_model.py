"""Tests for protocol-locked prospective model freezing and prediction."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import prospective_model
from cf_diff.prospective_model import (
    PREDICTION_COLUMNS,
    ProspectiveModelError,
    freeze_prospective_model,
    predict_prospective,
    verify_frozen_model,
)
from cf_diff.statement_features import STATEMENT_FEATURE_COLUMNS


DRAFT_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "prospective_protocol_v2.json"
)
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
SOURCE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
CONTEST_START_TEXT = "2026-08-15T01:00:00Z"
CUTOFF_SECONDS = int(
    datetime(2026, 8, 15, tzinfo=timezone.utc).timestamp()
)


def _frozen_protocol(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(DRAFT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["status"] = "frozen"
    protocol["protocol_frozen_at_utc"] = "2026-08-01T00:00:00Z"
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _protocol_columns(protocol_path: Path) -> tuple[list[str], list[str]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    bundle = protocol["model_bundle"]
    return (
        list(bundle["primary_feature_columns"]),
        list(bundle["comparator_feature_columns"]),
    )


def _historical_tables(protocol_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary, _ = _protocol_columns(protocol_path)
    keys = [(100, "A"), (100, "B1"), (101, "C2"), (999, "Z9")]
    model = pd.DataFrame(
        {
            "contest_id": [key[0] for key in keys],
            "index": [key[1] for key in keys],
            "rating": [900.0, 1200.0, 1700.0, 99999.0],
            "start_time_seconds": [
                CUTOFF_SECONDS - 300,
                CUTOFF_SECONDS - 200,
                CUTOFF_SECONDS - 1,
                CUTOFF_SECONDS,
            ],
            "index_rank": [99, 99, 99, 99],
            "index_number": [99, 99, 99, 99],
            "unlisted_noise": [-1, -2, -3, -4],
        }
    )
    statements: dict[str, object] = {
        "contest_id": [key[0] for key in keys],
        "index": [key[1] for key in keys],
    }
    for column_number, column in enumerate(primary, start=1):
        if column in {"index_rank", "index_number"}:
            continue
        statements[column] = [
            float(column_number + row_number)
            for row_number in range(len(keys))
        ]
    return model, pd.DataFrame(statements)


def _write_training_tables(
    tmp_path: Path,
    protocol_path: Path,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model, statements = _historical_tables(protocol_path)
    model_path = tmp_path / "model_table.parquet"
    statements_path = tmp_path / "statement_features.parquet"
    model.to_parquet(model_path, index=False)
    statements.to_parquet(statements_path, index=False)
    return model_path, statements_path


def _freeze(tmp_path: Path) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    protocol_path = _frozen_protocol(tmp_path)
    model_table_path, statement_features_path = _write_training_tables(
        tmp_path,
        protocol_path,
    )
    model_path = tmp_path / "bundle.json"
    manifest_path = tmp_path / "manifest.json"
    freeze_prospective_model(
        protocol_path=protocol_path,
        model_table_path=model_table_path,
        statement_features_path=statement_features_path,
        model_path=model_path,
        manifest_path=manifest_path,
        requirements_path=REQUIREMENTS_PATH,
        source_commit=SOURCE_COMMIT,
        frozen_at_utc="2026-08-02T00:00:00Z",
    )
    return protocol_path, model_path, manifest_path


def _prediction_input() -> pd.DataFrame:
    payload: dict[str, object] = {
        "contest_id": ["3000", "3000"],
        "index": ["A", "D2"],
    }
    for column_number, column in enumerate(
        STATEMENT_FEATURE_COLUMNS,
        start=1,
    ):
        payload[column] = [
            float(column_number + 10),
            float(column_number + 20),
        ]
    return pd.DataFrame(
        payload,
        columns=["contest_id", "index", *STATEMENT_FEATURE_COLUMNS],
    )


def _write_input(path: Path, frame: pd.DataFrame) -> None:
    if path.suffix == ".csv":
        frame.to_csv(path, index=False, lineterminator="\n")
    else:
        frame.to_parquet(path, index=False)


def _write_capture_sidecar(
    *,
    path: Path,
    protocol_path: Path,
    input_path: Path,
    frame: pd.DataFrame,
    declared_indices: list[str] | None = None,
) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    indices = (
        declared_indices
        if declared_indices is not None
        else frame["index"].astype(str).tolist()
    )
    row_count = len(indices)
    problems = [
        {
            "index": index,
            "url": f"https://codeforces.com/problemset/problem/3000/{index}",
            "fetch_started_at_utc": "2026-08-15T01:00:10Z",
            "fetch_completed_at_utc": "2026-08-15T01:00:20Z",
            "http_status": 200,
            "fetch_status": "fetched",
            "parse_status": "parsed",
            "raw_html_sha256": hashlib.sha256(index.encode()).hexdigest(),
            "raw_path": f"prospective/raw/3000/3000_{index}.html",
            "error": "",
        }
        for index in indices
    ]
    sidecar = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "contest_id": "3000",
        "contest_start_utc": CONTEST_START_TEXT,
        "contest_start_source": "explicit_cli_argument_unverified_at_t0",
        "lock_deadline_utc": "2026-08-15T01:30:00Z",
        "capture_started_at_utc": "2026-08-15T01:00:10Z",
        "capture_completed_at_utc": "2026-08-15T01:01:00Z",
        "requested_indices": indices,
        "request_policy": {
            "source": "direct_public_problem_statement_pages",
            "metadata_api_used": False,
            "accept_language": "en-US,en;q=0.9",
        },
        "problems": problems,
        "output": {
            "path": input_path.as_posix(),
            "columns": list(frame.columns),
            "row_count": row_count,
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        },
        "error": "",
    }
    path.write_text(
        json.dumps(sidecar, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _prediction_paths(
    tmp_path: Path,
    protocol_path: Path,
    *,
    suffix: str = ".csv",
    frame: pd.DataFrame | None = None,
    declared_indices: list[str] | None = None,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / f"prospective-input{suffix}"
    sidecar_path = tmp_path / "prospective-input.capture.json"
    input_frame = _prediction_input() if frame is None else frame
    _write_input(input_path, input_frame)
    _write_capture_sidecar(
        path=sidecar_path,
        protocol_path=protocol_path,
        input_path=input_path,
        frame=input_frame,
        declared_indices=declared_indices,
    )
    return input_path, sidecar_path


def _predict(
    *,
    protocol_path: Path,
    model_path: Path,
    manifest_path: Path,
    input_path: Path,
    sidecar_path: Path,
    output_path: Path,
    contest_start: str = CONTEST_START_TEXT,
    created_at: str = "2026-08-15T01:02:00Z",
) -> Path:
    return predict_prospective(
        protocol_path=protocol_path,
        model_path=model_path,
        manifest_path=manifest_path,
        input_path=input_path,
        capture_sidecar_path=sidecar_path,
        output_path=output_path,
        contest_start_utc=contest_start,
        prediction_created_at_utc=created_at,
    )


def test_freeze_writes_locked_json_and_full_provenance(tmp_path: Path) -> None:
    protocol_path, model_path, manifest_path = _freeze(tmp_path)
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert artifact["estimator"] == {
        "alpha": 1.0,
        "copy_X": True,
        "fit_intercept": True,
        "name": "Ridge",
        "positive": False,
        "solver": "svd",
    }
    assert artifact["protocol_sha256"] == hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    assert manifest["model_artifact_sha256"] == hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["training_cutoff_utc"] == "2026-08-15T00:00:00Z"
    assert manifest["training_row_count"] == 3
    assert set(manifest["runtime"]) == {
        "python",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit_learn",
    }
    assert manifest["dependency_spec"]["sha256"] == hashlib.sha256(
        REQUIREMENTS_PATH.read_bytes()
    ).hexdigest()
    verified = verify_frozen_model(
        protocol_path=protocol_path,
        model_path=model_path,
        manifest_path=manifest_path,
    )
    assert verified["training_row_count"] == 3


def test_freeze_only_uses_allowlisted_pre_cutoff_features(tmp_path: Path) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    model, statements = _historical_tables(protocol_path)
    artifacts: list[dict[str, object]] = []
    for number in range(2):
        directory = tmp_path / str(number)
        directory.mkdir()
        model_variant = model.copy()
        statements_variant = statements.copy()
        if number:
            model_variant["unlisted_noise"] = [9e20, 8e20, 7e20, 6e20]
            model_variant.loc[3, "rating"] = -99999.0
            statements_variant["unlisted_noise"] = ["x", "y", "z", "future"]
            statements_variant.loc[
                3,
                list(STATEMENT_FEATURE_COLUMNS),
            ] = 1e30
        model_table_path = directory / "model.parquet"
        statements_path = directory / "statements.parquet"
        model_variant.to_parquet(model_table_path, index=False)
        statements_variant.to_parquet(statements_path, index=False)
        artifact_path = directory / "bundle.json"
        freeze_prospective_model(
            protocol_path=protocol_path,
            model_table_path=model_table_path,
            statement_features_path=statements_path,
            model_path=artifact_path,
            manifest_path=directory / "manifest.json",
            requirements_path=REQUIREMENTS_PATH,
            source_commit=SOURCE_COMMIT,
            frozen_at_utc="2026-08-02T00:00:00Z",
        )
        artifacts.append(json.loads(artifact_path.read_text(encoding="utf-8")))

    assert artifacts[0] == artifacts[1]


def test_csv_and_parquet_prediction_are_deterministic_and_exact(
    tmp_path: Path,
) -> None:
    protocol_path, model_path, manifest_path = _freeze(tmp_path)
    results: list[pd.DataFrame] = []
    for suffix in (".csv", ".parquet"):
        directory = tmp_path / suffix[1:]
        directory.mkdir()
        input_path, sidecar_path = _prediction_paths(
            directory,
            protocol_path,
            suffix=suffix,
        )
        output_path = directory / f"predictions{suffix}"
        _predict(
            protocol_path=protocol_path,
            model_path=model_path,
            manifest_path=manifest_path,
            input_path=input_path,
            sidecar_path=sidecar_path,
            output_path=output_path,
        )
        result = (
            pd.read_csv(output_path)
            if suffix == ".csv"
            else pd.read_parquet(output_path)
        )
        assert tuple(result.columns) == PREDICTION_COLUMNS
        assert result["feature_row_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
        assert result["input_file_sha256"].nunique() == 1
        assert result["capture_sidecar_sha256"].nunique() == 1
        assert result["freeze_manifest_sha256"].nunique() == 1
        results.append(result)

    comparable = [
        "contest_id",
        "index",
        "primary_prediction",
        "comparator_prediction",
        "feature_row_sha256",
    ]
    pd.testing.assert_frame_equal(
        results[0][comparable].astype({"contest_id": str}),
        results[1][comparable].astype({"contest_id": str}),
    )


def test_numeric_index_and_blank_missing_value_are_preserved(tmp_path: Path) -> None:
    protocol_path, model_path, manifest_path = _freeze(tmp_path)
    frame = _prediction_input().iloc[[0]].copy()
    frame["index"] = "01"
    frame["time_limit_ms"] = ""
    input_path, sidecar_path = _prediction_paths(
        tmp_path / "prediction",
        protocol_path,
        frame=frame,
    )
    output_path = tmp_path / "prediction.csv"

    _predict(
        protocol_path=protocol_path,
        model_path=model_path,
        manifest_path=manifest_path,
        input_path=input_path,
        sidecar_path=sidecar_path,
        output_path=output_path,
    )

    result = pd.read_csv(output_path, dtype={"index": "string"})
    assert result.loc[0, "index"] == "01"


@pytest.mark.parametrize(
    ("suffix", "column", "reader"),
    [
        (".csv", "rating", "read_csv"),
        (".parquet", "submissionCount", "read_parquet"),
    ],
)
def test_forbidden_schema_is_rejected_before_values_are_loaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    column: str,
    reader: str,
) -> None:
    protocol_path, model_path, manifest_path = _freeze(tmp_path)
    frame = _prediction_input()
    frame[column] = 99999
    input_path = tmp_path / f"forbidden{suffix}"
    _write_input(input_path, frame)
    calls = {"count": 0}

    def fail_if_called(*args: object, **kwargs: object) -> pd.DataFrame:
        calls["count"] += 1
        raise AssertionError("row-value reader must not be called")

    monkeypatch.setattr(prospective_model.pd, reader, fail_if_called)
    with pytest.raises(ProspectiveModelError, match="forbidden"):
        _predict(
            protocol_path=protocol_path,
            model_path=model_path,
            manifest_path=manifest_path,
            input_path=input_path,
            sidecar_path=tmp_path / "unused.json",
            output_path=tmp_path / "unused-output.csv",
        )
    assert calls["count"] == 0


@pytest.mark.parametrize("extra_column", ["name", "index_rank", "audit_note"])
def test_exact_schema_rejects_all_extra_columns(
    tmp_path: Path,
    extra_column: str,
) -> None:
    protocol_path, model_path, manifest_path = _freeze(tmp_path)
    frame = _prediction_input()
    frame[extra_column] = "not model input"
    input_path = tmp_path / "extra.csv"
    frame.to_csv(input_path, index=False)

    with pytest.raises(ProspectiveModelError, match="exact frozen schema"):
        _predict(
            protocol_path=protocol_path,
            model_path=model_path,
            manifest_path=manifest_path,
            input_path=input_path,
            sidecar_path=tmp_path / "unused.json",
            output_path=tmp_path / "unused-output.csv",
        )


def test_exact_schema_rejects_reordered_and_duplicate_headers(
    tmp_path: Path,
) -> None:
    protocol_path, model_path, manifest_path = _freeze(tmp_path)
    reordered = _prediction_input()
    columns = list(reordered.columns)
    columns[2], columns[3] = columns[3], columns[2]
    reordered_path = tmp_path / "reordered.csv"
    reordered.loc[:, columns].to_csv(reordered_path, index=False)
    with pytest.raises(ProspectiveModelError, match="column order"):
        _predict(
            protocol_path=protocol_path,
            model_path=model_path,
            manifest_path=manifest_path,
            input_path=reordered_path,
            sidecar_path=tmp_path / "unused.json",
            output_path=tmp_path / "unused-output.csv",
        )

    duplicate_path = tmp_path / "duplicate.csv"
    duplicate_path.write_text("contest_id,index,index\n3000,A,A\n", encoding="utf-8")
    with pytest.raises(ProspectiveModelError, match="duplicate columns"):
        _predict(
            protocol_path=protocol_path,
            model_path=model_path,
            manifest_path=manifest_path,
            input_path=duplicate_path,
            sidecar_path=tmp_path / "unused.json",
            output_path=tmp_path / "second-output.csv",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("empty", "at least one row"),
        ("multi_contest", "exactly one contest"),
        ("duplicate_key", "duplicate normalized keys"),
    ],
)
def test_prediction_requires_one_complete_unique_contest(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    protocol_path, model_path, manifest_path = _freeze(tmp_path)
    frame = _prediction_input()
    declared_indices: list[str] | None = None
    if mutation == "empty":
        frame = frame.iloc[0:0]
        declared_indices = ["A"]
    elif mutation == "multi_contest":
        frame.loc[1, "contest_id"] = "3001"
    else:
        frame.loc[1, "index"] = "A"
        declared_indices = ["A", "B"]
    directory = tmp_path / "prediction"
    directory.mkdir()
    input_path, sidecar_path = _prediction_paths(
        directory,
        protocol_path,
        frame=frame,
        declared_indices=declared_indices,
    )

    with pytest.raises(ProspectiveModelError, match=message):
        _predict(
            protocol_path=protocol_path,
            model_path=model_path,
            manifest_path=manifest_path,
            input_path=input_path,
            sidecar_path=sidecar_path,
            output_path=tmp_path / "output.csv",
        )


@pytest.mark.parametrize("bad_value", ["not-a-number", "inf", "-inf"])
def test_prediction_rejects_malformed_or_infinite_numeric_values(
    tmp_path: Path,
    bad_value: str,
) -> None:
    protocol_path, model_path, manifest_path = _freeze(tmp_path)
    frame = _prediction_input()
    frame["statement_char_len"] = frame["statement_char_len"].astype(object)
    frame.loc[0, "statement_char_len"] = bad_value
    directory = tmp_path / "prediction"
    directory.mkdir()
    input_path, sidecar_path = _prediction_paths(
        directory,
        protocol_path,
        frame=frame,
    )

    with pytest.raises(ProspectiveModelError, match="malformed|infinite"):
        _predict(
            protocol_path=protocol_path,
            model_path=model_path,
            manifest_path=manifest_path,
            input_path=input_path,
            sidecar_path=sidecar_path,
            output_path=tmp_path / "output.csv",
        )


def test_prediction_never_overwrites_existing_output(tmp_path: Path) -> None:
    protocol_path, model_path, manifest_path = _freeze(tmp_path)
    output_path = tmp_path / "predictions.csv"
    output_path.write_text("keep", encoding="utf-8")

    with pytest.raises(ProspectiveModelError, match="will not be overwritten"):
        _predict(
            protocol_path=protocol_path,
            model_path=model_path,
            manifest_path=manifest_path,
            input_path=tmp_path / "unused.csv",
            sidecar_path=tmp_path / "unused.json",
            output_path=output_path,
        )
    assert output_path.read_text(encoding="utf-8") == "keep"


def test_tampered_model_manifest_and_sidecar_are_rejected(tmp_path: Path) -> None:
    protocol_path, model_path, manifest_path = _freeze(tmp_path)
    directory = tmp_path / "prediction"
    directory.mkdir()
    input_path, sidecar_path = _prediction_paths(directory, protocol_path)

    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    artifact["primary_model"]["intercept"] += 1
    model_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ProspectiveModelError, match="SHA-256"):
        _predict(
            protocol_path=protocol_path,
            model_path=model_path,
            manifest_path=manifest_path,
            input_path=input_path,
            sidecar_path=sidecar_path,
            output_path=tmp_path / "output-a.csv",
        )

    protocol_path, model_path, manifest_path = _freeze(tmp_path / "second")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training_cutoff_utc"] = "2026-08-14T00:00:00Z"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ProspectiveModelError, match="training cutoff"):
        verify_frozen_model(
            protocol_path=protocol_path,
            model_path=model_path,
            manifest_path=manifest_path,
        )

    directory = tmp_path / "third"
    directory.mkdir()
    protocol_path, model_path, manifest_path = _freeze(directory)
    input_path, sidecar_path = _prediction_paths(
        directory / "prediction",
        protocol_path,
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["output"]["sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(ProspectiveModelError, match="input SHA-256"):
        _predict(
            protocol_path=protocol_path,
            model_path=model_path,
            manifest_path=manifest_path,
            input_path=input_path,
            sidecar_path=sidecar_path,
            output_path=directory / "output.csv",
        )


@pytest.mark.parametrize(
    ("contest_start", "created", "message"),
    [
        ("2026-08-14T23:59:59Z", "2026-08-15T00:00:00Z", "outside"),
        (CONTEST_START_TEXT, "2026-08-15T00:59:59Z", "on or after"),
        (CONTEST_START_TEXT, "2026-08-15T01:30:01Z", "lock deadline"),
    ],
)
def test_prediction_enforces_cohort_and_recording_times(
    tmp_path: Path,
    contest_start: str,
    created: str,
    message: str,
) -> None:
    protocol_path, model_path, manifest_path = _freeze(tmp_path)
    with pytest.raises(ProspectiveModelError, match=message):
        _predict(
            protocol_path=protocol_path,
            model_path=model_path,
            manifest_path=manifest_path,
            input_path=tmp_path / "unused.csv",
            sidecar_path=tmp_path / "unused.json",
            output_path=tmp_path / "output.csv",
            contest_start=contest_start,
            created_at=created,
        )


@pytest.mark.parametrize(
    ("source_commit", "frozen_at", "message"),
    [
        ("abc123", "2026-08-02T00:00:00Z", "40-character"),
        (SOURCE_COMMIT, "2026-07-31T23:59:59Z", "predate"),
        (SOURCE_COMMIT, "2026-08-15T00:00:00Z", "before"),
    ],
)
def test_invalid_freeze_metadata_leaves_no_artifacts(
    tmp_path: Path,
    source_commit: str,
    frozen_at: str,
    message: str,
) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    model_table_path, statements_path = _write_training_tables(
        tmp_path,
        protocol_path,
    )
    model_path = tmp_path / "bundle.json"
    manifest_path = tmp_path / "manifest.json"

    with pytest.raises(ProspectiveModelError, match=message):
        freeze_prospective_model(
            protocol_path=protocol_path,
            model_table_path=model_table_path,
            statement_features_path=statements_path,
            model_path=model_path,
            manifest_path=manifest_path,
            requirements_path=REQUIREMENTS_PATH,
            source_commit=source_commit,
            frozen_at_utc=frozen_at,
        )
    assert not model_path.exists()
    assert not manifest_path.exists()


def test_freeze_rejects_draft_protocol_and_existing_artifacts(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "bundle.json"
    manifest_path = tmp_path / "manifest.json"
    with pytest.raises(ProspectiveModelError, match="status must be 'frozen'"):
        freeze_prospective_model(
            protocol_path=DRAFT_PROTOCOL_PATH,
            model_table_path=tmp_path / "unused.parquet",
            statement_features_path=tmp_path / "unused-statements.parquet",
            model_path=model_path,
            manifest_path=manifest_path,
            requirements_path=REQUIREMENTS_PATH,
            source_commit=SOURCE_COMMIT,
            frozen_at_utc="2026-08-02T00:00:00Z",
        )
    assert not model_path.exists()
    assert not manifest_path.exists()

    model_path.write_text("keep-model", encoding="utf-8")
    manifest_path.write_text("keep-manifest", encoding="utf-8")
    with pytest.raises(ProspectiveModelError, match="will not be overwritten"):
        freeze_prospective_model(
            protocol_path=DRAFT_PROTOCOL_PATH,
            model_table_path=tmp_path / "unused.parquet",
            statement_features_path=tmp_path / "unused-statements.parquet",
            model_path=model_path,
            manifest_path=manifest_path,
            requirements_path=REQUIREMENTS_PATH,
            source_commit=SOURCE_COMMIT,
            frozen_at_utc="2026-08-02T00:00:00Z",
        )
    assert model_path.read_text(encoding="utf-8") == "keep-model"
    assert manifest_path.read_text(encoding="utf-8") == "keep-manifest"


def test_freeze_requires_one_to_one_statement_join(tmp_path: Path) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    model, statements = _historical_tables(protocol_path)
    statements = statements.loc[
        ~((statements["contest_id"] == 100) & (statements["index"] == "B1"))
    ]
    model_path = tmp_path / "model.parquet"
    statements_path = tmp_path / "statements.parquet"
    model.to_parquet(model_path, index=False)
    statements.to_parquet(statements_path, index=False)

    with pytest.raises(ProspectiveModelError, match="missing"):
        freeze_prospective_model(
            protocol_path=protocol_path,
            model_table_path=model_path,
            statement_features_path=statements_path,
            model_path=tmp_path / "bundle.json",
            manifest_path=tmp_path / "manifest.json",
            requirements_path=REQUIREMENTS_PATH,
            source_commit=SOURCE_COMMIT,
            frozen_at_utc="2026-08-02T00:00:00Z",
        )
