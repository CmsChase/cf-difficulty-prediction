"""Tests for the immutable prospective model bundle."""

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

from cf_diff.prospective_model import (
    PREDICTION_COLUMNS,
    ProspectiveModelError,
    freeze_prospective_model,
    predict_prospective,
    verify_frozen_model,
)


PROTOCOL_PATH = PROJECT_ROOT / "configs" / "prospective_protocol_v1.json"
CUTOFF_SECONDS = int(
    datetime(2026, 7, 12, tzinfo=timezone.utc).timestamp()
)


def _protocol_columns() -> tuple[list[str], list[str]]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    bundle = protocol["model_bundle"]
    return (
        list(bundle["primary_feature_columns"]),
        list(bundle["comparator_feature_columns"]),
    )


def _historical_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    primary, _ = _protocol_columns()
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
            # These deliberately disagree with the indices.  Training derives them.
            "index_rank": [99, 99, 99, 99],
            "index_number": [99, 99, 99, 99],
            "unlisted_noise": [-1, -2, -3, -4],
        }
    )
    statements: dict[str, object] = {
        "contest_id": [key[0] for key in keys],
        "index": [key[1] for key in keys],
        "statement_fetch_status": ["cached"] * len(keys),
    }
    for column_number, column in enumerate(primary, start=1):
        if column in {"index_rank", "index_number"}:
            continue
        statements[column] = [
            float(column_number + row_number)
            for row_number in range(len(keys))
        ]
    return model, pd.DataFrame(statements)


def _write_training_tables(tmp_path: Path) -> tuple[Path, Path]:
    model, statements = _historical_tables()
    model_path = tmp_path / "model_table.parquet"
    statements_path = tmp_path / "statement_features.parquet"
    model.to_parquet(model_path, index=False)
    statements.to_parquet(statements_path, index=False)
    return model_path, statements_path


def _freeze(tmp_path: Path, *, suffix: str = "") -> tuple[Path, Path]:
    model_table_path, statement_features_path = _write_training_tables(tmp_path)
    model_path = tmp_path / f"bundle{suffix}.json"
    manifest_path = tmp_path / f"manifest{suffix}.json"
    freeze_prospective_model(
        protocol_path=PROTOCOL_PATH,
        model_table_path=model_table_path,
        statement_features_path=statement_features_path,
        model_path=model_path,
        manifest_path=manifest_path,
        source_commit="0123456789abcdef",
        frozen_at_utc="2026-07-10T08:00:00Z",
    )
    return model_path, manifest_path


def _prediction_input() -> pd.DataFrame:
    primary, _ = _protocol_columns()
    payload: dict[str, object] = {
        "contest_id": [3000, 3000],
        "index": ["A", "D2"],
        "statement_fetch_status": ["cached", "fetched"],
        "statement_parse_status": ["parsed", "parsed"],
        "statement_error": ["", ""],
        "name": ["One", "Two"],
        "url": ["https://example.test/one", "https://example.test/two"],
        # These are ignored and replaced from index.
        "index_rank": [1000, 1000],
        "index_number": [1000, 1000],
        "irrelevant_audit_note": ["first", "second"],
    }
    for column_number, column in enumerate(primary, start=1):
        if column in {"index_rank", "index_number"}:
            continue
        payload[column] = [float(column_number + 10), float(column_number + 20)]
    return pd.DataFrame(payload)


def test_freeze_writes_plain_json_hash_manifest_and_obeys_cutoff(
    tmp_path: Path,
) -> None:
    model_table_path, statement_features_path = _write_training_tables(tmp_path)
    model_path = tmp_path / "bundle.json"
    manifest_path = tmp_path / "manifest.json"
    paths = freeze_prospective_model(
        protocol_path=PROTOCOL_PATH,
        model_table_path=model_table_path,
        statement_features_path=statement_features_path,
        model_path=model_path,
        manifest_path=manifest_path,
        source_commit="0123456789abcdef",
        frozen_at_utc="2026-07-10T08:00:00Z",
    )

    assert paths == {"model": model_path, "manifest": manifest_path}
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_hash = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    artifact_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    assert artifact["estimator"] == {"alpha": 1.0, "name": "Ridge"}
    assert artifact["protocol_sha256"] == protocol_hash
    assert manifest["protocol_sha256"] == protocol_hash
    assert manifest["model_artifact_sha256"] == artifact_hash
    assert manifest["source_commit"] == "0123456789abcdef"
    assert manifest["training_cutoff_utc"] == "2026-07-12T00:00:00Z"
    assert manifest["training_row_count"] == 3
    assert manifest["training_start_time_seconds_max"] == CUTOFF_SECONDS - 1
    assert manifest["input_sha256"] == {
        "model_table": hashlib.sha256(model_table_path.read_bytes()).hexdigest(),
        "statement_features": hashlib.sha256(
            statement_features_path.read_bytes()
        ).hexdigest(),
    }
    verified = verify_frozen_model(
        protocol_path=PROTOCOL_PATH,
        model_path=model_path,
        manifest_path=manifest_path,
    )
    assert verified["model_artifact_sha256"] == artifact_hash
    assert verified["training_row_count"] == 3


def test_freeze_only_uses_allowlisted_pre_cutoff_features(tmp_path: Path) -> None:
    model, statements = _historical_tables()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_model = first_dir / "model.parquet"
    first_statements = first_dir / "statements.parquet"
    second_model = second_dir / "model.parquet"
    second_statements = second_dir / "statements.parquet"
    model.to_parquet(first_model, index=False)
    statements.to_parquet(first_statements, index=False)

    changed_model = model.copy()
    changed_model["unlisted_noise"] = [9e20, 8e20, 7e20, 6e20]
    changed_model.loc[3, "rating"] = -99999.0  # exactly at cutoff: excluded
    changed_statements = statements.copy()
    changed_statements["more_unlisted_noise"] = ["x", "y", "z", "future"]
    primary, _ = _protocol_columns()
    changed_statements.loc[3, [c for c in primary if c not in {"index_rank", "index_number"}]] = 1e30
    changed_model.to_parquet(second_model, index=False)
    changed_statements.to_parquet(second_statements, index=False)

    artifacts: list[dict[str, object]] = []
    for number, (model_path, statement_path) in enumerate(
        [(first_model, first_statements), (second_model, second_statements)]
    ):
        artifact_path = tmp_path / f"artifact-{number}.json"
        freeze_prospective_model(
            protocol_path=PROTOCOL_PATH,
            model_table_path=model_path,
            statement_features_path=statement_path,
            model_path=artifact_path,
            manifest_path=tmp_path / f"manifest-{number}.json",
            source_commit="abc1234",
            frozen_at_utc="2026-07-10T08:00:00Z",
        )
        artifacts.append(json.loads(artifact_path.read_text(encoding="utf-8")))

    assert artifacts[0] == artifacts[1]


def test_predict_is_deterministic_derives_index_and_has_exact_schema(
    tmp_path: Path,
) -> None:
    model_path, manifest_path = _freeze(tmp_path)
    prediction_input = _prediction_input()
    input_path = tmp_path / "new.csv"
    output_one = tmp_path / "predictions-one.csv"
    output_two = tmp_path / "predictions-two.csv"
    prediction_input.to_csv(input_path, index=False)
    kwargs = {
        "protocol_path": PROTOCOL_PATH,
        "model_path": model_path,
        "manifest_path": manifest_path,
        "input_path": input_path,
        "contest_start_utc": "2026-07-12T01:00:00+00:00",
        "prediction_created_at_utc": "2026-07-12T01:01:00Z",
    }
    predict_prospective(output_path=output_one, **kwargs)
    first = pd.read_csv(output_one)

    # Caller-supplied index features and unrelated audit columns cannot affect output.
    prediction_input["index_rank"] = [-5000, -5000]
    prediction_input["index_number"] = [-5000, -5000]
    prediction_input["irrelevant_audit_note"] = ["changed", "changed"]
    prediction_input.to_csv(input_path, index=False)
    predict_prospective(output_path=output_two, **kwargs)
    second = pd.read_csv(output_two)

    assert tuple(first.columns) == PREDICTION_COLUMNS
    pd.testing.assert_frame_equal(first, second)
    assert first["contest_start_utc"].eq("2026-07-12T01:00:00Z").all()
    assert first["prediction_created_at_utc"].eq("2026-07-12T01:01:00Z").all()
    assert first["feature_row_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_numeric_only_problem_index_uses_canonical_zero_rank(tmp_path: Path) -> None:
    """Legitimate Codeforces indices such as 01 remain eligible."""
    model_path, manifest_path = _freeze(tmp_path)
    frame = _prediction_input().iloc[[0]].copy()
    frame["index"] = "01"
    input_path = tmp_path / "numeric-index.csv"
    output_path = tmp_path / "numeric-index-prediction.csv"
    frame.to_csv(input_path, index=False)

    predict_prospective(
        protocol_path=PROTOCOL_PATH,
        model_path=model_path,
        manifest_path=manifest_path,
        input_path=input_path,
        output_path=output_path,
        contest_start_utc="2026-07-12T01:00:00Z",
        prediction_created_at_utc="2026-07-12T01:01:00Z",
    )

    result = pd.read_csv(output_path, dtype={"index": "string"})
    assert result.loc[0, "index"] == "01"


@pytest.mark.parametrize(
    "forbidden_column",
    [
        "rating",
        "points",
        "tags",
        "solved_count",
        "submissionCount",
        "accepted_count",
        "attempt_count",
        "participantCount",
        "verdict",
    ],
)
def test_predict_rejects_labels_metadata_and_behavior_columns(
    tmp_path: Path,
    forbidden_column: str,
) -> None:
    model_path, manifest_path = _freeze(tmp_path)
    frame = _prediction_input()
    frame[forbidden_column] = 1
    input_path = tmp_path / "forbidden.csv"
    frame.to_csv(input_path, index=False)

    with pytest.raises(ProspectiveModelError, match="forbidden"):
        predict_prospective(
            protocol_path=PROTOCOL_PATH,
            model_path=model_path,
            manifest_path=manifest_path,
            input_path=input_path,
            output_path=tmp_path / "should-not-exist.csv",
            contest_start_utc="2026-07-12T01:00:00Z",
            prediction_created_at_utc="2026-07-12T01:01:00Z",
        )


def test_predict_rejects_tampered_artifact(tmp_path: Path) -> None:
    model_path, manifest_path = _freeze(tmp_path)
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    artifact["primary_model"]["intercept"] += 1
    model_path.write_text(json.dumps(artifact), encoding="utf-8")
    input_path = tmp_path / "new.csv"
    _prediction_input().to_csv(input_path, index=False)

    with pytest.raises(ProspectiveModelError, match="SHA-256"):
        predict_prospective(
            protocol_path=PROTOCOL_PATH,
            model_path=model_path,
            manifest_path=manifest_path,
            input_path=input_path,
            output_path=tmp_path / "should-not-exist.csv",
            contest_start_utc="2026-07-12T01:00:00Z",
            prediction_created_at_utc="2026-07-12T01:01:00Z",
        )


@pytest.mark.parametrize(
    ("contest_start", "created", "message"),
    [
        (
            "2026-07-11T23:59:59Z",
            "2026-07-12T00:00:00Z",
            "outside",
        ),
        (
            "2026-07-12T01:00:00Z",
            "2026-07-12T00:59:59Z",
            "on or after",
        ),
        (
            "2026-07-12T01:00:00Z",
            "2026-07-12T01:30:01Z",
            "lock deadline",
        ),
    ],
)
def test_predict_enforces_cohort_and_recording_times(
    tmp_path: Path,
    contest_start: str,
    created: str,
    message: str,
) -> None:
    model_path, manifest_path = _freeze(tmp_path)
    input_path = tmp_path / "new.csv"
    _prediction_input().to_csv(input_path, index=False)

    with pytest.raises(ProspectiveModelError, match=message):
        predict_prospective(
            protocol_path=PROTOCOL_PATH,
            model_path=model_path,
            manifest_path=manifest_path,
            input_path=input_path,
            output_path=tmp_path / "should-not-exist.csv",
            contest_start_utc=contest_start,
            prediction_created_at_utc=created,
        )


def test_freeze_requires_one_to_one_historical_statement_join(tmp_path: Path) -> None:
    model, statements = _historical_tables()
    statements = statements.loc[
        ~((statements["contest_id"] == 100) & (statements["index"] == "B1"))
    ]
    model_path = tmp_path / "model.parquet"
    statements_path = tmp_path / "statements.parquet"
    model.to_parquet(model_path, index=False)
    statements.to_parquet(statements_path, index=False)

    with pytest.raises(ProspectiveModelError, match="missing"):
        freeze_prospective_model(
            protocol_path=PROTOCOL_PATH,
            model_table_path=model_path,
            statement_features_path=statements_path,
            model_path=tmp_path / "bundle.json",
            manifest_path=tmp_path / "manifest.json",
            source_commit="abc1234",
            frozen_at_utc="2026-07-10T08:00:00Z",
        )
