"""Tests for prospective prediction and reveal ledgers."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import prospective_ledger


def _write_protocol(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": "test-protocol-v1",
                "status": "frozen",
                "cohort": {
                    "eligibility_start_utc": "2026-07-12T00:00:00Z",
                    "eligibility_end_utc": "2026-08-01T00:00:00Z",
                },
                "prediction_timepoint": {
                    "lock_deadline_minutes_after_contest_start": 30
                },
                "outcome": {
                    "minimum_reveal_delay_hours_after_contest_start": 72
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_manifest(path: Path, protocol_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": "test-protocol-v1",
                "protocol_sha256": prospective_ledger.sha256_file(protocol_path),
                "model_bundle_id": "test-bundle-v1",
                "model_artifact_sha256": "a" * 64,
                "source_commit": "b" * 40,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contest_id": [9001, 9001],
            "index": ["A", "B"],
            "contest_start_utc": [
                "2026-07-12T00:00:00Z",
                "2026-07-12T00:00:00Z",
            ],
            "primary_prediction": [1200.25, 1550.75],
            "comparator_prediction": [1150.0, 1450.0],
            "feature_row_sha256": ["1" * 64, "2" * 64],
            "protocol_id": ["test-protocol-v1", "test-protocol-v1"],
            "model_bundle_id": ["test-bundle-v1", "test-bundle-v1"],
            "model_artifact_sha256": ["a" * 64, "a" * 64],
            "prediction_created_at_utc": [
                "2026-07-12T00:05:00Z",
                "2026-07-12T00:05:00Z",
            ],
        }
    )


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    protocol = tmp_path / "protocol.json"
    manifest = tmp_path / "manifest.json"
    predictions = tmp_path / "predictions.csv"
    _write_protocol(protocol)
    _write_manifest(manifest, protocol)
    _prediction_frame().to_csv(predictions, index=False)
    return protocol, manifest, predictions


def test_canonical_json_hash_input_ignores_key_order() -> None:
    """Canonical encoding is stable across mapping insertion order."""
    left = prospective_ledger.canonical_json_bytes({"b": 2, "a": 1})
    right = prospective_ledger.canonical_json_bytes({"a": 1, "b": 2})
    assert left == right == b'{"a":1,"b":2}'


def test_append_only_prefix_rejects_rewrite_or_deletion() -> None:
    """CI comparison accepts appends but rejects modified historical bytes."""
    previous = b'{"sequence":1}\n'
    prospective_ledger.assert_append_only_bytes(
        previous,
        previous + b'{"sequence":2}\n',
        label="predictions",
    )
    with pytest.raises(prospective_ledger.ProspectiveLedgerError, match="append-only"):
        prospective_ledger.assert_append_only_bytes(
            previous,
            b'{"sequence":9}\n',
            label="predictions",
        )


def test_record_and_verify_prediction_batch(tmp_path: Path) -> None:
    """A valid contest batch becomes one verifiable chained event."""
    protocol, manifest, predictions = _setup(tmp_path)
    ledger = tmp_path / "predictions.jsonl"

    event = prospective_ledger.record_prediction_batch(
        predictions,
        protocol_path=protocol,
        manifest_path=manifest,
        ledger_path=ledger,
        recorded_at=datetime(2026, 7, 12, 0, 6, tzinfo=timezone.utc),
    )
    summary = prospective_ledger.verify_prospective_ledgers(
        ledger,
        tmp_path / "reveals.jsonl",
    )

    assert event["previous_event_sha256"] == "0" * 64
    assert event["contest_id"] == "9001"
    assert [row["index"] for row in event["predictions"]] == ["A", "B"]
    assert summary == {
        "prediction_events": 1,
        "prediction_contests": 1,
        "missed_contests": 0,
        "predicted_problems": 2,
        "reveal_events": 0,
        "revealed_problems": 0,
    }


def test_prediction_schema_rejects_rating_or_other_extra_columns(
    tmp_path: Path,
) -> None:
    """A prediction-stage file cannot carry official outcomes."""
    protocol, manifest, predictions = _setup(tmp_path)
    frame = pd.read_csv(predictions)
    frame["official_rating"] = [1200, 1600]
    frame.to_csv(predictions, index=False)

    with pytest.raises(prospective_ledger.ProspectiveLedgerError, match="extra"):
        prospective_ledger.record_prediction_batch(
            predictions,
            protocol_path=protocol,
            manifest_path=manifest,
            ledger_path=tmp_path / "predictions.jsonl",
            recorded_at=datetime(2026, 7, 12, 0, 6, tzinfo=timezone.utc),
        )


def test_duplicate_contest_batch_is_rejected(tmp_path: Path) -> None:
    """The append path cannot overwrite or repeat a contest commitment."""
    protocol, manifest, predictions = _setup(tmp_path)
    ledger = tmp_path / "predictions.jsonl"
    kwargs = {
        "protocol_path": protocol,
        "manifest_path": manifest,
        "ledger_path": ledger,
        "recorded_at": datetime(2026, 7, 12, 0, 6, tzinfo=timezone.utc),
    }
    prospective_ledger.record_prediction_batch(predictions, **kwargs)

    with pytest.raises(prospective_ledger.ProspectiveLedgerError, match="already"):
        prospective_ledger.record_prediction_batch(predictions, **kwargs)


def test_tampering_and_partial_lines_are_detected(tmp_path: Path) -> None:
    """Changing a committed value or truncating a line breaks verification."""
    protocol, manifest, predictions = _setup(tmp_path)
    ledger = tmp_path / "predictions.jsonl"
    prospective_ledger.record_prediction_batch(
        predictions,
        protocol_path=protocol,
        manifest_path=manifest,
        ledger_path=ledger,
        recorded_at=datetime(2026, 7, 12, 0, 6, tzinfo=timezone.utc),
    )
    event = json.loads(ledger.read_text(encoding="utf-8"))
    event["predictions"][0]["primary_prediction"] = 9999
    ledger.write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(prospective_ledger.ProspectiveLedgerError, match="event hash"):
        prospective_ledger.verify_ledger(ledger)

    ledger.write_bytes(ledger.read_bytes().rstrip(b"\n"))
    with pytest.raises(prospective_ledger.ProspectiveLedgerError, match="partial"):
        prospective_ledger.verify_ledger(ledger)


def test_reveal_is_separate_and_does_not_modify_predictions(tmp_path: Path) -> None:
    """Ratings append to their own chain and reference the prediction hash."""
    protocol, manifest, predictions = _setup(tmp_path)
    prediction_ledger = tmp_path / "predictions.jsonl"
    reveal_ledger = tmp_path / "reveals.jsonl"
    prediction_event = prospective_ledger.record_prediction_batch(
        predictions,
        protocol_path=protocol,
        manifest_path=manifest,
        ledger_path=prediction_ledger,
        recorded_at=datetime(2026, 7, 12, 0, 6, tzinfo=timezone.utc),
    )
    before = prediction_ledger.read_bytes()
    ratings = tmp_path / "ratings.csv"
    pd.DataFrame(
        {
            "contest_id": [9001, 9001],
            "index": ["A", "B"],
            "official_rating": [1200, 1600],
        }
    ).to_csv(ratings, index=False)

    reveal_event = prospective_ledger.record_rating_reveal(
        ratings,
        protocol_path=protocol,
        prediction_ledger_path=prediction_ledger,
        reveal_ledger_path=reveal_ledger,
        revealed_at=datetime(2026, 7, 15, 0, 1, tzinfo=timezone.utc),
    )
    summary = prospective_ledger.verify_prospective_ledgers(
        prediction_ledger,
        reveal_ledger,
    )

    assert prediction_ledger.read_bytes() == before
    assert reveal_event["prediction_event_sha256"] == prediction_event["event_sha256"]
    assert summary["revealed_problems"] == 2


def test_reveal_before_frozen_delay_is_rejected(tmp_path: Path) -> None:
    """The separate outcome path cannot run before the 72-hour delay."""
    protocol, manifest, predictions = _setup(tmp_path)
    prediction_ledger = tmp_path / "predictions.jsonl"
    prospective_ledger.record_prediction_batch(
        predictions,
        protocol_path=protocol,
        manifest_path=manifest,
        ledger_path=prediction_ledger,
        recorded_at=datetime(2026, 7, 12, 0, 6, tzinfo=timezone.utc),
    )
    ratings = tmp_path / "ratings.csv"
    pd.DataFrame(
        {
            "contest_id": [9001],
            "index": ["A"],
            "official_rating": [1200],
        }
    ).to_csv(ratings, index=False)

    with pytest.raises(prospective_ledger.ProspectiveLedgerError, match="earlier"):
        prospective_ledger.record_rating_reveal(
            ratings,
            protocol_path=protocol,
            prediction_ledger_path=prediction_ledger,
            reveal_ledger_path=tmp_path / "reveals.jsonl",
            revealed_at=datetime(2026, 7, 14, 23, 59, tzinfo=timezone.utc),
        )
