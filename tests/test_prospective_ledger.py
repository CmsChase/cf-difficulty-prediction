"""Tests for append-only prospective event chains and public witnesses."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cf_diff import prospective_ledger
from cf_diff.prospective_ledger import (
    COMMITMENT_EVENT_TYPES,
    ProspectiveLedgerError,
    assert_append_only_bytes,
    build_commitment_state,
    check_git_append_only,
    record_github_actions_witness,
    record_operational_miss,
    record_prediction_commitment,
    record_prediction_invalidation,
    verify_commitment_ledger,
    verify_ledger,
)
from cf_diff.prospective_model import PREDICTION_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROTOCOL = PROJECT_ROOT / "configs" / "prospective_protocol_v2.json"
CONTEST_START = "2026-08-15T01:00:00Z"
DEADLINE = "2026-08-15T01:30:00Z"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _frozen_protocol(root: Path, *, workflow_bytes: bytes = b"workflow\n") -> tuple[Path, dict[str, object]]:
    protocol = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    protocol["status"] = "frozen"
    protocol["protocol_frozen_at_utc"] = "2026-07-12T00:00:00Z"
    protocol["external_timestamp"]["workflow_file_sha256"] = hashlib.sha256(
        workflow_bytes
    ).hexdigest()
    path = root / "configs" / "prospective_protocol_v2.json"
    _write_json(path, protocol)
    return path, protocol


def _prediction_fixture(root: Path) -> dict[str, Path]:
    protocol_path, protocol = _frozen_protocol(root)
    protocol_sha = _sha(protocol_path)
    input_path = root / "prospective" / "inputs" / "3000_input.csv"
    sidecar_path = root / "prospective" / "inputs" / "3000_capture.json"
    prediction_path = root / "prospective" / "predictions" / "3000_predictions.csv"
    model_path = root / "prospective" / "model_bundle_v2.json"
    manifest_path = root / "prospective" / "model_freeze_manifest_v2.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("fixture input\n", encoding="utf-8")
    _write_json(model_path, {"model": "fixture"})
    model_sha = _sha(model_path)
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "model_bundle_id": "text-light-ridge-v2",
        "model_artifact_sha256": model_sha,
    }
    _write_json(manifest_path, manifest)
    input_sha = _sha(input_path)
    sidecar = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "contest_id": "3000",
        "contest_start_utc": CONTEST_START,
        "lock_deadline_utc": DEADLINE,
        "requested_indices": ["A", "B"],
        "output": {
            "path": input_path.relative_to(root).as_posix(),
            "sha256": input_sha,
            "row_count": 2,
        },
    }
    _write_json(sidecar_path, sidecar)
    sidecar_sha = _sha(sidecar_path)
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PREDICTION_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for index, value in (("A", 900.0), ("B", 1200.0)):
            writer.writerow(
                {
                    "contest_id": "3000",
                    "index": index,
                    "contest_start_utc": CONTEST_START,
                    "primary_prediction": value,
                    "comparator_prediction": value + 50,
                    "feature_row_sha256": "a" * 64,
                    "protocol_id": protocol["protocol_id"],
                    "model_bundle_id": "text-light-ridge-v2",
                    "model_artifact_sha256": model_sha,
                    "freeze_manifest_sha256": _sha(manifest_path),
                    "input_file_sha256": input_sha,
                    "capture_sidecar_sha256": sidecar_sha,
                    "prediction_created_at_utc": "2026-08-15T01:05:00Z",
                }
            )
    return {
        "protocol": protocol_path,
        "input": input_path,
        "sidecar": sidecar_path,
        "prediction": prediction_path,
        "model": model_path,
        "manifest": manifest_path,
        "ledger": root / "prospective" / "ledger" / "commitments.jsonl",
    }


def _record_prediction(
    fixture: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    monkeypatch.setattr(
        prospective_ledger,
        "_utc_now",
        lambda: datetime(2026, 8, 15, 1, 10, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        prospective_ledger,
        "verify_frozen_model",
        lambda **kwargs: {"status": "verified"},
    )
    monkeypatch.setattr(
        prospective_ledger,
        "predict_prospective",
        lambda **kwargs: kwargs["output_path"].write_bytes(
            fixture["prediction"].read_bytes()
        ),
    )
    return record_prediction_commitment(
        fixture["prediction"],
        input_path=fixture["input"],
        capture_sidecar_path=fixture["sidecar"],
        model_path=fixture["model"],
        protocol_path=fixture["protocol"],
        manifest_path=fixture["manifest"],
        ledger_path=fixture["ledger"],
    )


def test_prediction_commitment_requires_explicit_hash_bound_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    fixture = _prediction_fixture(tmp_path)
    event = _record_prediction(fixture, monkeypatch)
    assert event["indices"] == ["A", "B"]
    assert event["input_path"] == "prospective/inputs/3000_input.csv"
    assert event["capture_sidecar_path"] == "prospective/inputs/3000_capture.json"
    assert event["model_artifact_path"] == "prospective/model_bundle_v2.json"
    assert verify_commitment_ledger(
        fixture["ledger"], protocol_path=fixture["protocol"]
    )["prediction_commitments"] == 1

    fixture["input"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(ProspectiveLedgerError, match="hash mismatch"):
        verify_commitment_ledger(
            fixture["ledger"], protocol_path=fixture["protocol"]
        )


def test_commitment_recomputes_prediction_values_before_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    fixture = _prediction_fixture(tmp_path)
    monkeypatch.setattr(
        prospective_ledger,
        "_utc_now",
        lambda: datetime(2026, 8, 15, 1, 10, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        prospective_ledger, "verify_frozen_model", lambda **kwargs: {}
    )
    monkeypatch.setattr(
        prospective_ledger,
        "predict_prospective",
        lambda **kwargs: kwargs["output_path"].write_bytes(b"different\n"),
    )
    with pytest.raises(ProspectiveLedgerError, match="recomputation"):
        record_prediction_commitment(
            fixture["prediction"],
            input_path=fixture["input"],
            capture_sidecar_path=fixture["sidecar"],
            model_path=fixture["model"],
            protocol_path=fixture["protocol"],
            manifest_path=fixture["manifest"],
            ledger_path=fixture["ledger"],
        )
    assert not fixture["ledger"].exists()


def test_direct_miss_and_later_invalidation_are_distinct_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    fixture = _prediction_fixture(tmp_path)
    prediction = _record_prediction(fixture, monkeypatch)
    monkeypatch.setattr(
        prospective_ledger,
        "_utc_now",
        lambda: datetime(2026, 8, 16, tzinfo=timezone.utc),
    )
    with pytest.raises(ProspectiveLedgerError, match="Direct operational"):
        record_operational_miss(
            contest_id="3000",
            contest_start_utc=CONTEST_START,
            miss_stage="publication",
            reason_code="publication_failed",
            reason_detail="Too late.",
            protocol_path=fixture["protocol"],
            ledger_path=fixture["ledger"],
        )
    invalidation = record_prediction_invalidation(
        contest_id="3000",
        reason_code="official_index_set_mismatch",
        reason_detail="Fixed snapshot differs.",
        protocol_path=fixture["protocol"],
        ledger_path=fixture["ledger"],
    )
    assert invalidation["target_event_sha256"] == prediction["event_sha256"]
    with pytest.raises(ProspectiveLedgerError, match="already invalidated"):
        record_prediction_invalidation(
            contest_id="3000",
            reason_code="official_index_set_mismatch",
            reason_detail="Duplicate.",
            protocol_path=fixture["protocol"],
            ledger_path=fixture["ledger"],
        )
    state = build_commitment_state(fixture["ledger"])["3000"]
    assert state["qualified"] is False


def test_direct_operational_miss_is_one_base_coverage_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    protocol_path, _ = _frozen_protocol(tmp_path)
    ledger = Path("prospective/ledger/commitments.jsonl")
    monkeypatch.setattr(
        prospective_ledger,
        "_utc_now",
        lambda: datetime(2026, 8, 16, tzinfo=timezone.utc),
    )
    record_operational_miss(
        contest_id="3001",
        contest_start_utc="2026-08-15T02:00:00Z",
        miss_stage="capture",
        reason_code="capture_failed",
        reason_detail="No complete input was published.",
        protocol_path=protocol_path,
        ledger_path=ledger,
    )
    summary = verify_commitment_ledger(ledger, protocol_path=protocol_path)
    assert summary["coverage_contests"] == 1
    assert summary["direct_operational_misses"] == 1


@pytest.mark.parametrize("tamper", ["content", "partial", "duplicate_key", "unknown_field"])
def test_chain_tampering_and_non_exact_json_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    monkeypatch.chdir(tmp_path)
    fixture = _prediction_fixture(tmp_path)
    _record_prediction(fixture, monkeypatch)
    path = fixture["ledger"]
    raw = path.read_bytes()
    if tamper == "content":
        raw = raw.replace(b'"row_count":2', b'"row_count":3')
    elif tamper == "partial":
        raw = raw[:-1]
    elif tamper == "duplicate_key":
        raw = raw.replace(b'{', b'{"schema_version":1,', 1)
    else:
        raw = raw.replace(b'{', b'{"unexpected":true,', 1)
    path.write_bytes(raw)
    with pytest.raises(ProspectiveLedgerError):
        verify_ledger(path, allowed_event_types=COMMITMENT_EVENT_TYPES)


def test_lock_and_prefix_checks_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    fixture = _prediction_fixture(tmp_path)
    fixture["ledger"].parent.mkdir(parents=True, exist_ok=True)
    fixture["ledger"].with_name("commitments.jsonl.lock").write_text("held")
    with pytest.raises(ProspectiveLedgerError, match="lock"):
        _record_prediction(fixture, monkeypatch)
    assert_append_only_bytes(b"abc", b"abcdef", label="fixture")
    with pytest.raises(ProspectiveLedgerError, match="append-only"):
        assert_append_only_bytes(b"abc", b"abX", label="fixture")


def _run_payload(
    *,
    run_id: int,
    workflow_bytes: bytes,
    created_at: str = "2026-08-15T01:20:00Z",
    repository: str = "CmsChase/cf-difficulty-prediction",
    attempt: int = 1,
) -> bytes:
    return json.dumps(
        {
            "id": run_id,
            "name": "Prospective commitment witness",
            "workflow_id": 42,
            "run_attempt": attempt,
            "head_sha": "b" * 40,
            "head_branch": "main",
            "event": "push",
            "path": ".github/workflows/prospective-witness.yml",
            "status": "completed",
            "conclusion": "success",
            "created_at": created_at,
            "html_url": (
                "https://github.com/CmsChase/cf-difficulty-prediction/"
                f"actions/runs/{run_id}"
            ),
            "repository": {"id": 1280990637, "full_name": repository},
        },
        separators=(",", ":"),
    ).encode("utf-8")


def test_witness_fetches_exact_api_bytes_and_uses_github_created_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    workflow = b"workflow\n"
    fixture = _prediction_fixture(tmp_path)
    prediction = _record_prediction(fixture, monkeypatch)
    monkeypatch.setattr(prospective_ledger, "_git_commit_contains_event", lambda *args: None)
    monkeypatch.setattr(
        prospective_ledger, "_validate_event_artifacts_at_commit", lambda *args: None
    )
    monkeypatch.setattr(
        prospective_ledger, "_git_blob_at_commit", lambda *args: workflow
    )
    monkeypatch.setattr(
        prospective_ledger, "_target_is_default_branch_ancestor", lambda *args: None
    )
    run_id = 123456
    raw = _run_payload(run_id=run_id, workflow_bytes=workflow)
    seen: list[str] = []

    def fetch(url: str) -> bytes:
        seen.append(url)
        return raw

    event = record_github_actions_witness(
        run_id,
        contest_id="3000",
        protocol_path=fixture["protocol"],
        ledger_path=fixture["ledger"],
        evidence_dir=Path("prospective/witnesses"),
        fetcher=fetch,
    )
    evidence = Path(event["run_api_response_path"])
    assert evidence.read_bytes() == raw
    assert event["target_event_sha256"] == prediction["event_sha256"]
    assert event["external_timestamp"] == {
        "provider": "github_actions_workflow_run",
        "field": "created_at",
        "value_utc": "2026-08-15T01:20:00Z",
    }
    assert event["timely"] is True
    assert seen == [
        "https://api.github.com/repos/CmsChase/cf-difficulty-prediction/"
        f"actions/runs/{run_id}"
    ]


@pytest.mark.parametrize(
    ("repository", "attempt", "created_at", "expected"),
    [
        ("someone/else", 1, "2026-08-15T01:20:00Z", "policy"),
        ("CmsChase/cf-difficulty-prediction", 2, "2026-08-15T01:20:00Z", "policy"),
    ],
)
def test_witness_rejects_wrong_repository_or_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repository: str,
    attempt: int,
    created_at: str,
    expected: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    fixture = _prediction_fixture(tmp_path)
    _record_prediction(fixture, monkeypatch)
    run_id = 99
    with pytest.raises(ProspectiveLedgerError, match=expected):
        record_github_actions_witness(
            run_id,
            contest_id="3000",
            protocol_path=fixture["protocol"],
            ledger_path=fixture["ledger"],
            fetcher=lambda url: _run_payload(
                run_id=run_id,
                workflow_bytes=b"workflow\n",
                repository=repository,
                attempt=attempt,
                created_at=created_at,
            ),
        )
    assert not Path(f"prospective/witnesses/github-run-{run_id}.json").exists()


def test_late_github_timestamp_is_recorded_but_not_qualified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    fixture = _prediction_fixture(tmp_path)
    _record_prediction(fixture, monkeypatch)
    monkeypatch.setattr(prospective_ledger, "_git_commit_contains_event", lambda *args: None)
    monkeypatch.setattr(
        prospective_ledger, "_validate_event_artifacts_at_commit", lambda *args: None
    )
    monkeypatch.setattr(
        prospective_ledger, "_git_blob_at_commit", lambda *args: b"workflow\n"
    )
    monkeypatch.setattr(
        prospective_ledger, "_target_is_default_branch_ancestor", lambda *args: None
    )
    event = record_github_actions_witness(
        777,
        contest_id="3000",
        protocol_path=fixture["protocol"],
        ledger_path=fixture["ledger"],
        fetcher=lambda url: _run_payload(
            run_id=777,
            workflow_bytes=b"workflow\n",
            created_at="2026-08-15T01:30:01Z",
        ),
    )
    assert event["timely"] is False
    assert build_commitment_state(fixture["ledger"])["3000"]["qualified"] is False


def test_direct_miss_can_receive_the_same_external_time_witness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    protocol_path, _ = _frozen_protocol(tmp_path)
    ledger = Path("prospective/ledger/commitments.jsonl")
    monkeypatch.setattr(
        prospective_ledger,
        "_utc_now",
        lambda: datetime(2026, 8, 15, 1, 5, tzinfo=timezone.utc),
    )
    miss = record_operational_miss(
        contest_id="3000",
        contest_start_utc=CONTEST_START,
        miss_stage="capture",
        reason_code="capture_failed",
        reason_detail="No complete capture.",
        protocol_path=protocol_path,
        ledger_path=ledger,
    )
    monkeypatch.setattr(prospective_ledger, "_git_commit_contains_event", lambda *args: None)
    monkeypatch.setattr(
        prospective_ledger, "_validate_event_artifacts_at_commit", lambda *args: None
    )
    monkeypatch.setattr(
        prospective_ledger, "_git_blob_at_commit", lambda *args: b"workflow\n"
    )
    monkeypatch.setattr(
        prospective_ledger, "_target_is_default_branch_ancestor", lambda *args: None
    )
    witness = record_github_actions_witness(
        778,
        contest_id="3000",
        protocol_path=protocol_path,
        ledger_path=ledger,
        fetcher=lambda url: _run_payload(
            run_id=778,
            workflow_bytes=b"workflow\n",
            created_at="2026-08-15T01:20:00Z",
        ),
    )
    assert witness["target_event_sha256"] == miss["event_sha256"]
    state = build_commitment_state(ledger)["3000"]
    assert state["witness"] == witness
    assert state["qualified"] is False
    summary = verify_commitment_ledger(ledger, protocol_path=protocol_path)
    assert summary["timely_coverage_events"] == 1


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_git_guard_rejects_evidence_rewrite_and_frozen_control_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "fixture@example.com")
    _git(tmp_path, "config", "user.name", "Fixture")
    protocol_path, protocol = _frozen_protocol(tmp_path)
    protocol["status"] = "draft"
    protocol.pop("protocol_frozen_at_utc")
    _write_json(protocol_path, protocol)
    evidence = tmp_path / "prospective" / "inputs" / "published.csv"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("one\n", encoding="utf-8")
    source = tmp_path / "src" / "cf_diff" / "prospective_analysis.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# one\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "base draft")
    base = _git(tmp_path, "rev-parse", "HEAD")

    evidence.write_text("two\n", encoding="utf-8")
    with pytest.raises(ProspectiveLedgerError, match="immutable"):
        check_git_append_only(
            base,
            [Path("prospective/ledger/commitments.jsonl")],
            protocol_path=Path("configs/prospective_protocol_v2.json"),
        )
    evidence.write_text("one\n", encoding="utf-8")

    protocol["status"] = "frozen"
    protocol["protocol_frozen_at_utc"] = "2026-07-12T00:00:00Z"
    _write_json(protocol_path, protocol)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "freeze protocol")
    frozen_base = _git(tmp_path, "rev-parse", "HEAD")
    source.write_text("# changed\n", encoding="utf-8")
    with pytest.raises(ProspectiveLedgerError, match="Frozen prospective control"):
        check_git_append_only(
            frozen_base,
            [Path("prospective/ledger/commitments.jsonl")],
            protocol_path=Path("configs/prospective_protocol_v2.json"),
        )


def test_cli_uses_run_id_and_explicit_prediction_artifacts() -> None:
    parser = prospective_ledger._build_parser()
    witness = parser.parse_args(["witness", "--run-id", "123", "--contest-id", "3000"])
    assert witness.run_id == 123
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["witness", "--run-response", "response.json", "--contest-id", "3000"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(["prediction", "--predictions", "prediction.csv"])
