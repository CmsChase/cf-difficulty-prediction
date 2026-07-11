"""Integration tests for prospective census, mapping, and analysis input."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cf_diff import prospective_cohort, prospective_ledger, prospective_snapshot
from cf_diff.prospective_analysis import run_confirmatory_analysis
from cf_diff.prospective_cohort import ProspectiveCohortError, finalize_cohort
from cf_diff.prospective_model import PREDICTION_COLUMNS
from cf_diff.prospective_snapshot import AttemptResponse, run_fixed_snapshot_window


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROTOCOL = PROJECT_ROOT / "configs" / "prospective_protocol_v2.json"


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _frozen_protocol(root: Path) -> tuple[Path, dict[str, object]]:
    protocol = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    protocol["status"] = "frozen"
    protocol["protocol_frozen_at_utc"] = "2026-07-12T00:00:00Z"
    path = root / "configs" / "prospective_protocol_v2.json"
    _write_json(path, protocol)
    return path, protocol


def _api_response(
    window: prospective_snapshot.SnapshotWindow, payload: object
) -> AttemptResponse:
    return AttemptResponse(
        prospective_snapshot._expected_request_url(window),
        200,
        json.dumps(payload, separators=(",", ":")).encode("utf-8"),
    )


def _build_repository(
    root: Path,
    *,
    contest_count: int = 30,
    rows_per_contest: int = 7,
    omit_key: tuple[str, str] | None = None,
    invalidate_contest: str | None = None,
    direct_miss_contest: str | None = None,
    missing_rating_key: tuple[str, str] | None = None,
    non_programming_key: tuple[str, str] | None = None,
) -> dict[str, object]:
    protocol_path, protocol = _frozen_protocol(root)
    protocol_sha = _sha(protocol_path)
    model_path = root / "prospective" / "model_bundle_v2.json"
    manifest_path = root / "prospective" / "model_freeze_manifest_v2.json"
    _write_json(model_path, {"model": "fixture"})
    _write_json(manifest_path, {"manifest": "fixture"})
    model_sha = _sha(model_path)
    manifest_sha = _sha(manifest_path)
    commitment_path = root / "prospective" / "ledger" / "commitments.jsonl"
    observation_path = root / "prospective" / "ledger" / "observations.jsonl"

    base_start = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    census_result: list[dict[str, object]] = []
    outcome_problems: list[dict[str, object]] = []
    prediction_payloads: list[dict[str, object]] = []
    indices = [chr(ord("A") + number) for number in range(rows_per_contest)]
    for offset in range(contest_count):
        contest_id = str(3000 + offset)
        start = base_start + timedelta(days=offset * 3)
        census_result.append(
            {
                "id": int(contest_id),
                "name": f"Round {contest_id}",
                "type": "CF",
                "phase": "FINISHED",
                "startTimeSeconds": int(start.timestamp()),
                "durationSeconds": 7200,
            }
        )
        for position, index in enumerate(indices):
            key = (contest_id, index)
            if key == omit_key:
                continue
            problem: dict[str, object] = {
                "contestId": int(contest_id),
                "index": index,
                "type": "PROGRAMMING",
            }
            if key == non_programming_key:
                problem["type"] = "OUTPUT_ONLY"
            elif key != missing_rating_key:
                problem["rating"] = 1000 + position * 100
            outcome_problems.append(problem)
        if contest_id == direct_miss_contest:
            prediction_payloads.append(
                {
                    "event_type": "operational_miss",
                    "contest_id": contest_id,
                    "operator_contest_start_utc": _utc(start),
                    "lock_deadline_utc": _utc(start + timedelta(minutes=30)),
                    "miss_stage": "capture",
                    "reason_code": "capture_failed",
                    "reason_detail": "Synthetic fixture capture failed.",
                    "evidence_path": None,
                    "evidence_sha256": None,
                }
            )
            continue
        input_path = root / "prospective" / "inputs" / f"{contest_id}_input.csv"
        sidecar_path = root / "prospective" / "inputs" / f"{contest_id}_capture.json"
        prediction_path = (
            root / "prospective" / "predictions" / f"{contest_id}_predictions.csv"
        )
        input_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text("fixture input\n", encoding="utf-8")
        _write_json(sidecar_path, {"fixture": contest_id})
        input_sha = _sha(input_path)
        sidecar_sha = _sha(sidecar_path)
        with prediction_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=PREDICTION_COLUMNS, lineterminator="\n"
            )
            writer.writeheader()
            for position, index in enumerate(indices):
                rating = 1000 + position * 100
                writer.writerow(
                    {
                        "contest_id": contest_id,
                        "index": index,
                        "contest_start_utc": _utc(start),
                        "primary_prediction": rating - 10,
                        "comparator_prediction": rating - 50,
                        "feature_row_sha256": "a" * 64,
                        "protocol_id": protocol["protocol_id"],
                        "model_bundle_id": "text-light-ridge-v2",
                        "model_artifact_sha256": model_sha,
                        "freeze_manifest_sha256": manifest_sha,
                        "input_file_sha256": input_sha,
                        "capture_sidecar_sha256": sidecar_sha,
                        "prediction_created_at_utc": _utc(start + timedelta(minutes=5)),
                    }
                )
        prediction_payloads.append(
            {
                "event_type": "prediction_commitment",
                "contest_id": contest_id,
                "operator_contest_start_utc": _utc(start),
                "lock_deadline_utc": _utc(start + timedelta(minutes=30)),
                "prediction_created_at_utc": _utc(start + timedelta(minutes=5)),
                "indices": indices,
                "row_count": len(indices),
                "input_path": input_path.relative_to(root).as_posix(),
                "input_sha256": input_sha,
                "capture_sidecar_path": sidecar_path.relative_to(root).as_posix(),
                "capture_sidecar_sha256": sidecar_sha,
                "prediction_path": prediction_path.relative_to(root).as_posix(),
                "prediction_sha256": _sha(prediction_path),
                "model_artifact_path": model_path.relative_to(root).as_posix(),
                "freeze_manifest_path": manifest_path.relative_to(root).as_posix(),
                "model_bundle_id": "text-light-ridge-v2",
                "model_artifact_sha256": model_sha,
                "freeze_manifest_sha256": manifest_sha,
            }
        )
    base_events = prospective_ledger._append_payloads(
        commitment_path,
        prediction_payloads,
        protocol=protocol,
        protocol_sha256=protocol_sha,
        allowed_event_types=prospective_ledger.COMMITMENT_EVENT_TYPES,
    )
    witness_payloads: list[dict[str, object]] = []
    for number, event in enumerate(base_events, start=1):
        run_id = 900000 + number
        evidence_path = root / "prospective" / "witnesses" / f"github-run-{run_id}.json"
        _write_json(evidence_path, {"id": run_id, "fixture": True})
        head_sha = f"{number:040x}"
        witness_payloads.append(
            {
                "event_type": "github_actions_witness",
                "contest_id": event["contest_id"],
                "target_event_sha256": event["event_sha256"],
                "target_commit_sha": head_sha,
                "run_api_response_path": evidence_path.relative_to(root).as_posix(),
                "run_api_response_sha256": _sha(evidence_path),
                "repository_id": 1280990637,
                "repository_full_name": "CmsChase/cf-difficulty-prediction",
                "head_sha": head_sha,
                "head_branch": "main",
                "trigger_event": "push",
                "workflow_id": 42,
                "workflow_path": ".github/workflows/prospective-witness.yml",
                "workflow_file_sha256": protocol["external_timestamp"][
                    "workflow_file_sha256"
                ],
                "run_id": run_id,
                "run_attempt": 1,
                "external_timestamp": {
                    "provider": "github_actions_workflow_run",
                    "field": "created_at",
                    "value_utc": _utc(
                        _parse_fixture_utc(event["operator_contest_start_utc"])
                        + timedelta(minutes=10)
                    ),
                },
                "status": "completed",
                "conclusion": "success",
                "run_url": (
                    "https://github.com/CmsChase/cf-difficulty-prediction/"
                    f"actions/runs/{run_id}"
                ),
                "timely": True,
            }
        )
    prospective_ledger._append_payloads(
        commitment_path,
        witness_payloads,
        protocol=protocol,
        protocol_sha256=protocol_sha,
        allowed_event_types=prospective_ledger.COMMITMENT_EVENT_TYPES,
    )
    if invalidate_contest is not None:
        events = prospective_ledger.verify_ledger(
            commitment_path,
            allowed_event_types=prospective_ledger.COMMITMENT_EVENT_TYPES,
        )
        target = next(
            event
            for event in events
            if event["event_type"] == "prediction_commitment"
            and event["contest_id"] == invalidate_contest
        )
        prospective_ledger._append_payloads(
            commitment_path,
            [
                {
                    "event_type": "prediction_invalidated",
                    "contest_id": invalidate_contest,
                    "target_event_sha256": target["event_sha256"],
                    "reason_code": "official_index_set_mismatch",
                    "reason_detail": "Synthetic fixed snapshot mismatch.",
                    "evidence_path": None,
                    "evidence_sha256": None,
                }
            ],
            protocol=protocol,
            protocol_sha256=protocol_sha,
            allowed_event_types=prospective_ledger.COMMITMENT_EVENT_TYPES,
        )

    census_run = root / "prospective" / "snapshots" / "v2" / "cohort-census"
    outcome_run = (
        root / "prospective" / "snapshots" / "v2" / "confirmatory-outcome"
    )
    census_window = prospective_snapshot.load_snapshot_window(
        protocol_path, "cohort_census"
    )
    outcome_window = prospective_snapshot.load_snapshot_window(
        protocol_path, "confirmatory_outcome"
    )
    census_clock = MutableClock(census_window.anchor_utc)
    outcome_clock = MutableClock(outcome_window.anchor_utc)
    run_fixed_snapshot_window(
        protocol_path,
        "cohort_census",
        census_run,
        clock=census_clock,
        sleep_fn=census_clock.sleep,
        fetch_once=lambda window: _api_response(
            window, {"status": "OK", "result": census_result}
        ),
    )
    run_fixed_snapshot_window(
        protocol_path,
        "confirmatory_outcome",
        outcome_run,
        clock=outcome_clock,
        sleep_fn=outcome_clock.sleep,
        fetch_once=lambda window: _api_response(
            window,
            {
                "status": "OK",
                "result": {
                    "problems": outcome_problems,
                    "problemStatistics": [],
                },
            },
        ),
    )
    prospective_ledger.record_snapshot_observation(
        census_run / "selection.json",
        protocol_path=protocol_path,
        observation_ledger_path=observation_path,
    )
    prospective_ledger.record_snapshot_observation(
        outcome_run / "selection.json",
        protocol_path=protocol_path,
        observation_ledger_path=observation_path,
    )
    return {
        "protocol_path": protocol_path,
        "protocol": protocol,
        "commitment": commitment_path,
        "observation": observation_path,
        "census_run": census_run,
        "outcome_run": outcome_run,
        "output": root / "prospective" / "cohort" / "v2",
    }


def _parse_fixture_utc(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _finalize(package: dict[str, object]) -> dict[str, object]:
    return finalize_cohort(
        package["protocol_path"],
        package["census_run"],
        package["outcome_run"],
        package["commitment"],
        package["observation"],
        package["output"],
        repository_root=Path.cwd(),
        clock=lambda: datetime(2027, 3, 4, 1, 0, tzinfo=timezone.utc),
    )


def test_finalize_and_analysis_form_one_hash_bound_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    package = _build_repository(tmp_path)
    manifest = _finalize(package)
    output = package["output"]
    assert manifest["finalized_csv"]["row_count"] == 210
    report = json.loads((output / "mapping_report.json").read_text())
    assert report["qualified_prediction_contest_count"] == 30
    assert report["paired_problem_row_count"] == 210

    result = run_confirmatory_analysis(
        package["protocol_path"],
        output / "confirmatory_analysis_input.json",
        tmp_path / "prospective" / "results" / "v2" / "analysis.json",
        commitment_ledger_path=package["commitment"],
        observation_ledger_path=package["observation"],
        clock=lambda: datetime(2027, 3, 5, 1, 0, tzinfo=timezone.utc),
    )
    assert result["eligible_contest_count"] == 30
    assert result["eligible_problem_row_count"] == 210
    assert result["primary"]["confirmatory_success"] is True


def test_unmapped_census_contest_publishes_failure_not_analysis_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    package = _build_repository(tmp_path, direct_miss_contest="3029")
    # Remove the direct miss event by truncating to make census contest 3029 unmapped.
    events = prospective_ledger.verify_ledger(
        package["commitment"],
        allowed_event_types=prospective_ledger.COMMITMENT_EVENT_TYPES,
    )
    retained = [
        event
        for event in events
        if event.get("contest_id") != "3029"
    ]
    # Rebuild canonical hashes after an intentionally different synthetic history.
    package["commitment"].unlink()
    payloads = [
        {
            key: value
            for key, value in event.items()
            if key
            not in {
                "schema_version",
                "chain_id",
                "sequence",
                "protocol_id",
                "protocol_sha256",
                "local_recorded_at_utc",
                "previous_event_sha256",
                "event_sha256",
            }
        }
        for event in retained
        if event["event_type"] in {"prediction_commitment", "operational_miss"}
    ]
    protocol = package["protocol"]
    prospective_ledger._append_payloads(
        package["commitment"],
        payloads,
        protocol=protocol,
        protocol_sha256=_sha(package["protocol_path"]),
        allowed_event_types=prospective_ledger.COMMITMENT_EVENT_TYPES,
    )
    with pytest.raises(ProspectiveCohortError, match="map exactly"):
        _finalize(package)
    assert (package["output"] / "mapping_failure_report.json").is_file()
    assert not (package["output"] / "confirmatory_analysis_input.json").exists()


def test_qualified_index_mismatch_requires_ledger_invalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    package = _build_repository(tmp_path, omit_key=("3000", "G"))
    with pytest.raises(ProspectiveCohortError, match="invalidation"):
        _finalize(package)
    failure = json.loads(
        (package["output"] / "mapping_failure_report.json").read_text()
    )
    assert failure["qualified_mismatch_without_invalidation_ids"] == ["3000"]


def test_invalidated_prediction_retains_missing_nonprogramming_and_absent_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    package = _build_repository(
        tmp_path,
        omit_key=("3000", "C"),
        invalidate_contest="3000",
        missing_rating_key=("3000", "A"),
        non_programming_key=("3000", "B"),
    )
    _finalize(package)
    with (package["output"] / "confirmatory_outcomes.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = [row for row in csv.DictReader(handle) if row["contest_id"] == "3000"]
    assert {row["outcome_status"] for row in rows[:3]} == {
        "missing_rating",
        "non_programming",
        "absent_from_snapshot",
    }
    assert all(row["eligible_for_analysis"] == "false" for row in rows)


def test_prediction_artifact_tamper_blocks_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    package = _build_repository(tmp_path)
    target = tmp_path / "prospective" / "predictions" / "3000_predictions.csv"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises((ProspectiveCohortError, prospective_ledger.ProspectiveLedgerError), match="hash"):
        _finalize(package)


def test_cli_accepts_no_ratings_file_or_time_override(tmp_path: Path) -> None:
    parser = prospective_cohort._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--ratings", str(tmp_path / "ratings.csv")])
    with pytest.raises(SystemExit):
        parser.parse_args(["--now", "2099-01-01T00:00:00Z"])
