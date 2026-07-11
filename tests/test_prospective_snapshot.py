"""Tests for fixed first-success prospective API snapshots."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cf_diff import prospective_snapshot
from cf_diff.prospective_snapshot import (
    AttemptResponse,
    ProspectiveSnapshotError,
    load_snapshot_window,
    run_fixed_snapshot_window,
    verify_snapshot_selection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROTOCOL = PROJECT_ROOT / "configs" / "prospective_protocol_v2.json"


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def _frozen_protocol(tmp_path: Path) -> Path:
    protocol = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    protocol["status"] = "frozen"
    protocol["protocol_frozen_at_utc"] = "2026-07-12T00:00:00Z"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return path


def _api_bytes(payload: object) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _response(window: prospective_snapshot.SnapshotWindow, payload: object) -> AttemptResponse:
    return AttemptResponse(
        request_url=prospective_snapshot._expected_request_url(window),
        http_status=200,
        raw_bytes=_api_bytes(payload),
    )


def _valid_census() -> dict[str, object]:
    return {
        "status": "OK",
        "result": [
            {"id": 3001, "startTimeSeconds": 1786748400, "name": "Round"},
            {"id": 3002, "startTimeSeconds": 1786834800, "name": "Round 2"},
        ],
    }


def _valid_outcome_without_rating() -> dict[str, object]:
    return {
        "status": "OK",
        "result": {
            "problems": [
                {"contestId": 3001, "index": "A", "type": "PROGRAMMING"}
            ],
            "problemStatistics": [],
        },
    }


def test_protocol_derives_exact_48_slot_windows(tmp_path: Path) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    census = load_snapshot_window(protocol_path, "cohort_census")
    outcome = load_snapshot_window(protocol_path, "confirmatory_outcome")

    assert census.anchor_utc == datetime(2027, 3, 1, 0, 4, 59, tzinfo=timezone.utc)
    assert census.deadline_utc == datetime(2027, 3, 2, 0, 4, 59, tzinfo=timezone.utc)
    assert outcome.anchor_utc == datetime(2027, 3, 4, 0, 4, 59, tzinfo=timezone.utc)
    assert outcome.deadline_utc == datetime(2027, 3, 5, 0, 4, 59, tzinfo=timezone.utc)
    assert census.maximum_attempts == outcome.maximum_attempts == 48
    assert census.scheduled_at(48) == census.deadline_utc - timedelta(minutes=30)
    with pytest.raises(ProspectiveSnapshotError, match="outside"):
        census.scheduled_at(49)


def test_third_structure_success_is_immediately_sealed(tmp_path: Path) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    window = load_snapshot_window(protocol_path, "cohort_census")
    clock = MutableClock(window.anchor_utc)
    calls: list[datetime] = []

    def fetch(current: prospective_snapshot.SnapshotWindow) -> AttemptResponse:
        calls.append(clock())
        if len(calls) == 1:
            return _response(current, {"status": "FAILED", "comment": "busy"})
        if len(calls) == 2:
            return AttemptResponse(
                prospective_snapshot._expected_request_url(current),
                200,
                b'{"status":"OK","result":',
            )
        return _response(current, _valid_census())

    selected = run_fixed_snapshot_window(
        protocol_path,
        "cohort_census",
        tmp_path / "run",
        clock=clock,
        sleep_fn=clock.sleep,
        fetch_once=fetch,
    )

    assert len(calls) == 3
    assert calls == [
        window.anchor_utc,
        window.anchor_utc + timedelta(minutes=30),
        window.anchor_utc + timedelta(minutes=60),
    ]
    assert selected["selected_attempt_number"] == 3
    assert selected["record_counts"] == {"contests": 2}
    assert len((tmp_path / "run" / "attempts.jsonl").read_text().splitlines()) == 3


def test_missing_rating_does_not_authorize_another_response(tmp_path: Path) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    window = load_snapshot_window(protocol_path, "confirmatory_outcome")
    clock = MutableClock(window.anchor_utc)
    calls = 0

    def fetch(current: prospective_snapshot.SnapshotWindow) -> AttemptResponse:
        nonlocal calls
        calls += 1
        return _response(current, _valid_outcome_without_rating())

    selected = run_fixed_snapshot_window(
        protocol_path,
        "confirmatory_outcome",
        tmp_path / "outcome",
        clock=clock,
        sleep_fn=clock.sleep,
        fetch_once=fetch,
    )
    assert calls == 1
    assert selected["selected_attempt_number"] == 1
    assert selected["record_counts"]["problems"] == 1


def test_sealed_rerun_verifies_without_network(tmp_path: Path) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    window = load_snapshot_window(protocol_path, "cohort_census")
    run_dir = tmp_path / "run"
    clock = MutableClock(window.anchor_utc)
    run_fixed_snapshot_window(
        protocol_path,
        "cohort_census",
        run_dir,
        clock=clock,
        sleep_fn=clock.sleep,
        fetch_once=lambda current: _response(current, _valid_census()),
    )

    selected = run_fixed_snapshot_window(
        protocol_path,
        "cohort_census",
        run_dir,
        clock=clock,
        sleep_fn=clock.sleep,
        fetch_once=lambda current: pytest.fail("network must not be called"),
    )
    assert selected == verify_snapshot_selection(
        protocol_path, run_dir, expected_kind="cohort_census"
    )


def test_missed_slot_invalidates_before_network(tmp_path: Path) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    window = load_snapshot_window(protocol_path, "cohort_census")
    clock = MutableClock(window.anchor_utc + timedelta(seconds=61))

    with pytest.raises(ProspectiveSnapshotError, match="missed"):
        run_fixed_snapshot_window(
            protocol_path,
            "cohort_census",
            tmp_path / "late",
            clock=clock,
            sleep_fn=clock.sleep,
            fetch_once=lambda current: pytest.fail("network must not be called"),
        )
    terminal = json.loads((tmp_path / "late" / "terminal.json").read_text())
    assert terminal["status"] == "invalidated"
    assert terminal["reason_code"] == "missed_scheduled_attempt"


def test_response_crossing_deadline_cannot_be_selected(tmp_path: Path) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    window = load_snapshot_window(protocol_path, "cohort_census")
    clock = MutableClock(window.anchor_utc)

    def fetch(current: prospective_snapshot.SnapshotWindow) -> AttemptResponse:
        clock.value = current.deadline_utc + timedelta(seconds=1)
        return _response(current, _valid_census())

    with pytest.raises(ProspectiveSnapshotError, match="deadline"):
        run_fixed_snapshot_window(
            protocol_path,
            "cohort_census",
            tmp_path / "crossed",
            clock=clock,
            sleep_fn=clock.sleep,
            fetch_once=fetch,
        )
    assert not (tmp_path / "crossed" / "selection.json").exists()


def test_all_48_failures_exhaust_without_a_49th_request(tmp_path: Path) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    window = load_snapshot_window(protocol_path, "cohort_census")
    clock = MutableClock(window.anchor_utc)
    calls = 0

    def fetch(current: prospective_snapshot.SnapshotWindow) -> AttemptResponse:
        nonlocal calls
        calls += 1
        return _response(current, {"status": "FAILED"})

    with pytest.raises(ProspectiveSnapshotError, match="48"):
        run_fixed_snapshot_window(
            protocol_path,
            "cohort_census",
            tmp_path / "exhausted",
            clock=clock,
            sleep_fn=clock.sleep,
            fetch_once=fetch,
        )
    assert calls == 48
    assert len((tmp_path / "exhausted" / "attempts.jsonl").read_text().splitlines()) == 48
    assert json.loads((tmp_path / "exhausted" / "terminal.json").read_text())[
        "status"
    ] == "exhausted"


@pytest.mark.parametrize("target", ["raw", "attempts", "selection"])
def test_tampering_breaks_offline_verification(tmp_path: Path, target: str) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    window = load_snapshot_window(protocol_path, "cohort_census")
    run_dir = tmp_path / "run"
    clock = MutableClock(window.anchor_utc)
    selected = run_fixed_snapshot_window(
        protocol_path,
        "cohort_census",
        run_dir,
        clock=clock,
        sleep_fn=clock.sleep,
        fetch_once=lambda current: _response(current, _valid_census()),
    )
    if target == "raw":
        path = run_dir / str(selected["raw_snapshot_path"])
        path.write_bytes(path.read_bytes() + b" ")
    elif target == "attempts":
        path = run_dir / "attempts.jsonl"
        path.write_bytes(path.read_bytes().replace(b'"status":"selected"', b'"status":"failed"'))
    else:
        path = run_dir / "selection.json"
        manifest = json.loads(path.read_text())
        manifest["selected_attempt_number"] = 2
        path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ProspectiveSnapshotError):
        verify_snapshot_selection(
            protocol_path, run_dir, expected_kind="cohort_census"
        )


def test_draft_and_cli_overrides_fail_closed(tmp_path: Path) -> None:
    parser = prospective_snapshot._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "acquire-census",
                "--run-dir",
                str(tmp_path / "x"),
                "--endpoint",
                "anything",
            ]
        )
    with pytest.raises(ProspectiveSnapshotError, match="frozen"):
        run_fixed_snapshot_window(
            SOURCE_PROTOCOL,
            "cohort_census",
            tmp_path / "draft",
            fetch_once=lambda current: pytest.fail("network must not be called"),
        )
