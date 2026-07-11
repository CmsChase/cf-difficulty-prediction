"""Acquire the first structure-valid response in a frozen API time window."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Final, Mapping, Sequence

import requests

from cf_diff.prospective_model import ProspectiveModelError, load_frozen_protocol


DEFAULT_PROTOCOL_PATH: Final[Path] = Path("configs/prospective_protocol_v2.json")
DEFAULT_SNAPSHOT_ROOT: Final[Path] = Path("prospective/snapshots/v2")
API_BASE_URL: Final[str] = "https://codeforces.com/api"
ZERO_HASH: Final[str] = "0" * 64
SNAPSHOT_KINDS: Final[frozenset[str]] = frozenset(
    {"cohort_census", "confirmatory_outcome"}
)
WINDOW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "endpoint",
        "query_params",
        "anchor_utc",
        "deadline_utc",
        "first_attempt_offset_seconds_after_eligibility_end",
        "retry_interval_seconds",
        "window_duration_seconds",
        "maximum_attempts",
        "attempt_start_grace_seconds",
        "request_timeout_seconds",
        "request_window",
        "missed_scheduled_attempt_policy",
        "result_shape",
        "first_success_predicate",
    }
)
ATTEMPT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "window_id",
        "sequence",
        "event_type",
        "kind",
        "protocol_id",
        "protocol_sha256",
        "scheduled_at_utc",
        "started_at_utc",
        "completed_at_utc",
        "endpoint",
        "query_params",
        "request_url",
        "http_status",
        "api_status",
        "status",
        "raw_response_path",
        "raw_response_sha256",
        "byte_count",
        "record_counts",
        "error_code",
        "error_detail",
        "previous_event_sha256",
        "event_sha256",
    }
)
SELECTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "artifact_type",
        "kind",
        "status",
        "protocol_id",
        "protocol_sha256",
        "window_id",
        "window_manifest_path",
        "window_manifest_sha256",
        "attempt_ledger_path",
        "attempt_ledger_sha256",
        "selected_attempt_number",
        "scheduled_at_utc",
        "started_at_utc",
        "completed_at_utc",
        "selected_at_utc",
        "endpoint",
        "query_params",
        "request_url",
        "http_status",
        "api_status",
        "raw_snapshot_path",
        "raw_snapshot_sha256",
        "byte_count",
        "record_counts",
    }
)


class ProspectiveSnapshotError(RuntimeError):
    """Raised when a fixed snapshot window cannot be proven valid."""


class _ResponseValidationError(ProspectiveSnapshotError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class SnapshotWindow:
    """One protocol-derived, immutable acquisition schedule."""

    kind: str
    protocol_id: str
    protocol_sha256: str
    endpoint: str
    query_params: dict[str, str]
    anchor_utc: datetime
    deadline_utc: datetime
    retry_interval_seconds: int
    maximum_attempts: int
    attempt_start_grace_seconds: int
    request_timeout_seconds: int

    @property
    def window_id(self) -> str:
        return f"{self.protocol_id}:{self.kind}"

    def scheduled_at(self, attempt_number: int) -> datetime:
        if not 1 <= attempt_number <= self.maximum_attempts:
            raise ProspectiveSnapshotError("Attempt number is outside the window.")
        return self.anchor_utc + timedelta(
            seconds=(attempt_number - 1) * self.retry_interval_seconds
        )


@dataclass(frozen=True)
class AttemptResponse:
    """Exact result of one HTTP request; no retry occurs inside this object."""

    request_url: str
    http_status: int
    raw_bytes: bytes


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ProspectiveSnapshotError(f"{field} must be a UTC timestamp.")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ProspectiveSnapshotError(f"{field} is not ISO-8601.") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProspectiveSnapshotError(f"{field} must include a UTC offset.")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ProspectiveSnapshotError("Timestamp must be timezone-aware.")
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProspectiveSnapshotError(f"Non-canonical JSON value: {error}") from error


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ProspectiveSnapshotError(f"Required snapshot file is missing: {path}")
    return _sha256_bytes(path.read_bytes())


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _ResponseValidationError(
                "invalid_json", f"Duplicate JSON key: {key}"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise _ResponseValidationError(
        "invalid_json", f"Non-finite JSON constant: {value}"
    )


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProspectiveSnapshotError(f"Cannot read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ProspectiveSnapshotError(f"{label} must be a JSON object.")
    return payload


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProspectiveSnapshotError(f"{field} must be a positive integer.")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProspectiveSnapshotError(f"{field} must be a non-negative integer.")
    return value


def _relative_artifact_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ProspectiveSnapshotError(f"{field} must be a relative POSIX path.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ProspectiveSnapshotError(f"{field} is not a safe relative path.")
    return path


def load_snapshot_window(
    protocol_path: Path,
    kind: str,
) -> SnapshotWindow:
    """Load one exact schedule and reject prose-only or inconsistent settings."""

    if kind not in SNAPSHOT_KINDS:
        raise ProspectiveSnapshotError(f"Unknown snapshot kind: {kind!r}")
    try:
        protocol = load_frozen_protocol(protocol_path)
    except ProspectiveModelError as error:
        raise ProspectiveSnapshotError(str(error)) from error
    protocol_sha = _sha256_file(protocol_path)
    windows = protocol.get("fixed_snapshot_windows")
    cohort = protocol.get("cohort")
    if not isinstance(windows, dict) or not isinstance(cohort, dict):
        raise ProspectiveSnapshotError("Protocol lacks fixed snapshot settings.")
    value = windows.get(kind)
    if not isinstance(value, dict) or frozenset(value) != WINDOW_FIELDS:
        raise ProspectiveSnapshotError(
            f"Protocol {kind} window must have the exact structured fields."
        )
    expected_endpoint = {
        "cohort_census": "contest.list",
        "confirmatory_outcome": "problemset.problems",
    }[kind]
    expected_query = {
        "cohort_census": {"gym": "false", "lang": "en"},
        "confirmatory_outcome": {"lang": "en"},
    }[kind]
    if value.get("endpoint") != expected_endpoint or value.get(
        "query_params"
    ) != expected_query:
        raise ProspectiveSnapshotError("Snapshot endpoint or query is not frozen.")
    if value.get("request_window") != "half_open" or value.get(
        "missed_scheduled_attempt_policy"
    ) != "invalidate_window":
        raise ProspectiveSnapshotError("Snapshot window policy is not fail closed.")
    interval = _positive_int(value.get("retry_interval_seconds"), "retry interval")
    duration = _positive_int(value.get("window_duration_seconds"), "duration")
    maximum = _positive_int(value.get("maximum_attempts"), "maximum attempts")
    grace = _positive_int(
        value.get("attempt_start_grace_seconds"), "attempt start grace"
    )
    timeout = _positive_int(value.get("request_timeout_seconds"), "request timeout")
    offset = _positive_int(
        value.get("first_attempt_offset_seconds_after_eligibility_end"),
        "first attempt offset",
    )
    anchor = _parse_utc(value.get("anchor_utc"), f"{kind}.anchor_utc")
    deadline = _parse_utc(value.get("deadline_utc"), f"{kind}.deadline_utc")
    eligibility_end = _parse_utc(
        cohort.get("eligibility_end_utc"), "cohort.eligibility_end_utc"
    )
    if anchor != eligibility_end + timedelta(seconds=offset):
        raise ProspectiveSnapshotError("Snapshot anchor and frozen offset disagree.")
    if deadline != anchor + timedelta(seconds=duration):
        raise ProspectiveSnapshotError("Snapshot deadline and duration disagree.")
    if maximum != 48 or interval != 1800 or duration != 86400:
        raise ProspectiveSnapshotError("Snapshot retry schedule must be 48 half-hours.")
    if anchor + maximum * timedelta(seconds=interval) != deadline:
        raise ProspectiveSnapshotError("Snapshot slots do not form a half-open day.")
    if grace >= interval or timeout > interval:
        raise ProspectiveSnapshotError("Snapshot grace or timeout is too large.")
    protocol_id = protocol.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ProspectiveSnapshotError("Protocol id is invalid.")
    return SnapshotWindow(
        kind=kind,
        protocol_id=protocol_id,
        protocol_sha256=protocol_sha,
        endpoint=expected_endpoint,
        query_params=dict(expected_query),
        anchor_utc=anchor,
        deadline_utc=deadline,
        retry_interval_seconds=interval,
        maximum_attempts=maximum,
        attempt_start_grace_seconds=grace,
        request_timeout_seconds=timeout,
    )


def _expected_request_url(window: SnapshotWindow) -> str:
    prepared = requests.Request(
        "GET",
        f"{API_BASE_URL}/{window.endpoint}",
        params=window.query_params,
    ).prepare()
    if prepared.url is None:
        raise ProspectiveSnapshotError("Cannot construct the frozen request URL.")
    return prepared.url


def _fetch_once(window: SnapshotWindow) -> AttemptResponse:
    try:
        response = requests.get(
            f"{API_BASE_URL}/{window.endpoint}",
            params=window.query_params,
            timeout=window.request_timeout_seconds,
        )
    except requests.RequestException as error:
        raise ProspectiveSnapshotError(
            f"transport_error:{type(error).__name__}"
        ) from error
    return AttemptResponse(
        request_url=response.url,
        http_status=response.status_code,
        raw_bytes=response.content,
    )


def _validate_payload(
    window: SnapshotWindow,
    response: AttemptResponse,
) -> tuple[str, dict[str, int]]:
    if response.request_url != _expected_request_url(window):
        raise _ResponseValidationError(
            "invalid_structure", "Response URL differs from the frozen request."
        )
    if not 200 <= response.http_status < 300:
        raise _ResponseValidationError(
            "http_error", f"HTTP status {response.http_status} is not successful."
        )
    try:
        payload = json.loads(
            response.raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except _ResponseValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise _ResponseValidationError(
            "invalid_json", "Response is not valid UTF-8 JSON."
        ) from error
    if not isinstance(payload, dict):
        raise _ResponseValidationError(
            "invalid_structure", "Top-level API payload is not an object."
        )
    api_status = payload.get("status")
    if api_status != "OK":
        raise _ResponseValidationError(
            "api_failure", "Top-level Codeforces API status is not OK."
        )
    result = payload.get("result")
    if window.kind == "cohort_census":
        if not isinstance(result, list):
            raise _ResponseValidationError(
                "invalid_structure", "contest.list result is not a list."
            )
        ids: set[int] = set()
        for contest in result:
            if not isinstance(contest, dict):
                raise _ResponseValidationError(
                    "invalid_structure", "Contest record is not an object."
                )
            contest_id = contest.get("id")
            start = contest.get("startTimeSeconds")
            if (
                isinstance(contest_id, bool)
                or not isinstance(contest_id, int)
                or contest_id < 1
                or isinstance(start, bool)
                or not isinstance(start, int)
            ):
                raise _ResponseValidationError(
                    "invalid_structure", "Contest id or start time is invalid."
                )
            if contest_id in ids:
                raise _ResponseValidationError(
                    "invalid_structure", "Contest ids are not unique."
                )
            ids.add(contest_id)
        return "OK", {"contests": len(result)}

    if not isinstance(result, dict) or not isinstance(result.get("problems"), list):
        raise _ResponseValidationError(
            "invalid_structure", "problemset result lacks a problems list."
        )
    problems = result["problems"]
    keys: set[tuple[int, str]] = set()
    for problem in problems:
        if not isinstance(problem, dict):
            raise _ResponseValidationError(
                "invalid_structure", "Problem record is not an object."
            )
        contest_id = problem.get("contestId")
        index = problem.get("index")
        if (
            isinstance(contest_id, bool)
            or not isinstance(contest_id, int)
            or contest_id < 1
            or not isinstance(index, str)
            or not re.fullmatch(r"[A-Za-z0-9]+", index.strip())
        ):
            raise _ResponseValidationError(
                "invalid_structure", "Problem contestId or index is invalid."
            )
        key = (contest_id, index.strip().upper())
        if key in keys:
            raise _ResponseValidationError(
                "invalid_structure", "Problem keys are not unique."
            )
        keys.add(key)
    statistics = result.get("problemStatistics")
    statistics_count = len(statistics) if isinstance(statistics, list) else 0
    return "OK", {
        "problems": len(problems),
        "problem_statistics": statistics_count,
    }


def _attempt_hash(event_without_hash: Mapping[str, object]) -> str:
    return _sha256_bytes(_canonical_json_bytes(event_without_hash))


def _verify_attempts(path: Path, window: SnapshotWindow) -> list[dict[str, object]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ProspectiveSnapshotError("Attempt ledger has a partial final line.")
    previous = ZERO_HASH
    events: list[dict[str, object]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        try:
            event = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ProspectiveSnapshotError(
                f"Attempt ledger line {number} is invalid JSON."
            ) from error
        if not isinstance(event, dict) or frozenset(event) != ATTEMPT_FIELDS:
            raise ProspectiveSnapshotError(
                f"Attempt ledger line {number} has non-exact fields."
            )
        if (
            event.get("schema_version") != 1
            or event.get("window_id") != window.window_id
            or event.get("sequence") != number
            or event.get("event_type") != "snapshot_attempt"
            or event.get("kind") != window.kind
            or event.get("protocol_id") != window.protocol_id
            or event.get("protocol_sha256") != window.protocol_sha256
            or event.get("endpoint") != window.endpoint
            or event.get("query_params") != window.query_params
            or event.get("previous_event_sha256") != previous
        ):
            raise ProspectiveSnapshotError(
                f"Attempt ledger line {number} violates window identity."
            )
        scheduled = _parse_utc(event.get("scheduled_at_utc"), "scheduled_at_utc")
        started = _parse_utc(event.get("started_at_utc"), "started_at_utc")
        completed = _parse_utc(event.get("completed_at_utc"), "completed_at_utc")
        if (
            event.get("scheduled_at_utc") != _format_utc(scheduled)
            or event.get("started_at_utc") != _format_utc(started)
            or event.get("completed_at_utc") != _format_utc(completed)
            or scheduled != window.scheduled_at(number)
            or not scheduled
            <= started
            <= scheduled + timedelta(seconds=window.attempt_start_grace_seconds)
            or completed < started
        ):
            raise ProspectiveSnapshotError(
                f"Attempt ledger line {number} has invalid timing."
            )
        status = event.get("status")
        if status not in {"failed", "selected"}:
            raise ProspectiveSnapshotError("Attempt status is invalid.")
        raw_path = event.get("raw_response_path")
        raw_sha = event.get("raw_response_sha256")
        byte_count = event.get("byte_count")
        if raw_path is None:
            if raw_sha is not None or byte_count is not None:
                raise ProspectiveSnapshotError("Attempt raw evidence pair is invalid.")
        else:
            relative_raw = _relative_artifact_path(
                raw_path, "raw_response_path"
            )
            if not isinstance(raw_sha, str) or not re.fullmatch(
                r"[0-9a-f]{64}", raw_sha
            ):
                raise ProspectiveSnapshotError("Attempt raw SHA-256 is invalid.")
            expected_bytes = _nonnegative_int(byte_count, "attempt byte_count")
            evidence_path = path.parent / relative_raw
            if (
                not evidence_path.is_file()
                or _sha256_file(evidence_path) != raw_sha
                or evidence_path.stat().st_size != expected_bytes
            ):
                raise ProspectiveSnapshotError(
                    "Attempt raw response evidence does not match its event."
                )
        if status == "selected":
            if (
                event.get("error_code") is not None
                or event.get("error_detail") is not None
                or event.get("api_status") != "OK"
                or event.get("http_status") != 200
                or not isinstance(event.get("record_counts"), dict)
                or completed >= window.deadline_utc
            ):
                raise ProspectiveSnapshotError("Selected attempt is inconsistent.")
        else:
            if not isinstance(event.get("error_code"), str):
                raise ProspectiveSnapshotError("Failed attempt lacks an error code.")
        digest = event.get("event_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ProspectiveSnapshotError("Attempt event SHA-256 is invalid.")
        unhashed = dict(event)
        del unhashed["event_sha256"]
        if digest != _attempt_hash(unhashed):
            raise ProspectiveSnapshotError("Attempt event hash is invalid.")
        previous = digest
        events.append(event)
    selected_positions = [
        index for index, event in enumerate(events) if event["status"] == "selected"
    ]
    if selected_positions and selected_positions != [len(events) - 1]:
        raise ProspectiveSnapshotError("Attempts continued after first success.")
    return events


def _append_attempt(
    path: Path,
    payload: dict[str, object],
    window: SnapshotWindow,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with lock_path.open("x", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError as error:
        raise ProspectiveSnapshotError("Attempt ledger is locked.") from error
    try:
        original = path.read_bytes() if path.exists() else b""
        events = _verify_attempts(path, window)
        event = {
            "schema_version": 1,
            "window_id": window.window_id,
            "sequence": len(events) + 1,
            "event_type": "snapshot_attempt",
            "kind": window.kind,
            "protocol_id": window.protocol_id,
            "protocol_sha256": window.protocol_sha256,
            **payload,
            "previous_event_sha256": (
                events[-1]["event_sha256"] if events else ZERO_HASH
            ),
        }
        event["event_sha256"] = _attempt_hash(event)
        data = original + _canonical_json_bytes(event) + b"\n"
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if (path.read_bytes() if path.exists() else b"") != original:
            raise ProspectiveSnapshotError("Attempt ledger changed concurrently.")
        os.replace(temporary, path)
        _verify_attempts(path, window)
        return event
    finally:
        temporary.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


def _write_bytes_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise ProspectiveSnapshotError(f"Artifact already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_exclusive(path: Path, value: object) -> None:
    raw = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _write_bytes_exclusive(path, raw)


def _window_manifest(window: SnapshotWindow) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "fixed_api_snapshot_window",
        "window_id": window.window_id,
        "kind": window.kind,
        "protocol_id": window.protocol_id,
        "protocol_sha256": window.protocol_sha256,
        "endpoint": window.endpoint,
        "query_params": window.query_params,
        "anchor_utc": _format_utc(window.anchor_utc),
        "deadline_utc": _format_utc(window.deadline_utc),
        "retry_interval_seconds": window.retry_interval_seconds,
        "maximum_attempts": window.maximum_attempts,
        "attempt_start_grace_seconds": window.attempt_start_grace_seconds,
        "request_timeout_seconds": window.request_timeout_seconds,
        "request_window": "half_open",
        "missed_scheduled_attempt_policy": "invalidate_window",
    }


def _ensure_window_manifest(run_dir: Path, window: SnapshotWindow) -> Path:
    path = run_dir / "window.json"
    expected = _window_manifest(window)
    if path.exists():
        if _read_json(path, "window manifest") != expected:
            raise ProspectiveSnapshotError("Existing window manifest does not match.")
    else:
        _write_json_exclusive(path, expected)
    return path


def _selection_from_event(
    run_dir: Path,
    window: SnapshotWindow,
    event: Mapping[str, object],
) -> dict[str, object]:
    window_path = run_dir / "window.json"
    attempt_path = run_dir / "attempts.jsonl"
    return {
        "schema_version": 1,
        "artifact_type": "fixed_api_snapshot_selection",
        "kind": window.kind,
        "status": "selected",
        "protocol_id": window.protocol_id,
        "protocol_sha256": window.protocol_sha256,
        "window_id": window.window_id,
        "window_manifest_path": "window.json",
        "window_manifest_sha256": _sha256_file(window_path),
        "attempt_ledger_path": "attempts.jsonl",
        "attempt_ledger_sha256": _sha256_file(attempt_path),
        "selected_attempt_number": event["sequence"],
        "scheduled_at_utc": event["scheduled_at_utc"],
        "started_at_utc": event["started_at_utc"],
        "completed_at_utc": event["completed_at_utc"],
        "selected_at_utc": event["completed_at_utc"],
        "endpoint": event["endpoint"],
        "query_params": event["query_params"],
        "request_url": event["request_url"],
        "http_status": event["http_status"],
        "api_status": event["api_status"],
        "raw_snapshot_path": event["raw_response_path"],
        "raw_snapshot_sha256": event["raw_response_sha256"],
        "byte_count": event["byte_count"],
        "record_counts": event["record_counts"],
    }


def _write_terminal(
    run_dir: Path,
    window: SnapshotWindow,
    *,
    status: str,
    reason_code: str,
    attempt_number: int,
    recorded_at: datetime,
) -> None:
    terminal = {
        "schema_version": 1,
        "artifact_type": "fixed_api_snapshot_terminal",
        "window_id": window.window_id,
        "kind": window.kind,
        "status": status,
        "reason_code": reason_code,
        "attempt_number": attempt_number,
        "recorded_at_utc": _format_utc(recorded_at),
        "attempt_ledger_sha256": (
            _sha256_file(run_dir / "attempts.jsonl")
            if (run_dir / "attempts.jsonl").exists()
            else None
        ),
    }
    _write_json_exclusive(run_dir / "terminal.json", terminal)


def _wait_until(
    target: datetime,
    *,
    clock: Callable[[], datetime],
    sleep_fn: Callable[[float], None],
) -> datetime:
    current = clock().astimezone(timezone.utc)
    while current < target:
        sleep_fn(min((target - current).total_seconds(), 30.0))
        updated = clock().astimezone(timezone.utc)
        if updated <= current:
            raise ProspectiveSnapshotError("Clock did not advance while waiting.")
        current = updated
    return current


def run_fixed_snapshot_window(
    protocol_path: Path,
    kind: str,
    run_dir: Path,
    *,
    clock: Callable[[], datetime] = _utc_now,
    sleep_fn: Callable[[float], None] = time.sleep,
    fetch_once: Callable[[SnapshotWindow], AttemptResponse] = _fetch_once,
) -> dict[str, object]:
    """Run or safely resume a fixed window; testing hooks are not exposed by CLI."""

    window = load_snapshot_window(protocol_path, kind)
    run_dir.mkdir(parents=True, exist_ok=True)
    _ensure_window_manifest(run_dir, window)
    selection_path = run_dir / "selection.json"
    if selection_path.exists():
        return verify_snapshot_selection(protocol_path, run_dir, expected_kind=kind)
    if (run_dir / "terminal.json").exists():
        raise ProspectiveSnapshotError("Snapshot window already ended without selection.")
    attempt_path = run_dir / "attempts.jsonl"
    events = _verify_attempts(attempt_path, window)
    if events and events[-1]["status"] == "selected":
        selection = _selection_from_event(run_dir, window, events[-1])
        _write_json_exclusive(selection_path, selection)
        return verify_snapshot_selection(protocol_path, run_dir, expected_kind=kind)

    for attempt_number in range(len(events) + 1, window.maximum_attempts + 1):
        scheduled = window.scheduled_at(attempt_number)
        started = _wait_until(scheduled, clock=clock, sleep_fn=sleep_fn)
        if started > scheduled + timedelta(
            seconds=window.attempt_start_grace_seconds
        ):
            _write_terminal(
                run_dir,
                window,
                status="invalidated",
                reason_code="missed_scheduled_attempt",
                attempt_number=attempt_number,
                recorded_at=started,
            )
            raise ProspectiveSnapshotError("A scheduled attempt was missed.")
        response: AttemptResponse | None = None
        validation_error: _ResponseValidationError | None = None
        transport_error: str | None = None
        try:
            response = fetch_once(window)
        except ProspectiveSnapshotError as error:
            transport_error = "transport_error"
        completed = clock().astimezone(timezone.utc)
        if completed < started:
            raise ProspectiveSnapshotError("System clock moved backwards during request.")
        raw_relative: str | None = None
        raw_sha: str | None = None
        byte_count: int | None = None
        if response is not None:
            if not isinstance(response.raw_bytes, bytes):
                raise ProspectiveSnapshotError("Fetcher did not return exact bytes.")
            raw_relative = f"responses/attempt-{attempt_number:04d}.raw"
            raw_path = run_dir / raw_relative
            _write_bytes_exclusive(raw_path, response.raw_bytes)
            raw_sha = _sha256_bytes(response.raw_bytes)
            byte_count = len(response.raw_bytes)
            try:
                api_status, counts = _validate_payload(window, response)
            except _ResponseValidationError as error:
                validation_error = error
                api_status = None
                counts = None
        else:
            api_status = None
            counts = None

        selected = (
            response is not None
            and validation_error is None
            and transport_error is None
            and completed < window.deadline_utc
        )
        if completed >= window.deadline_utc:
            error_code = "completed_after_deadline"
            error_detail = "Request completed after the half-open window deadline."
        elif transport_error is not None:
            error_code = transport_error
            error_detail = "The single scheduled HTTP request failed."
        elif validation_error is not None:
            error_code = validation_error.code
            error_detail = validation_error.detail
        else:
            error_code = None
            error_detail = None
        payload = {
            "scheduled_at_utc": _format_utc(scheduled),
            "started_at_utc": _format_utc(started),
            "completed_at_utc": _format_utc(completed),
            "endpoint": window.endpoint,
            "query_params": window.query_params,
            "request_url": (
                response.request_url if response is not None else _expected_request_url(window)
            ),
            "http_status": response.http_status if response is not None else None,
            "api_status": api_status,
            "status": "selected" if selected else "failed",
            "raw_response_path": raw_relative,
            "raw_response_sha256": raw_sha,
            "byte_count": byte_count,
            "record_counts": counts,
            "error_code": error_code,
            "error_detail": error_detail,
        }
        try:
            event = _append_attempt(attempt_path, payload, window)
        except Exception:
            if raw_relative is not None:
                (run_dir / raw_relative).unlink(missing_ok=True)
            raise
        if selected:
            selection = _selection_from_event(run_dir, window, event)
            _write_json_exclusive(selection_path, selection)
            return verify_snapshot_selection(
                protocol_path, run_dir, expected_kind=kind
            )
        if completed >= window.deadline_utc:
            _write_terminal(
                run_dir,
                window,
                status="invalidated",
                reason_code="completed_after_deadline",
                attempt_number=attempt_number,
                recorded_at=completed,
            )
            raise ProspectiveSnapshotError("Request completed after the deadline.")

    terminal_time = clock().astimezone(timezone.utc)
    _write_terminal(
        run_dir,
        window,
        status="exhausted",
        reason_code="window_exhausted",
        attempt_number=window.maximum_attempts,
        recorded_at=terminal_time,
    )
    raise ProspectiveSnapshotError("All 48 fixed snapshot attempts failed.")


def verify_snapshot_selection(
    protocol_path: Path,
    run_dir: Path,
    *,
    expected_kind: str | None = None,
) -> dict[str, object]:
    """Re-derive a sealed selection without making a network request."""

    selection = _read_json(run_dir / "selection.json", "selection manifest")
    if frozenset(selection) != SELECTION_FIELDS:
        raise ProspectiveSnapshotError("Selection manifest has non-exact fields.")
    kind = selection.get("kind")
    if not isinstance(kind, str) or (
        expected_kind is not None and kind != expected_kind
    ):
        raise ProspectiveSnapshotError("Selection kind mismatch.")
    window = load_snapshot_window(protocol_path, kind)
    if (
        selection.get("schema_version") != 1
        or selection.get("artifact_type") != "fixed_api_snapshot_selection"
        or selection.get("status") != "selected"
        or selection.get("protocol_id") != window.protocol_id
        or selection.get("protocol_sha256") != window.protocol_sha256
        or selection.get("window_id") != window.window_id
        or selection.get("endpoint") != window.endpoint
        or selection.get("query_params") != window.query_params
    ):
        raise ProspectiveSnapshotError("Selection does not match the frozen window.")
    window_relative = _relative_artifact_path(
        selection.get("window_manifest_path"), "window_manifest_path"
    )
    attempt_relative = _relative_artifact_path(
        selection.get("attempt_ledger_path"), "attempt_ledger_path"
    )
    raw_relative = _relative_artifact_path(
        selection.get("raw_snapshot_path"), "raw_snapshot_path"
    )
    window_path = run_dir / window_relative
    attempt_path = run_dir / attempt_relative
    raw_path = run_dir / raw_relative
    if _read_json(window_path, "window manifest") != _window_manifest(window):
        raise ProspectiveSnapshotError("Window manifest was altered.")
    if selection.get("window_manifest_sha256") != _sha256_file(window_path):
        raise ProspectiveSnapshotError("Window manifest SHA-256 mismatch.")
    if selection.get("attempt_ledger_sha256") != _sha256_file(attempt_path):
        raise ProspectiveSnapshotError("Attempt ledger SHA-256 mismatch.")
    if selection.get("raw_snapshot_sha256") != _sha256_file(raw_path):
        raise ProspectiveSnapshotError("Raw snapshot SHA-256 mismatch.")
    raw = raw_path.read_bytes()
    if selection.get("byte_count") != len(raw):
        raise ProspectiveSnapshotError("Raw snapshot byte count mismatch.")
    events = _verify_attempts(attempt_path, window)
    attempt_number = _positive_int(
        selection.get("selected_attempt_number"), "selected attempt number"
    )
    if not events or attempt_number != len(events):
        raise ProspectiveSnapshotError("Selection is not the first terminal success.")
    event = events[-1]
    if event.get("status") != "selected":
        raise ProspectiveSnapshotError("Selected attempt ledger event is absent.")
    expected_selection = _selection_from_event(run_dir, window, event)
    if selection != expected_selection:
        raise ProspectiveSnapshotError("Selection fields do not match its attempt.")
    response = AttemptResponse(
        request_url=str(selection["request_url"]),
        http_status=int(selection["http_status"]),
        raw_bytes=raw,
    )
    api_status, counts = _validate_payload(window, response)
    if api_status != selection["api_status"] or counts != selection["record_counts"]:
        raise ProspectiveSnapshotError("Selected response structure was altered.")
    return selection


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, kind in (
        ("acquire-census", "cohort_census"),
        ("acquire-outcome", "confirmatory_outcome"),
    ):
        subparser = subparsers.add_parser(command)
        subparser.set_defaults(kind=kind)
        subparser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
        subparser.add_argument("--run-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--kind", choices=sorted(SNAPSHOT_KINDS), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_snapshot_selection(
                args.protocol, args.run_dir, expected_kind=args.kind
            )
        else:
            result = run_fixed_snapshot_window(
                args.protocol, args.kind, args.run_dir
            )
        print(json.dumps(result, sort_keys=True))
    except (ProspectiveSnapshotError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
