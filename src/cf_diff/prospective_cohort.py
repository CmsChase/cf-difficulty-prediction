"""Finalize the fixed prospective census-to-outcome cohort mapping."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Final, Mapping, Sequence

from cf_diff.prospective_analysis import FINALIZED_COLUMNS
from cf_diff.prospective_ledger import (
    COMMITMENT_EVENT_TYPES,
    OBSERVATION_EVENT_TYPES,
    ProspectiveLedgerError,
    build_commitment_state,
    verify_commitment_ledger,
    verify_ledger,
    verify_observation_ledger,
)
from cf_diff.prospective_model import (
    PREDICTION_COLUMNS,
    ProspectiveModelError,
    load_frozen_protocol,
)
from cf_diff.prospective_snapshot import (
    ProspectiveSnapshotError,
    verify_snapshot_selection,
)


DEFAULT_PROTOCOL_PATH: Final[Path] = Path("configs/prospective_protocol_v2.json")
DEFAULT_CENSUS_RUN_DIR: Final[Path] = Path(
    "prospective/snapshots/v2/cohort-census"
)
DEFAULT_OUTCOME_RUN_DIR: Final[Path] = Path(
    "prospective/snapshots/v2/confirmatory-outcome"
)
DEFAULT_COMMITMENT_LEDGER: Final[Path] = Path(
    "prospective/ledger/commitments.jsonl"
)
DEFAULT_OBSERVATION_LEDGER: Final[Path] = Path(
    "prospective/ledger/observations.jsonl"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path("prospective/cohort/v2")

CENSUS_COLUMNS: Final[tuple[str, ...]] = (
    "contest_id",
    "contest_name",
    "contest_type",
    "phase_at_snapshot",
    "start_time_seconds",
    "scheduled_start_utc",
    "duration_seconds",
)
MAPPING_COLUMNS: Final[tuple[str, ...]] = (
    "contest_id",
    "official_start_utc",
    "coverage_event_type",
    "coverage_event_sha256",
    "public_witness_status",
    "locked_start_utc",
    "start_matches",
    "locked_problem_count",
    "official_problem_count",
    "index_set_matches",
    "disposition",
    "reason_code",
    "paired_rated_problem_count",
)


class ProspectiveCohortError(RuntimeError):
    """Raised when cohort mapping is incomplete, mutable, or inconsistent."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ProspectiveCohortError(f"{field} must be a UTC timestamp.")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ProspectiveCohortError(f"{field} is not ISO-8601.") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProspectiveCohortError(f"{field} must include a UTC offset.")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ProspectiveCohortError("Timestamp must be timezone-aware.")
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ProspectiveCohortError(f"Required cohort artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, object]:
    def reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ProspectiveCohortError(f"Duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ProspectiveCohortError(f"Non-finite JSON value in {label}: {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProspectiveCohortError(f"Cannot read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ProspectiveCohortError(f"{label} must be a JSON object.")
    return payload


def _canonical_contest_id(value: object) -> str:
    if isinstance(value, bool):
        raise ProspectiveCohortError("contest_id must be a positive integer.")
    text = str(value).strip()
    if not re.fullmatch(r"[1-9][0-9]*", text):
        raise ProspectiveCohortError(f"Invalid contest_id: {value!r}")
    return text


def _canonical_index(value: object) -> str:
    text = str(value).strip().upper()
    if not re.fullmatch(r"[A-Z0-9]+", text):
        raise ProspectiveCohortError(f"Invalid problem index: {value!r}")
    return text


def _finite(value: object, field: str) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError) as error:
        raise ProspectiveCohortError(f"{field} must be numeric.") from error
    if not math.isfinite(result):
        raise ProspectiveCohortError(f"{field} must be finite.")
    return result


def _safe_repository_path(value: object, root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ProspectiveCohortError(f"{field} must be a repository-relative path.")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise ProspectiveCohortError(f"{field} is not a safe repository path.")
    path = root.resolve() / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ProspectiveCohortError(f"{field} escapes the repository.") from error
    return path


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ProspectiveCohortError(
            f"Artifact must be inside the repository root: {path}"
        ) from error


def _load_protocol(path: Path) -> tuple[dict[str, object], str]:
    try:
        protocol = load_frozen_protocol(path)
    except ProspectiveModelError as error:
        raise ProspectiveCohortError(str(error)) from error
    cohort = protocol.get("cohort")
    if not isinstance(cohort, dict) or cohort.get("contest_exclusion_codes") != []:
        raise ProspectiveCohortError("Protocol must have no contest-exclusion bucket.")
    return protocol, _sha256_file(path)


def _selection_raw_path(run_dir: Path, selection: Mapping[str, object]) -> Path:
    value = selection.get("raw_snapshot_path")
    if not isinstance(value, str) or not value or "\\" in value:
        raise ProspectiveCohortError("Selection raw snapshot path is invalid.")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ProspectiveCohortError("Selection raw snapshot path escapes its run.")
    return run_dir / relative


def _load_census(
    raw_path: Path,
    protocol: Mapping[str, object],
) -> list[dict[str, object]]:
    payload = _read_json(raw_path, "selected census response")
    result = payload.get("result")
    if payload.get("status") != "OK" or not isinstance(result, list):
        raise ProspectiveCohortError("Selected census response is malformed.")
    cohort = protocol["cohort"]
    assert isinstance(cohort, dict)
    start = _parse_utc(cohort.get("eligibility_start_utc"), "eligibility start")
    end = _parse_utc(cohort.get("eligibility_end_utc"), "eligibility end")
    rows: list[dict[str, object]] = []
    ids: set[str] = set()
    for item in result:
        if not isinstance(item, dict):
            raise ProspectiveCohortError("Census contest record is not an object.")
        contest_id = _canonical_contest_id(item.get("id"))
        start_seconds = item.get("startTimeSeconds")
        duration = item.get("durationSeconds")
        if (
            isinstance(start_seconds, bool)
            or not isinstance(start_seconds, int)
            or isinstance(duration, bool)
            or not isinstance(duration, int)
            or duration < 0
        ):
            raise ProspectiveCohortError("Census contest timing fields are invalid.")
        name = item.get("name")
        contest_type = item.get("type")
        phase = item.get("phase")
        if not all(isinstance(value, str) for value in (name, contest_type, phase)):
            raise ProspectiveCohortError("Census descriptive fields are invalid.")
        scheduled = datetime.fromtimestamp(start_seconds, tz=timezone.utc)
        if start <= scheduled <= end:
            if contest_id in ids:
                raise ProspectiveCohortError("In-window census ids are duplicated.")
            ids.add(contest_id)
            rows.append(
                {
                    "contest_id": contest_id,
                    "contest_name": name,
                    "contest_type": contest_type,
                    "phase_at_snapshot": phase,
                    "start_time_seconds": start_seconds,
                    "scheduled_start_utc": _format_utc(scheduled),
                    "duration_seconds": duration,
                }
            )
    return sorted(rows, key=lambda row: str(row["contest_id"]))


def _load_outcomes(raw_path: Path) -> dict[tuple[str, str], dict[str, object]]:
    payload = _read_json(raw_path, "selected outcome response")
    result = payload.get("result")
    if (
        payload.get("status") != "OK"
        or not isinstance(result, dict)
        or not isinstance(result.get("problems"), list)
    ):
        raise ProspectiveCohortError("Selected outcome response is malformed.")
    outcomes: dict[tuple[str, str], dict[str, object]] = {}
    for item in result["problems"]:
        if not isinstance(item, dict):
            raise ProspectiveCohortError("Official problem record is not an object.")
        contest_id = _canonical_contest_id(item.get("contestId"))
        index = _canonical_index(item.get("index"))
        problem_type = item.get("type")
        if not isinstance(problem_type, str) or not problem_type:
            raise ProspectiveCohortError("Official problem type is missing.")
        rating_value = item.get("rating")
        rating: float | None
        if rating_value is None:
            rating = None
        elif isinstance(rating_value, bool) or not isinstance(rating_value, (int, float)):
            raise ProspectiveCohortError("Official rating is not numeric or null.")
        else:
            rating = float(rating_value)
            if not math.isfinite(rating):
                raise ProspectiveCohortError("Official rating is non-finite.")
        key = (contest_id, index)
        if key in outcomes:
            raise ProspectiveCohortError("Official problem keys are duplicated.")
        outcomes[key] = {"type": problem_type, "rating": rating}
    return outcomes


def _verify_observation_bindings(
    *,
    protocol_path: Path,
    observation_ledger_path: Path,
    census_run_dir: Path,
    outcome_run_dir: Path,
    census_selection: Mapping[str, object],
    outcome_selection: Mapping[str, object],
    repository_root: Path,
) -> None:
    try:
        summary = verify_observation_ledger(
            observation_ledger_path, protocol_path=protocol_path
        )
        events = verify_ledger(
            observation_ledger_path,
            allowed_event_types=OBSERVATION_EVENT_TYPES,
        )
    except ProspectiveLedgerError as error:
        raise ProspectiveCohortError(str(error)) from error
    if (
        summary["cohort_census_snapshots"] != 1
        or summary["confirmatory_outcome_snapshots"] != 1
    ):
        raise ProspectiveCohortError(
            "Observation chain must contain exactly one census and outcome snapshot."
        )
    expected = {
        "cohort_census_snapshot": (
            census_run_dir / "selection.json",
            _selection_raw_path(census_run_dir, census_selection),
            census_selection,
        ),
        "confirmatory_outcome_snapshot": (
            outcome_run_dir / "selection.json",
            _selection_raw_path(outcome_run_dir, outcome_selection),
            outcome_selection,
        ),
    }
    for event_type, (selection_path, raw_path, selection) in expected.items():
        event = next(event for event in events if event["event_type"] == event_type)
        if (
            event.get("selection_manifest_path")
            != _relative_to_root(selection_path, repository_root)
            or event.get("selection_manifest_sha256") != _sha256_file(selection_path)
            or event.get("raw_snapshot_path")
            != _relative_to_root(raw_path, repository_root)
            or event.get("raw_snapshot_sha256")
            != selection.get("raw_snapshot_sha256")
        ):
            raise ProspectiveCohortError(
                f"Observation event {event_type} does not bind its selection."
            )


def _read_prediction_rows(
    event: Mapping[str, object],
    repository_root: Path,
) -> list[dict[str, object]]:
    artifact_pairs = (
        ("input_path", "input_sha256"),
        ("capture_sidecar_path", "capture_sidecar_sha256"),
        ("prediction_path", "prediction_sha256"),
        ("model_artifact_path", "model_artifact_sha256"),
        ("freeze_manifest_path", "freeze_manifest_sha256"),
    )
    resolved: dict[str, Path] = {}
    for path_field, hash_field in artifact_pairs:
        path = _safe_repository_path(event.get(path_field), repository_root, path_field)
        if event.get(hash_field) != _sha256_file(path):
            raise ProspectiveCohortError(
                f"Committed prediction artifact hash mismatch: {path_field}"
            )
        resolved[path_field] = path
    prediction_path = resolved["prediction_path"]
    try:
        with prediction_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
                raise ProspectiveCohortError("Prediction CSV header is not exact.")
            raw_rows = list(reader)
    except OSError as error:
        raise ProspectiveCohortError(f"Cannot read prediction CSV: {error}") from error
    contest_id = _canonical_contest_id(event.get("contest_id"))
    indices = event.get("indices")
    if not isinstance(indices, list):
        raise ProspectiveCohortError("Prediction event indices are invalid.")
    expected_indices = [_canonical_index(value) for value in indices]
    if len(raw_rows) != event.get("row_count") or len(raw_rows) != len(expected_indices):
        raise ProspectiveCohortError("Prediction event row count mismatch.")
    result: list[dict[str, object]] = []
    observed_indices: list[str] = []
    for raw in raw_rows:
        row_contest = _canonical_contest_id(raw["contest_id"])
        index = _canonical_index(raw["index"])
        if row_contest != contest_id:
            raise ProspectiveCohortError("Prediction CSV contains another contest.")
        observed_indices.append(index)
        if (
            raw["contest_start_utc"] != event.get("operator_contest_start_utc")
            or raw["model_artifact_sha256"] != event.get("model_artifact_sha256")
            or raw["freeze_manifest_sha256"] != event.get("freeze_manifest_sha256")
            or raw["input_file_sha256"] != event.get("input_sha256")
            or raw["capture_sidecar_sha256"] != event.get("capture_sidecar_sha256")
        ):
            raise ProspectiveCohortError("Prediction CSV provenance fields disagree.")
        result.append(
            {
                "contest_id": contest_id,
                "index": index,
                "primary_prediction": _finite(
                    raw["primary_prediction"], "primary_prediction"
                ),
                "comparator_prediction": _finite(
                    raw["comparator_prediction"], "comparator_prediction"
                ),
            }
        )
    if observed_indices != expected_indices or len(set(observed_indices)) != len(
        observed_indices
    ):
        raise ProspectiveCohortError("Prediction CSV indices disagree with commitment.")
    return result


def _csv_bytes(fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise ProspectiveCohortError(f"Cohort artifact already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_failure_report(
    output_dir: Path,
    *,
    protocol: Mapping[str, object],
    protocol_sha: str,
    details: Mapping[str, object],
    recorded_at: datetime,
) -> None:
    path = output_dir / "mapping_failure_report.json"
    if path.exists():
        return
    payload = {
        "schema_version": 1,
        "kind": "confirmatory_mapping_failure",
        "status": "failed",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "recorded_at_utc": _format_utc(recorded_at),
        **details,
    }
    _write_exclusive(path, _json_bytes(payload))


def finalize_cohort(
    protocol_path: Path,
    census_run_dir: Path,
    outcome_run_dir: Path,
    commitment_ledger_path: Path,
    observation_ledger_path: Path,
    output_dir: Path,
    *,
    repository_root: Path = Path("."),
    clock: Callable[[], datetime] = _utc_now,
) -> dict[str, object]:
    """Create the only confirmatory analysis input from fixed public evidence."""

    protocol, protocol_sha = _load_protocol(protocol_path)
    try:
        census_selection = verify_snapshot_selection(
            protocol_path, census_run_dir, expected_kind="cohort_census"
        )
        outcome_selection = verify_snapshot_selection(
            protocol_path, outcome_run_dir, expected_kind="confirmatory_outcome"
        )
    except ProspectiveSnapshotError as error:
        raise ProspectiveCohortError(str(error)) from error
    try:
        verify_commitment_ledger(
            commitment_ledger_path, protocol_path=protocol_path
        )
        states = build_commitment_state(commitment_ledger_path)
    except ProspectiveLedgerError as error:
        raise ProspectiveCohortError(str(error)) from error
    _verify_observation_bindings(
        protocol_path=protocol_path,
        observation_ledger_path=observation_ledger_path,
        census_run_dir=census_run_dir,
        outcome_run_dir=outcome_run_dir,
        census_selection=census_selection,
        outcome_selection=outcome_selection,
        repository_root=repository_root,
    )
    census_raw = _selection_raw_path(census_run_dir, census_selection)
    outcome_raw = _selection_raw_path(outcome_run_dir, outcome_selection)
    census_rows = _load_census(census_raw, protocol)
    outcomes = _load_outcomes(outcome_raw)
    census_by_id = {str(row["contest_id"]): row for row in census_rows}
    census_ids = set(census_by_id)
    state_ids = set(states)
    unmapped = sorted(census_ids - state_ids)
    ledger_extra = sorted(state_ids - census_ids)
    integrity_details: dict[str, object] = {
        "census_contest_count": len(census_ids),
        "unmapped_contest_ids": unmapped,
        "ledger_not_in_census_ids": ledger_extra,
        "start_mismatch_ids": [],
        "index_mismatch_ids": [],
        "qualified_mismatch_without_invalidation_ids": [],
    }
    if unmapped or ledger_extra:
        _write_failure_report(
            output_dir,
            protocol=protocol,
            protocol_sha=protocol_sha,
            details=integrity_details,
            recorded_at=clock(),
        )
        raise ProspectiveCohortError(
            "Census and commitment ledger do not map exactly once."
        )

    prediction_rows_by_contest: dict[str, list[dict[str, object]]] = {}
    global_keys: set[tuple[str, str]] = set()
    for contest_id, state in states.items():
        base = state["base_event"]
        assert isinstance(base, Mapping)
        if base.get("event_type") != "prediction_commitment":
            continue
        rows = _read_prediction_rows(base, repository_root)
        for row in rows:
            key = (contest_id, str(row["index"]))
            if key in global_keys:
                raise ProspectiveCohortError(
                    "Prediction keys are not globally unique."
                )
            global_keys.add(key)
        prediction_rows_by_contest[contest_id] = rows

    mapping_rows: list[dict[str, object]] = []
    disposition_rows: list[dict[str, object]] = []
    finalized_rows: list[dict[str, object]] = []
    start_mismatches: list[str] = []
    index_mismatches: list[str] = []
    pending_invalidations: list[str] = []
    outcome_indices: dict[str, set[str]] = {}
    for contest_id, index in outcomes:
        outcome_indices.setdefault(contest_id, set()).add(index)

    for contest_id in sorted(census_ids):
        census = census_by_id[contest_id]
        state = states[contest_id]
        base = state["base_event"]
        assert isinstance(base, Mapping)
        official_start = str(census["scheduled_start_utc"])
        locked_start = str(base["operator_contest_start_utc"])
        start_matches = official_start == locked_start
        official_indices = outcome_indices.get(contest_id, set())
        rows = prediction_rows_by_contest.get(contest_id, [])
        locked_indices = {str(row["index"]) for row in rows}
        index_matches: bool | None = None
        if base.get("event_type") == "prediction_commitment":
            index_matches = locked_indices == official_indices
            if not start_matches:
                start_mismatches.append(contest_id)
            if not index_matches:
                index_mismatches.append(contest_id)

        invalidation = state.get("invalidation")
        witness = state.get("witness")
        if base.get("event_type") == "operational_miss":
            disposition = "operational_miss"
            reason_code = str(base["reason_code"])
        elif (not start_matches or index_matches is False) and bool(
            state.get("qualified")
        ):
            pending_invalidations.append(contest_id)
            disposition = "operational_miss"
            reason_code = (
                "official_start_mismatch"
                if not start_matches
                else "official_index_set_mismatch"
            )
        elif bool(state.get("qualified")) and start_matches and index_matches:
            disposition = "qualified_prediction"
            reason_code = None
        else:
            disposition = "operational_miss"
            if isinstance(invalidation, Mapping):
                reason_code = str(invalidation["reason_code"])
            elif not isinstance(witness, Mapping):
                reason_code = "external_witness_missing"
            elif witness.get("timely") is not True:
                reason_code = "external_witness_late"
            elif not start_matches:
                reason_code = "official_start_mismatch"
            else:
                reason_code = "official_index_set_mismatch"

        paired_count = 0
        for row in rows:
            key = (contest_id, str(row["index"]))
            official = outcomes.get(key)
            if official is None:
                status = "absent_from_snapshot"
                present = False
                problem_type = ""
                rating_text = ""
            elif official["type"] != "PROGRAMMING":
                status = "non_programming"
                present = True
                problem_type = str(official["type"])
                rating_text = ""
            elif official["rating"] is None:
                status = "missing_rating"
                present = True
                problem_type = "PROGRAMMING"
                rating_text = ""
            else:
                status = "paired"
                present = True
                problem_type = "PROGRAMMING"
                rating_text = format(float(official["rating"]), ".17g")
                paired_count += 1
            finalized_rows.append(
                {
                    "contest_id": contest_id,
                    "index": row["index"],
                    "primary_prediction": format(
                        float(row["primary_prediction"]), ".17g"
                    ),
                    "comparator_prediction": format(
                        float(row["comparator_prediction"]), ".17g"
                    ),
                    "official_problem_present": "true" if present else "false",
                    "official_problem_type": problem_type,
                    "official_rating": rating_text,
                    "outcome_status": status,
                    "eligible_for_analysis": (
                        "true"
                        if status == "paired"
                        and disposition == "qualified_prediction"
                        else "false"
                    ),
                }
            )
        witness_status = (
            "timely"
            if isinstance(witness, Mapping) and witness.get("timely") is True
            else "late"
            if isinstance(witness, Mapping)
            else "missing"
        )
        mapping_rows.append(
            {
                "contest_id": contest_id,
                "official_start_utc": official_start,
                "coverage_event_type": base["event_type"],
                "coverage_event_sha256": base["event_sha256"],
                "public_witness_status": witness_status,
                "locked_start_utc": locked_start,
                "start_matches": "true" if start_matches else "false",
                "locked_problem_count": len(rows),
                "official_problem_count": len(official_indices),
                "index_set_matches": (
                    ""
                    if index_matches is None
                    else "true"
                    if index_matches
                    else "false"
                ),
                "disposition": disposition,
                "reason_code": reason_code or "",
                "paired_rated_problem_count": paired_count,
            }
        )
        disposition_rows.append(
            {
                "contest_id": contest_id,
                "disposition": disposition,
                "locked_prediction_row_count": len(rows),
                "reason_code": reason_code,
            }
        )

    integrity_details["start_mismatch_ids"] = start_mismatches
    integrity_details["index_mismatch_ids"] = index_mismatches
    integrity_details[
        "qualified_mismatch_without_invalidation_ids"
    ] = pending_invalidations
    if pending_invalidations:
        _write_failure_report(
            output_dir,
            protocol=protocol,
            protocol_sha=protocol_sha,
            details=integrity_details,
            recorded_at=clock(),
        )
        raise ProspectiveCohortError(
            "Official start/index mismatch requires an append-only prediction invalidation."
        )

    finalized_rows.sort(key=lambda row: (str(row["contest_id"]), str(row["index"])))
    status_counts = {
        status: sum(row["outcome_status"] == status for row in finalized_rows)
        for status in (
            "paired",
            "missing_rating",
            "non_programming",
            "absent_from_snapshot",
        )
    }
    report = {
        "schema_version": 1,
        "kind": "confirmatory_mapping_report",
        "status": "finalized",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "census_integrity_status": "passed",
        "outcome_integrity_status": "passed",
        "census_contest_count": len(census_rows),
        "mapped_once_contest_count": len(census_rows),
        "unmapped_contest_count": 0,
        "multiply_mapped_contest_count": 0,
        "qualified_prediction_contest_count": sum(
            row["disposition"] == "qualified_prediction"
            for row in disposition_rows
        ),
        "operational_miss_contest_count": sum(
            row["disposition"] == "operational_miss" for row in disposition_rows
        ),
        "timely_coverage_event_count": sum(
            isinstance(state.get("witness"), Mapping)
            and state["witness"].get("timely") is True
            for state in states.values()
        ),
        "late_coverage_event_count": sum(
            isinstance(state.get("witness"), Mapping)
            and state["witness"].get("timely") is False
            for state in states.values()
        ),
        "missing_coverage_witness_count": sum(
            state.get("witness") is None for state in states.values()
        ),
        "prespecified_exclusion_contest_count": 0,
        "locked_prediction_contest_count": sum(
            int(row["locked_prediction_row_count"]) > 0
            for row in disposition_rows
        ),
        "locked_prediction_row_count": len(finalized_rows),
        "paired_problem_row_count": status_counts["paired"],
        "missing_rating_row_count": status_counts["missing_rating"],
        "non_programming_row_count": status_counts["non_programming"],
        "absent_from_snapshot_row_count": status_counts["absent_from_snapshot"],
        "contest_dispositions": disposition_rows,
    }
    census_bytes = _csv_bytes(CENSUS_COLUMNS, census_rows)
    mapping_bytes = _csv_bytes(MAPPING_COLUMNS, mapping_rows)
    finalized_bytes = _csv_bytes(FINALIZED_COLUMNS, finalized_rows)
    report_bytes = _json_bytes(report)
    finalized_at = clock().astimezone(timezone.utc)
    census_selected = _parse_utc(
        census_selection["selected_at_utc"], "census selected_at_utc"
    )
    outcome_selected = _parse_utc(
        outcome_selection["selected_at_utc"], "outcome selected_at_utc"
    )
    if finalized_at < max(census_selected, outcome_selected):
        raise ProspectiveCohortError("Finalization clock predates selected evidence.")
    analysis = protocol.get("confirmatory_analysis")
    assert isinstance(analysis, dict)
    manifest = {
        "schema_version": 1,
        "kind": "confirmatory_analysis_input",
        "status": "finalized",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "finalized_at_utc": _format_utc(finalized_at),
        "analysis_not_before_utc": analysis["earliest_execution_utc"],
        "finalized_csv": {
            "path": "confirmatory_outcomes.csv",
            "sha256": hashlib.sha256(finalized_bytes).hexdigest(),
            "row_count": len(finalized_rows),
        },
        "mapping_report": {
            "path": "mapping_report.json",
            "sha256": hashlib.sha256(report_bytes).hexdigest(),
        },
        "commitment_ledger_sha256": _sha256_file(commitment_ledger_path),
        "observation_ledger_sha256": _sha256_file(observation_ledger_path),
        "cohort_census_selection": {
            "selection_manifest_sha256": _sha256_file(
                census_run_dir / "selection.json"
            ),
            "raw_snapshot_sha256": census_selection["raw_snapshot_sha256"],
            "selected_at_utc": census_selection["selected_at_utc"],
        },
        "confirmatory_outcome_selection": {
            "selection_manifest_sha256": _sha256_file(
                outcome_run_dir / "selection.json"
            ),
            "raw_snapshot_sha256": outcome_selection["raw_snapshot_sha256"],
            "selected_at_utc": outcome_selection["selected_at_utc"],
        },
    }
    targets = {
        "cohort_census.csv": census_bytes,
        "cohort_mapping.csv": mapping_bytes,
        "confirmatory_outcomes.csv": finalized_bytes,
        "mapping_report.json": report_bytes,
        "confirmatory_analysis_input.json": _json_bytes(manifest),
    }
    existing = [name for name in targets if (output_dir / name).exists()]
    if existing:
        raise ProspectiveCohortError(
            f"Finalized cohort artifacts already exist: {existing}"
        )
    for name, raw in targets.items():
        _write_exclusive(output_dir / name, raw)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--census-run-dir", type=Path, default=DEFAULT_CENSUS_RUN_DIR)
    parser.add_argument("--outcome-run-dir", type=Path, default=DEFAULT_OUTCOME_RUN_DIR)
    parser.add_argument(
        "--commitment-ledger", type=Path, default=DEFAULT_COMMITMENT_LEDGER
    )
    parser.add_argument(
        "--observation-ledger", type=Path, default=DEFAULT_OBSERVATION_LEDGER
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = finalize_cohort(
            args.protocol,
            args.census_run_dir,
            args.outcome_run_dir,
            args.commitment_ledger,
            args.observation_ledger,
            args.output_dir,
        )
        print(json.dumps(manifest, sort_keys=True))
    except (
        ProspectiveCohortError,
        ProspectiveLedgerError,
        ProspectiveSnapshotError,
        OSError,
        ValueError,
        KeyError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
