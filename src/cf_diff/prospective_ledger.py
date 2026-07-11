"""Append-only prospective commitment and observation event chains."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Final, Mapping, Sequence

import pandas as pd

from cf_diff.prospective_model import (
    PREDICTION_COLUMNS as MODEL_PREDICTION_COLUMNS,
)
from cf_diff.prospective_model import (
    ProspectiveModelError,
    load_frozen_protocol,
    predict_prospective,
    verify_frozen_model,
)


ZERO_HASH: Final[str] = "0" * 64
COMMITMENT_CHAIN_ID: Final[str] = (
    "cf-difficulty-prospective-v2/commitments"
)
OBSERVATION_CHAIN_ID: Final[str] = (
    "cf-difficulty-prospective-v2/observations"
)
DEFAULT_PROTOCOL_PATH: Final[Path] = Path(
    "configs/prospective_protocol_v2.json"
)
DEFAULT_MANIFEST_PATH: Final[Path] = Path(
    "prospective/model_freeze_manifest_v2.json"
)
DEFAULT_COMMITMENT_LEDGER: Final[Path] = Path(
    "prospective/ledger/commitments.jsonl"
)
DEFAULT_OBSERVATION_LEDGER: Final[Path] = Path(
    "prospective/ledger/observations.jsonl"
)
DEFAULT_WITNESS_EVIDENCE_DIR: Final[Path] = Path(
    "prospective/witnesses"
)
DEFAULT_MODEL_PATH: Final[Path] = Path("prospective/model_bundle_v2.json")

IMMUTABLE_EVIDENCE_PREFIXES: Final[tuple[str, ...]] = (
    "prospective/inputs/",
    "prospective/predictions/",
    "prospective/witnesses/",
    "prospective/snapshots/",
    "prospective/cohort/",
)
FROZEN_CONTROL_PATHS: Final[frozenset[str]] = frozenset(
    {
        "configs/prospective_protocol_v2.json",
        ".gitattributes",
        "requirements.txt",
        ".github/workflows/prospective-witness.yml",
        ".github/workflows/tests.yml",
        "src/cf_diff/prospective_input.py",
        "src/cf_diff/prospective_model.py",
        "src/cf_diff/prospective_ledger.py",
        "src/cf_diff/prospective_snapshot.py",
        "src/cf_diff/prospective_cohort.py",
        "src/cf_diff/prospective_analysis.py",
        "src/cf_diff/statement_features.py",
        "tests/test_prospective_protocol.py",
        "tests/test_prospective_input.py",
        "tests/test_prospective_model.py",
        "tests/test_prospective_ledger.py",
        "tests/test_prospective_snapshot.py",
        "tests/test_prospective_cohort.py",
        "tests/test_prospective_analysis.py",
    }
)
FREEZE_ARTIFACT_PATHS: Final[frozenset[str]] = frozenset(
    {
        "prospective/model_bundle_v2.json",
        "prospective/model_freeze_manifest_v2.json",
    }
)

EXTERNAL_TIMESTAMP_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "provider",
        "repository_full_name",
        "repository_id",
        "default_branch",
        "workflow_name",
        "workflow_path",
        "workflow_file_sha256",
        "trigger_event",
        "deadline_field",
        "required_status",
        "required_conclusion",
        "required_run_attempt",
        "target_commit_must_be_default_branch_ancestor",
        "commit_message_skip_directives_prohibited",
        "local_git_timestamps_are_external_evidence",
        "witness_record_policy",
    }
)

COMMITMENT_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "prediction_commitment",
        "operational_miss",
        "github_actions_witness",
        "prediction_invalidated",
    }
)
OBSERVATION_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "audit_outcome_snapshot",
        "cohort_census_snapshot",
        "confirmatory_outcome_snapshot",
    }
)
MISS_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "capture_failed",
        "prediction_failed",
        "publication_failed",
        "operator_unavailable",
        "external_witness_missing",
        "external_witness_late",
        "official_start_mismatch",
        "official_index_set_mismatch",
        "artifact_integrity_failure",
        "snapshot_window_failed",
    }
)
INVALIDATION_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "external_witness_missing",
        "external_witness_late",
        "external_witness_invalid",
        "official_start_mismatch",
        "official_index_set_mismatch",
        "artifact_integrity_failure",
    }
)

COMMON_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "chain_id",
        "sequence",
        "event_type",
        "protocol_id",
        "protocol_sha256",
        "local_recorded_at_utc",
        "previous_event_sha256",
        "event_sha256",
    }
)
EVENT_FIELDS: Final[dict[str, frozenset[str]]] = {
    "prediction_commitment": COMMON_FIELDS
    | frozenset(
        {
            "contest_id",
            "operator_contest_start_utc",
            "lock_deadline_utc",
            "prediction_created_at_utc",
            "indices",
            "row_count",
            "input_path",
            "input_sha256",
            "capture_sidecar_path",
            "capture_sidecar_sha256",
            "prediction_path",
            "prediction_sha256",
            "model_artifact_path",
            "freeze_manifest_path",
            "model_bundle_id",
            "model_artifact_sha256",
            "freeze_manifest_sha256",
        }
    ),
    "operational_miss": COMMON_FIELDS
    | frozenset(
        {
            "contest_id",
            "operator_contest_start_utc",
            "lock_deadline_utc",
            "miss_stage",
            "reason_code",
            "reason_detail",
            "evidence_path",
            "evidence_sha256",
        }
    ),
    "github_actions_witness": COMMON_FIELDS
    | frozenset(
        {
            "contest_id",
            "target_event_sha256",
            "target_commit_sha",
            "run_api_response_path",
            "run_api_response_sha256",
            "repository_id",
            "repository_full_name",
            "head_sha",
            "head_branch",
            "trigger_event",
            "workflow_id",
            "workflow_path",
            "workflow_file_sha256",
            "run_id",
            "run_attempt",
            "external_timestamp",
            "status",
            "conclusion",
            "run_url",
            "timely",
        }
    ),
    "prediction_invalidated": COMMON_FIELDS
    | frozenset(
        {
            "contest_id",
            "target_event_sha256",
            "reason_code",
            "reason_detail",
            "evidence_path",
            "evidence_sha256",
        }
    ),
    "audit_outcome_snapshot": COMMON_FIELDS
    | frozenset(
        {
            "contest_id",
            "selection_manifest_path",
            "selection_manifest_sha256",
            "raw_snapshot_path",
            "raw_snapshot_sha256",
        }
    ),
    "cohort_census_snapshot": COMMON_FIELDS
    | frozenset(
        {
            "selection_manifest_path",
            "selection_manifest_sha256",
            "raw_snapshot_path",
            "raw_snapshot_sha256",
        }
    ),
    "confirmatory_outcome_snapshot": COMMON_FIELDS
    | frozenset(
        {
            "selection_manifest_path",
            "selection_manifest_sha256",
            "raw_snapshot_path",
            "raw_snapshot_sha256",
        }
    ),
}


class ProspectiveLedgerError(RuntimeError):
    """Raised when an event-chain operation is unsafe or inconsistent."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one JSON value canonically for event hashing."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProspectiveLedgerError(
            f"Value is not canonical JSON: {error}"
        ) from error


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ProspectiveLedgerError(f"Required file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ProspectiveLedgerError(
            f"{field} must be a lowercase SHA-256 digest."
        )
    return value


def _require_commit_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ProspectiveLedgerError(
            f"{field} must be a complete lowercase Git commit SHA."
        )
    return value


def _parse_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ProspectiveLedgerError(f"{field} must be a UTC timestamp.")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ProspectiveLedgerError(
            f"{field} is not an ISO-8601 timestamp."
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProspectiveLedgerError(f"{field} must include a UTC offset.")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ProspectiveLedgerError("Timestamp must include a UTC offset.")
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _normalize_contest_id(value: object) -> str:
    if isinstance(value, bool):
        raise ProspectiveLedgerError("contest_id must be a positive integer.")
    text = str(value).strip()
    if not re.fullmatch(r"[0-9]+", text) or int(text) < 1:
        raise ProspectiveLedgerError("contest_id must be a positive integer.")
    return str(int(text))


def _normalize_index(value: object) -> str:
    text = str(value).strip().upper()
    if not re.fullmatch(r"[A-Z0-9]+", text):
        raise ProspectiveLedgerError(
            f"Invalid Codeforces problem index: {value!r}"
        )
    return text


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise ProspectiveLedgerError(f"{label} does not exist: {path}") from error
    except OSError as error:
        raise ProspectiveLedgerError(f"Cannot read {label}: {error}") from error
    return _decode_json_object_bytes(raw, label)


def _decode_json_object_bytes(raw: bytes, label: str) -> dict[str, object]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProspectiveLedgerError(f"Cannot read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ProspectiveLedgerError(f"{label} must be a JSON object.")
    return payload


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProspectiveLedgerError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ProspectiveLedgerError(f"Non-finite JSON constant is forbidden: {value}")


def _require_positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProspectiveLedgerError(f"{field} must be a positive integer.")
    return value


def _external_timestamp_policy(
    protocol: Mapping[str, object],
) -> dict[str, object]:
    value = protocol.get("external_timestamp")
    if not isinstance(value, dict):
        raise ProspectiveLedgerError(
            "Frozen protocol lacks structured external_timestamp settings."
        )
    present = frozenset(value)
    if present != EXTERNAL_TIMESTAMP_FIELDS:
        raise ProspectiveLedgerError(
            "protocol.external_timestamp must have exact fields; "
            f"missing={sorted(EXTERNAL_TIMESTAMP_FIELDS - present)}, "
            f"extra={sorted(present - EXTERNAL_TIMESTAMP_FIELDS)}."
        )
    required_literals = {
        "provider": "github_actions_workflow_run",
        "trigger_event": "push",
        "deadline_field": "created_at",
        "required_status": "completed",
        "required_conclusion": "success",
        "required_run_attempt": 1,
        "target_commit_must_be_default_branch_ancestor": True,
        "commit_message_skip_directives_prohibited": True,
        "local_git_timestamps_are_external_evidence": False,
    }
    for field, expected in required_literals.items():
        if value.get(field) != expected:
            raise ProspectiveLedgerError(
                f"protocol.external_timestamp.{field} must equal {expected!r}."
            )
    _require_positive_integer(
        value.get("repository_id"),
        "protocol.external_timestamp.repository_id",
    )
    repository = value.get("repository_full_name")
    if not isinstance(repository, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository
    ):
        raise ProspectiveLedgerError(
            "protocol.external_timestamp.repository_full_name is invalid."
        )
    branch = value.get("default_branch")
    if not isinstance(branch, str) or not re.fullmatch(
        r"[A-Za-z0-9._/-]+", branch
    ):
        raise ProspectiveLedgerError(
            "protocol.external_timestamp.default_branch is invalid."
        )
    workflow_path = value.get("workflow_path")
    if (
        not isinstance(workflow_path, str)
        or not re.fullmatch(r"\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml", workflow_path)
    ):
        raise ProspectiveLedgerError(
            "protocol.external_timestamp.workflow_path is invalid."
        )
    _require_sha256(
        value.get("workflow_file_sha256"),
        "protocol.external_timestamp.workflow_file_sha256",
    )
    workflow_name = value.get("workflow_name")
    if not isinstance(workflow_name, str) or not workflow_name.strip():
        raise ProspectiveLedgerError(
            "protocol.external_timestamp.workflow_name is invalid."
        )
    witness_policy = value.get("witness_record_policy")
    if not isinstance(witness_policy, str) or not witness_policy.strip():
        raise ProspectiveLedgerError(
            "protocol.external_timestamp.witness_record_policy is invalid."
        )
    return dict(value)


def _load_protocol(path: Path) -> tuple[dict[str, object], str]:
    try:
        protocol = load_frozen_protocol(path)
    except ProspectiveModelError as error:
        raise ProspectiveLedgerError(str(error)) from error
    protocol_id = protocol.get("protocol_id")
    if (
        not isinstance(protocol_id, str)
        or f"{protocol_id}/commitments" != COMMITMENT_CHAIN_ID
        or f"{protocol_id}/observations" != OBSERVATION_CHAIN_ID
    ):
        raise ProspectiveLedgerError(
            "Frozen protocol_id does not match the v2 ledger chain identifiers."
        )
    _external_timestamp_policy(protocol)
    return protocol, sha256_file(path)


def _chain_id_for_types(allowed_event_types: frozenset[str]) -> str:
    if allowed_event_types == COMMITMENT_EVENT_TYPES:
        return COMMITMENT_CHAIN_ID
    if allowed_event_types == OBSERVATION_EVENT_TYPES:
        return OBSERVATION_CHAIN_ID
    raise ProspectiveLedgerError("Unknown ledger event-type set.")


def _validate_event_schema(
    event: Mapping[str, object],
    *,
    line_number: int,
    allowed_event_types: frozenset[str],
) -> None:
    event_type = event.get("event_type")
    if event_type not in allowed_event_types:
        raise ProspectiveLedgerError(
            f"Ledger line {line_number} has invalid event_type {event_type!r}."
        )
    expected = EVENT_FIELDS[str(event_type)]
    present = frozenset(event)
    if present != expected:
        raise ProspectiveLedgerError(
            f"Ledger line {line_number} has non-exact fields; "
            f"missing={sorted(expected - present)}, "
            f"extra={sorted(present - expected)}."
        )
    if event.get("schema_version") != 1:
        raise ProspectiveLedgerError(
            f"Ledger line {line_number} has an invalid schema_version."
        )
    if event.get("chain_id") != _chain_id_for_types(allowed_event_types):
        raise ProspectiveLedgerError(
            f"Ledger line {line_number} has the wrong chain_id."
        )
    _parse_utc(event.get("local_recorded_at_utc"), "local_recorded_at_utc")
    _require_sha256(event.get("protocol_sha256"), "protocol_sha256")


def _require_canonical_utc(value: object, field: str) -> datetime:
    parsed = _parse_utc(value, field)
    if value != _format_utc(parsed):
        raise ProspectiveLedgerError(f"{field} must use canonical UTC Z format.")
    return parsed


def _require_repository_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProspectiveLedgerError(f"{field} must be a repository-relative path.")
    if "\\" in value or _repository_relative(Path(value)) != value:
        raise ProspectiveLedgerError(f"{field} must be a canonical POSIX path.")
    return value


def _validate_evidence_pair(
    event: Mapping[str, object],
    path_field: str,
    hash_field: str,
) -> None:
    path = event.get(path_field)
    digest = event.get(hash_field)
    if path is None and digest is None:
        return
    _require_repository_path(path, path_field)
    _require_sha256(digest, hash_field)


def _validate_event_semantics(
    event: Mapping[str, object],
    *,
    line_number: int,
) -> None:
    sequence = _require_positive_integer(event.get("sequence"), "sequence")
    if sequence != line_number:
        raise ProspectiveLedgerError(
            f"Ledger line {line_number} has a non-contiguous sequence."
        )
    protocol_id = event.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ProspectiveLedgerError("protocol_id must be a non-empty string.")
    expected_chain = f"{protocol_id}/" + (
        "commitments"
        if event.get("chain_id") == COMMITMENT_CHAIN_ID
        else "observations"
    )
    if event.get("chain_id") != expected_chain:
        raise ProspectiveLedgerError(
            "Event protocol_id and chain_id are inconsistent."
        )
    _require_canonical_utc(event.get("local_recorded_at_utc"), "local_recorded_at_utc")
    _require_sha256(event.get("previous_event_sha256"), "previous_event_sha256")

    event_type = str(event["event_type"])
    if "contest_id" in event:
        if event.get("contest_id") != _normalize_contest_id(event.get("contest_id")):
            raise ProspectiveLedgerError("contest_id is not canonical.")

    if event_type == "prediction_commitment":
        start = _require_canonical_utc(
            event.get("operator_contest_start_utc"),
            "operator_contest_start_utc",
        )
        deadline = _require_canonical_utc(
            event.get("lock_deadline_utc"),
            "lock_deadline_utc",
        )
        created = _require_canonical_utc(
            event.get("prediction_created_at_utc"),
            "prediction_created_at_utc",
        )
        if not start <= created <= deadline:
            raise ProspectiveLedgerError(
                "Prediction event timestamps are outside the T0 window."
            )
        indices = event.get("indices")
        if not isinstance(indices, list) or not indices:
            raise ProspectiveLedgerError("Prediction indices must be a non-empty list.")
        normalized = [_normalize_index(value) for value in indices]
        if normalized != indices or len(normalized) != len(set(normalized)):
            raise ProspectiveLedgerError(
                "Prediction indices must be canonical and unique."
            )
        row_count = _require_positive_integer(event.get("row_count"), "row_count")
        if row_count != len(indices):
            raise ProspectiveLedgerError("Prediction row_count does not match indices.")
        for path_field, hash_field in (
            ("input_path", "input_sha256"),
            ("capture_sidecar_path", "capture_sidecar_sha256"),
            ("prediction_path", "prediction_sha256"),
            ("model_artifact_path", "model_artifact_sha256"),
            ("freeze_manifest_path", "freeze_manifest_sha256"),
        ):
            _validate_evidence_pair(event, path_field, hash_field)
        if not isinstance(event.get("model_bundle_id"), str) or not event.get(
            "model_bundle_id"
        ):
            raise ProspectiveLedgerError("model_bundle_id must be a non-empty string.")
    elif event_type == "operational_miss":
        start = _require_canonical_utc(
            event.get("operator_contest_start_utc"),
            "operator_contest_start_utc",
        )
        deadline = _require_canonical_utc(
            event.get("lock_deadline_utc"), "lock_deadline_utc"
        )
        if deadline <= start:
            raise ProspectiveLedgerError(
                "Operational miss deadline must follow contest start."
            )
        stage = event.get("miss_stage")
        if not isinstance(stage, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]{1,63}", stage
        ):
            raise ProspectiveLedgerError("miss_stage is invalid.")
        if event.get("reason_code") not in MISS_REASON_CODES:
            raise ProspectiveLedgerError("Operational miss reason_code is invalid.")
        if not isinstance(event.get("reason_detail"), str) or not str(
            event["reason_detail"]
        ).strip():
            raise ProspectiveLedgerError("Operational miss reason_detail is empty.")
        _validate_evidence_pair(event, "evidence_path", "evidence_sha256")
    elif event_type == "prediction_invalidated":
        _require_sha256(event.get("target_event_sha256"), "target_event_sha256")
        if event.get("reason_code") not in INVALIDATION_REASON_CODES:
            raise ProspectiveLedgerError("Prediction invalidation reason_code is invalid.")
        if not isinstance(event.get("reason_detail"), str) or not str(
            event["reason_detail"]
        ).strip():
            raise ProspectiveLedgerError("Prediction invalidation detail is empty.")
        _validate_evidence_pair(event, "evidence_path", "evidence_sha256")
    elif event_type == "github_actions_witness":
        _require_sha256(event.get("target_event_sha256"), "target_event_sha256")
        target_commit = _require_commit_sha(
            event.get("target_commit_sha"), "target_commit_sha"
        )
        if event.get("head_sha") != target_commit:
            raise ProspectiveLedgerError("Witness target_commit_sha and head_sha differ.")
        _validate_evidence_pair(
            event, "run_api_response_path", "run_api_response_sha256"
        )
        for field in ("repository_id", "workflow_id", "run_id", "run_attempt"):
            _require_positive_integer(event.get(field), field)
        if event.get("run_attempt") != 1:
            raise ProspectiveLedgerError("Witness run_attempt must equal 1.")
        repository = event.get("repository_full_name")
        if not isinstance(repository, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository
        ):
            raise ProspectiveLedgerError("Witness repository_full_name is invalid.")
        if event.get("trigger_event") != "push":
            raise ProspectiveLedgerError("Witness trigger_event must be push.")
        _require_repository_path(event.get("workflow_path"), "workflow_path")
        _require_sha256(
            event.get("workflow_file_sha256"), "workflow_file_sha256"
        )
        external = event.get("external_timestamp")
        if not isinstance(external, dict) or frozenset(external) != {
            "provider",
            "field",
            "value_utc",
        }:
            raise ProspectiveLedgerError("Witness external_timestamp is malformed.")
        if (
            external.get("provider") != "github_actions_workflow_run"
            or external.get("field") != "created_at"
        ):
            raise ProspectiveLedgerError("Witness external timestamp policy is invalid.")
        _require_canonical_utc(
            external.get("value_utc"), "external_timestamp.value_utc"
        )
        if event.get("status") != "completed" or event.get("conclusion") != "success":
            raise ProspectiveLedgerError("Witness run was not completed successfully.")
        if not isinstance(event.get("timely"), bool):
            raise ProspectiveLedgerError("Witness timely must be boolean.")
        expected_url = (
            f"https://github.com/{repository}/actions/runs/{event['run_id']}"
        )
        if event.get("run_url") != expected_url:
            raise ProspectiveLedgerError("Witness run_url is not canonical.")
    elif event_type in OBSERVATION_EVENT_TYPES:
        for path_field, hash_field in (
            ("selection_manifest_path", "selection_manifest_sha256"),
            ("raw_snapshot_path", "raw_snapshot_sha256"),
        ):
            _validate_evidence_pair(event, path_field, hash_field)


def _validate_chain_invariants(
    events: Sequence[Mapping[str, object]],
    *,
    allowed_event_types: frozenset[str],
) -> None:
    if not events:
        return
    protocol_ids = {str(event.get("protocol_id")) for event in events}
    protocol_hashes = {str(event.get("protocol_sha256")) for event in events}
    if len(protocol_ids) != 1 or len(protocol_hashes) != 1:
        raise ProspectiveLedgerError(
            "One event chain cannot mix protocol identifiers or hashes."
        )
    protocol_id = next(iter(protocol_ids))
    if _chain_id_for_types(allowed_event_types) != (
        f"{protocol_id}/commitments"
        if allowed_event_types == COMMITMENT_EVENT_TYPES
        else f"{protocol_id}/observations"
    ):
        raise ProspectiveLedgerError(
            "Event chain identity is inconsistent with protocol_id."
        )


def _event_sha256(event_without_hash: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(event_without_hash)).hexdigest()


def _verify_ledger_bytes(
    raw: bytes,
    *,
    label: str,
    allowed_event_types: frozenset[str],
) -> list[dict[str, object]]:
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        raise ProspectiveLedgerError(f"{label} has a partial final line.")
    events: list[dict[str, object]] = []
    previous_hash = ZERO_HASH
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        try:
            event = json.loads(
                raw_line.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
            )
        except (
            UnicodeError,
            json.JSONDecodeError,
            ProspectiveLedgerError,
        ) as error:
            raise ProspectiveLedgerError(
                f"{label} line {line_number} is invalid JSON: {error}"
            ) from error
        if not isinstance(event, dict):
            raise ProspectiveLedgerError(
                f"{label} line {line_number} must be an object."
            )
        _validate_event_schema(
            event,
            line_number=line_number,
            allowed_event_types=allowed_event_types,
        )
        _validate_event_semantics(event, line_number=line_number)
        if event.get("previous_event_sha256") != previous_hash:
            raise ProspectiveLedgerError(
                f"{label} line {line_number} breaks the hash chain."
            )
        recorded_hash = _require_sha256(
            event.get("event_sha256"),
            f"{label} line {line_number} event_sha256",
        )
        unhashed = dict(event)
        del unhashed["event_sha256"]
        if recorded_hash != _event_sha256(unhashed):
            raise ProspectiveLedgerError(
                f"{label} line {line_number} has an invalid event hash."
            )
        previous_hash = recorded_hash
        events.append(event)
    _validate_chain_invariants(
        events,
        allowed_event_types=allowed_event_types,
    )
    return events


def verify_ledger(
    path: Path,
    *,
    allowed_event_types: frozenset[str],
) -> list[dict[str, object]]:
    raw = path.read_bytes() if path.exists() else b""
    return _verify_ledger_bytes(
        raw,
        label=path.as_posix(),
        allowed_event_types=allowed_event_types,
    )


def _append_payloads(
    path: Path,
    payloads: list[dict[str, object]],
    *,
    protocol: Mapping[str, object],
    protocol_sha256: str,
    allowed_event_types: frozenset[str],
) -> list[dict[str, object]]:
    if not payloads:
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    temporary_path = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with lock_path.open("x", encoding="utf-8") as lock:
            lock.write(str(os.getpid()))
    except FileExistsError as error:
        raise ProspectiveLedgerError(
            f"Ledger lock already exists: {lock_path}"
        ) from error

    try:
        original = path.read_bytes() if path.exists() else b""
        existing = _verify_ledger_bytes(
            original,
            label=path.as_posix(),
            allowed_event_types=allowed_event_types,
        )
        previous_hash = (
            str(existing[-1]["event_sha256"]) if existing else ZERO_HASH
        )
        sequence = len(existing)
        recorded_at = _format_utc(_utc_now())
        chain_id = _chain_id_for_types(allowed_event_types)
        events: list[dict[str, object]] = []
        for payload in payloads:
            sequence += 1
            event = {
                "schema_version": 1,
                "chain_id": chain_id,
                **payload,
                "sequence": sequence,
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_sha256,
                "local_recorded_at_utc": recorded_at,
                "previous_event_sha256": previous_hash,
            }
            event["event_sha256"] = _event_sha256(event)
            _validate_event_schema(
                event,
                line_number=sequence,
                allowed_event_types=allowed_event_types,
            )
            _validate_event_semantics(event, line_number=sequence)
            previous_hash = str(event["event_sha256"])
            events.append(event)
        appended = b"".join(
            canonical_json_bytes(event) + b"\n" for event in events
        )
        with temporary_path.open("xb") as handle:
            handle.write(original)
            handle.write(appended)
            handle.flush()
            os.fsync(handle.fileno())
        current = path.read_bytes() if path.exists() else b""
        if current != original:
            raise ProspectiveLedgerError(
                "Ledger changed concurrently while an append was prepared."
            )
        os.replace(temporary_path, path)
        verify_ledger(path, allowed_event_types=allowed_event_types)
        return events
    finally:
        temporary_path.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


def assert_append_only_bytes(
    previous: bytes,
    current: bytes,
    *,
    label: str,
) -> None:
    if not current.startswith(previous):
        raise ProspectiveLedgerError(
            f"{label} is not append-only: previously committed bytes changed."
        )


def _repository_relative(path: Path) -> str:
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(Path.cwd().resolve())
        except ValueError as error:
            raise ProspectiveLedgerError(
                f"Evidence path must be inside the repository: {path}"
            ) from error
    if ".." in path.parts:
        raise ProspectiveLedgerError(
            f"Evidence path cannot traverse parent directories: {path}"
        )
    return path.as_posix()


def check_git_append_only(
    base_ref: str,
    ledger_paths: Sequence[Path],
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> dict[str, str]:
    check = subprocess.run(
        ["git", "rev-parse", "--verify", f"{base_ref}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if check.returncode != 0:
        raise ProspectiveLedgerError(f"Git base ref does not exist: {base_ref}")
    result: dict[str, str] = {}
    protocol_repository_path = _repository_relative(protocol_path)
    base_protocol_result = subprocess.run(
        ["git", "show", f"{base_ref}:{protocol_repository_path}"],
        capture_output=True,
        check=False,
    )
    base_frozen = False
    if base_protocol_result.returncode == 0:
        try:
            base_protocol = _decode_json_object_bytes(
                base_protocol_result.stdout,
                f"{base_ref}:{protocol_repository_path}",
            )
        except ProspectiveLedgerError:
            base_protocol = {}
        base_frozen = base_protocol.get("status") == "frozen"

    diff = subprocess.run(
        ["git", "diff", "--name-status", "--find-renames", base_ref, "--"],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
    )
    if diff.returncode != 0:
        raise ProspectiveLedgerError("Cannot inspect changed Git paths.")
    ledger_names = {_repository_relative(path) for path in ledger_paths}
    for raw_line in diff.stdout.splitlines():
        fields = raw_line.split("\t")
        if len(fields) < 2:
            raise ProspectiveLedgerError(
                f"Cannot parse Git name-status line: {raw_line!r}"
            )
        status = fields[0]
        paths = fields[1:]
        for changed_path in paths:
            if changed_path in ledger_names:
                if status[0] not in {"A", "M"}:
                    raise ProspectiveLedgerError(
                        f"Ledger path cannot be deleted or renamed: {changed_path}"
                    )
                continue
            if any(
                changed_path.startswith(prefix)
                for prefix in IMMUTABLE_EVIDENCE_PREFIXES
            ) and status[0] != "A":
                raise ProspectiveLedgerError(
                    "Previously published prospective evidence is immutable: "
                    f"{status} {changed_path}"
                )
            if changed_path in FREEZE_ARTIFACT_PATHS:
                existed = subprocess.run(
                    ["git", "cat-file", "-e", f"{base_ref}:{changed_path}"],
                    capture_output=True,
                    check=False,
                ).returncode == 0
                if existed:
                    raise ProspectiveLedgerError(
                        f"Frozen artifact cannot change in place: {changed_path}"
                    )
            if base_frozen and changed_path in FROZEN_CONTROL_PATHS:
                raise ProspectiveLedgerError(
                    f"Frozen prospective control cannot change in place: {changed_path}"
                )

    for path in ledger_paths:
        repository_path = _repository_relative(path)
        previous = subprocess.run(
            ["git", "show", f"{base_ref}:{repository_path}"],
            capture_output=True,
            check=False,
        )
        if previous.returncode != 0:
            result[repository_path] = "new_or_absent"
            continue
        if not path.is_file():
            raise ProspectiveLedgerError(
                f"Previously committed ledger was deleted: {repository_path}"
            )
        assert_append_only_bytes(
            previous.stdout,
            path.read_bytes(),
            label=repository_path,
        )
        result[repository_path] = "append_only"
    tracked = subprocess.run(
        ["git", "ls-files", "--stage"],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
    )
    if tracked.returncode != 0:
        raise ProspectiveLedgerError("Cannot inspect tracked Git file modes.")
    protected_paths: list[str] = []
    for line in tracked.stdout.splitlines():
        try:
            metadata, path = line.split("\t", 1)
            mode = metadata.split(" ", 1)[0]
        except ValueError as error:
            raise ProspectiveLedgerError(
                f"Cannot parse tracked Git entry: {line!r}"
            ) from error
        if (
            path.startswith("prospective/")
            or path in FROZEN_CONTROL_PATHS
            or path in FREEZE_ARTIFACT_PATHS
        ):
            protected_paths.append(path)
            if mode == "120000":
                raise ProspectiveLedgerError(
                    f"Prospective evidence cannot be a symbolic link: {path}"
                )
    folded: dict[str, str] = {}
    for path in protected_paths:
        key = path.casefold()
        previous_path = folded.get(key)
        if previous_path is not None and previous_path != path:
            raise ProspectiveLedgerError(
                "Case-colliding prospective paths are forbidden: "
                f"{previous_path}, {path}"
            )
        folded[key] = path
    result["immutable_evidence"] = "verified"
    return result


def _read_prediction_csv(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".csv":
        raise ProspectiveLedgerError("Committed prediction must be a CSV file.")
    frame = pd.read_csv(
        path,
        dtype={"contest_id": "string", "index": "string"},
        keep_default_na=False,
    )
    if list(frame.columns) != list(MODEL_PREDICTION_COLUMNS):
        raise ProspectiveLedgerError(
            "Prediction file columns do not match the frozen v2 schema."
        )
    if frame.empty:
        raise ProspectiveLedgerError("Prediction file is empty.")
    frame["contest_id"] = frame["contest_id"].map(_normalize_contest_id)
    frame["index"] = frame["index"].map(_normalize_index)
    if frame.duplicated(["contest_id", "index"]).any():
        raise ProspectiveLedgerError("Prediction file has duplicate problem keys.")
    if frame["contest_id"].nunique() != 1:
        raise ProspectiveLedgerError(
            "One prediction commitment must contain exactly one contest."
        )
    for column in ("primary_prediction", "comparator_prediction"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not all(math.isfinite(float(value)) for value in numeric):
            raise ProspectiveLedgerError(
                f"Prediction column {column} must contain finite numbers."
            )
    for column in (
        "feature_row_sha256",
        "model_artifact_sha256",
        "freeze_manifest_sha256",
        "input_file_sha256",
        "capture_sidecar_sha256",
    ):
        if not frame[column].astype(str).map(
            lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", value))
        ).all():
            raise ProspectiveLedgerError(
                f"Prediction column {column} contains an invalid SHA-256 digest."
            )
    return frame


def _one_string(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].astype(str).drop_duplicates().tolist()
    if len(values) != 1 or not values[0]:
        raise ProspectiveLedgerError(
            f"Prediction column {column} must have one constant value."
        )
    return values[0]


def _existing_base_events(
    events: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for event in events:
        if event.get("event_type") not in {
            "prediction_commitment",
            "operational_miss",
        }:
            continue
        contest_id = str(event.get("contest_id"))
        if contest_id in result:
            raise ProspectiveLedgerError(
                f"Contest {contest_id} has multiple base coverage events."
            )
        result[contest_id] = event
    return result


def _validate_capture_binding(
    *,
    protocol: Mapping[str, object],
    protocol_sha256: str,
    contest_id: str,
    contest_start: datetime,
    deadline: datetime,
    indices: list[str],
    input_path: Path,
    input_sha256: str,
    capture_sidecar_path: Path,
    capture_sidecar_sha256: str,
) -> None:
    if sha256_file(input_path) != input_sha256:
        raise ProspectiveLedgerError("T0 input SHA-256 does not match prediction rows.")
    if sha256_file(capture_sidecar_path) != capture_sidecar_sha256:
        raise ProspectiveLedgerError(
            "Capture sidecar SHA-256 does not match prediction rows."
        )
    sidecar = _read_json_object(capture_sidecar_path, "capture sidecar")
    if (
        sidecar.get("schema_version") != 1
        or sidecar.get("status") != "complete"
        or sidecar.get("protocol_id") != protocol.get("protocol_id")
        or sidecar.get("protocol_sha256") != protocol_sha256
        or _normalize_contest_id(sidecar.get("contest_id")) != contest_id
        or _require_canonical_utc(
            sidecar.get("contest_start_utc"),
            "capture_sidecar.contest_start_utc",
        )
        != contest_start
        or _require_canonical_utc(
            sidecar.get("lock_deadline_utc"),
            "capture_sidecar.lock_deadline_utc",
        )
        != deadline
    ):
        raise ProspectiveLedgerError(
            "Capture sidecar does not match the frozen protocol or contest."
        )
    requested = sidecar.get("requested_indices")
    if not isinstance(requested, list) or [
        _normalize_index(value) for value in requested
    ] != indices:
        raise ProspectiveLedgerError(
            "Capture sidecar indices do not match prediction rows."
        )
    output = sidecar.get("output")
    if not isinstance(output, dict):
        raise ProspectiveLedgerError("Capture sidecar lacks completed output metadata.")
    output_path = output.get("path")
    if (
        not isinstance(output_path, str)
        or _repository_relative(Path(output_path)) != _repository_relative(input_path)
        or output.get("sha256") != input_sha256
        or output.get("row_count") != len(indices)
    ):
        raise ProspectiveLedgerError(
            "Capture sidecar output metadata does not match the explicit T0 input."
        )


def _verify_prediction_recomputation(
    *,
    prediction_path: Path,
    input_path: Path,
    capture_sidecar_path: Path,
    model_path: Path,
    manifest_path: Path,
    protocol_path: Path,
    contest_start_utc: str,
    prediction_created_at: datetime,
) -> None:
    temporary = prediction_path.with_name(
        f".{prediction_path.stem}.verification-{uuid.uuid4().hex}.csv"
    )
    try:
        predict_prospective(
            protocol_path=protocol_path,
            model_path=model_path,
            manifest_path=manifest_path,
            input_path=input_path,
            capture_sidecar_path=capture_sidecar_path,
            output_path=temporary,
            contest_start_utc=contest_start_utc,
            _clock=lambda: prediction_created_at,
        )
        if temporary.read_bytes() != prediction_path.read_bytes():
            raise ProspectiveLedgerError(
                "Prediction CSV does not exactly match a frozen-model recomputation."
            )
    except ProspectiveModelError as error:
        raise ProspectiveLedgerError(
            f"Prediction recomputation failed: {error}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def record_prediction_commitment(
    prediction_path: Path,
    *,
    input_path: Path,
    capture_sidecar_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    protocol_path: Path,
    manifest_path: Path,
    ledger_path: Path = DEFAULT_COMMITMENT_LEDGER,
) -> dict[str, object]:
    protocol, protocol_hash = _load_protocol(protocol_path)
    try:
        verify_frozen_model(
            protocol_path=protocol_path,
            model_path=model_path,
            manifest_path=manifest_path,
        )
    except ProspectiveModelError as error:
        raise ProspectiveLedgerError(
            f"Frozen model verification failed: {error}"
        ) from error
    manifest = _read_json_object(manifest_path, "model freeze manifest")
    frame = _read_prediction_csv(prediction_path)
    contest_id = _one_string(frame, "contest_id")
    contest_start_text = _one_string(frame, "contest_start_utc")
    prediction_created_text = _one_string(
        frame,
        "prediction_created_at_utc",
    )
    contest_start = _parse_utc(
        contest_start_text,
        "prediction contest_start_utc",
    )
    prediction_created = _parse_utc(
        prediction_created_text,
        "prediction prediction_created_at_utc",
    )
    cohort = protocol.get("cohort")
    timepoint = protocol.get("prediction_timepoint")
    if not isinstance(cohort, dict) or not isinstance(timepoint, dict):
        raise ProspectiveLedgerError("Protocol lacks cohort or T0 settings.")
    cohort_start = _parse_utc(
        cohort.get("eligibility_start_utc"),
        "cohort.eligibility_start_utc",
    )
    cohort_end = _parse_utc(
        cohort.get("eligibility_end_utc"),
        "cohort.eligibility_end_utc",
    )
    if not cohort_start <= contest_start <= cohort_end:
        raise ProspectiveLedgerError(
            "Prediction contest is outside the frozen cohort."
        )
    minutes = timepoint.get("lock_deadline_minutes_after_contest_start")
    if isinstance(minutes, bool) or not isinstance(minutes, int) or minutes < 1:
        raise ProspectiveLedgerError("Protocol has an invalid T0 deadline.")
    deadline = contest_start + timedelta(minutes=minutes)
    recorded_at = _utc_now()
    if not contest_start <= prediction_created <= recorded_at <= deadline:
        raise ProspectiveLedgerError(
            "Prediction commitment was not recorded inside the T0 window."
        )

    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ProspectiveLedgerError("Manifest protocol_id mismatch.")
    if manifest.get("protocol_sha256") != protocol_hash:
        raise ProspectiveLedgerError("Manifest protocol SHA-256 mismatch.")
    manifest_hash = sha256_file(manifest_path)
    if _one_string(frame, "freeze_manifest_sha256") != manifest_hash:
        raise ProspectiveLedgerError("Prediction freeze manifest hash mismatch.")
    if _one_string(frame, "model_bundle_id") != str(
        manifest.get("model_bundle_id")
    ):
        raise ProspectiveLedgerError("Prediction model bundle id mismatch.")
    if _one_string(frame, "model_artifact_sha256") != str(
        manifest.get("model_artifact_sha256")
    ):
        raise ProspectiveLedgerError("Prediction model artifact hash mismatch.")
    if sha256_file(model_path) != manifest.get("model_artifact_sha256"):
        raise ProspectiveLedgerError(
            "Explicit model artifact does not match the freeze manifest."
        )
    if _one_string(frame, "protocol_id") != str(protocol["protocol_id"]):
        raise ProspectiveLedgerError("Prediction protocol_id mismatch.")
    input_sha = _require_sha256(
        _one_string(frame, "input_file_sha256"),
        "input_file_sha256",
    )
    sidecar_sha = _require_sha256(
        _one_string(frame, "capture_sidecar_sha256"),
        "capture_sidecar_sha256",
    )
    indices = frame["index"].astype(str).tolist()
    _validate_capture_binding(
        protocol=protocol,
        protocol_sha256=protocol_hash,
        contest_id=contest_id,
        contest_start=contest_start,
        deadline=deadline,
        indices=indices,
        input_path=input_path,
        input_sha256=input_sha,
        capture_sidecar_path=capture_sidecar_path,
        capture_sidecar_sha256=sidecar_sha,
    )
    _verify_prediction_recomputation(
        prediction_path=prediction_path,
        input_path=input_path,
        capture_sidecar_path=capture_sidecar_path,
        model_path=model_path,
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        contest_start_utc=_format_utc(contest_start),
        prediction_created_at=prediction_created,
    )
    existing = verify_ledger(
        ledger_path,
        allowed_event_types=COMMITMENT_EVENT_TYPES,
    )
    if contest_id in _existing_base_events(existing):
        raise ProspectiveLedgerError(
            f"Contest {contest_id} already has a coverage event."
        )
    payload = {
        "event_type": "prediction_commitment",
        "contest_id": contest_id,
        "operator_contest_start_utc": _format_utc(contest_start),
        "lock_deadline_utc": _format_utc(deadline),
        "prediction_created_at_utc": _format_utc(prediction_created),
        "indices": indices,
        "row_count": len(indices),
        "input_path": _repository_relative(input_path),
        "input_sha256": input_sha,
        "capture_sidecar_path": _repository_relative(capture_sidecar_path),
        "capture_sidecar_sha256": sidecar_sha,
        "prediction_path": _repository_relative(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "model_artifact_path": _repository_relative(model_path),
        "freeze_manifest_path": _repository_relative(manifest_path),
        "model_bundle_id": str(manifest["model_bundle_id"]),
        "model_artifact_sha256": _require_sha256(
            manifest.get("model_artifact_sha256"),
            "manifest.model_artifact_sha256",
        ),
        "freeze_manifest_sha256": manifest_hash,
    }
    return _append_payloads(
        ledger_path,
        [payload],
        protocol=protocol,
        protocol_sha256=protocol_hash,
        allowed_event_types=COMMITMENT_EVENT_TYPES,
    )[0]


def record_operational_miss(
    *,
    contest_id: object,
    contest_start_utc: str,
    miss_stage: str,
    reason_code: str,
    reason_detail: str,
    protocol_path: Path,
    ledger_path: Path = DEFAULT_COMMITMENT_LEDGER,
    evidence_path: Path | None = None,
) -> dict[str, object]:
    protocol, protocol_hash = _load_protocol(protocol_path)
    normalized_id = _normalize_contest_id(contest_id)
    start = _parse_utc(contest_start_utc, "contest_start_utc")
    if _utc_now() < start:
        raise ProspectiveLedgerError(
            "An operational miss cannot be recorded before contest start."
        )
    if reason_code not in MISS_REASON_CODES:
        raise ProspectiveLedgerError(
            f"Unknown operational miss reason_code: {reason_code}"
        )
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", miss_stage):
        raise ProspectiveLedgerError("miss_stage must be a stable snake_case token.")
    if not reason_detail.strip():
        raise ProspectiveLedgerError("Operational miss requires reason_detail.")
    cohort = protocol.get("cohort")
    timepoint = protocol.get("prediction_timepoint")
    if not isinstance(cohort, dict) or not isinstance(timepoint, dict):
        raise ProspectiveLedgerError("Protocol lacks cohort or T0 settings.")
    if not (
        _parse_utc(
            cohort.get("eligibility_start_utc"),
            "cohort.eligibility_start_utc",
        )
        <= start
        <= _parse_utc(
            cohort.get("eligibility_end_utc"),
            "cohort.eligibility_end_utc",
        )
    ):
        raise ProspectiveLedgerError("Operational miss is outside the cohort.")
    events = verify_ledger(
        ledger_path,
        allowed_event_types=COMMITMENT_EVENT_TYPES,
    )
    bases = _existing_base_events(events)
    if normalized_id in bases:
        raise ProspectiveLedgerError(
            "Direct operational misses cannot replace or invalidate an existing "
            "coverage event; use prediction invalidation for a committed prediction."
        )
    minutes = timepoint.get("lock_deadline_minutes_after_contest_start")
    if isinstance(minutes, bool) or not isinstance(minutes, int) or minutes < 1:
        raise ProspectiveLedgerError("Protocol has an invalid T0 deadline.")
    deadline = start + timedelta(minutes=minutes)
    evidence_text: str | None = None
    evidence_sha: str | None = None
    if evidence_path is not None:
        evidence_text = _repository_relative(evidence_path)
        evidence_sha = sha256_file(evidence_path)
    payload = {
        "event_type": "operational_miss",
        "contest_id": normalized_id,
        "operator_contest_start_utc": _format_utc(start),
        "lock_deadline_utc": _format_utc(deadline),
        "miss_stage": miss_stage,
        "reason_code": reason_code,
        "reason_detail": reason_detail.strip(),
        "evidence_path": evidence_text,
        "evidence_sha256": evidence_sha,
    }
    return _append_payloads(
        ledger_path,
        [payload],
        protocol=protocol,
        protocol_sha256=protocol_hash,
        allowed_event_types=COMMITMENT_EVENT_TYPES,
    )[0]


def _find_prediction_event(
    events: Sequence[Mapping[str, object]],
    contest_id: str,
) -> Mapping[str, object]:
    event = _existing_base_events(events).get(contest_id)
    if event is None or event.get("event_type") != "prediction_commitment":
        raise ProspectiveLedgerError(
            f"Contest {contest_id} has no prediction commitment."
        )
    return event


def _find_coverage_event(
    events: Sequence[Mapping[str, object]],
    contest_id: str,
) -> Mapping[str, object]:
    event = _existing_base_events(events).get(contest_id)
    if event is None:
        raise ProspectiveLedgerError(
            f"Contest {contest_id} has no base coverage event."
        )
    return event


def _git_blob_at_commit(commit_sha: str, path: Path) -> bytes:
    repository_path = _repository_relative(path)
    result = subprocess.run(
        ["git", "show", f"{commit_sha}:{repository_path}"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ProspectiveLedgerError(
            f"Commit {commit_sha} does not contain {repository_path}."
        )
    return result.stdout


def _git_commit_contains_event(
    commit_sha: str,
    ledger_path: Path,
    event_sha256: str,
) -> None:
    raw = _git_blob_at_commit(commit_sha, ledger_path)
    events = _verify_ledger_bytes(
        raw,
        label=f"{commit_sha}:{_repository_relative(ledger_path)}",
        allowed_event_types=COMMITMENT_EVENT_TYPES,
    )
    if not any(event.get("event_sha256") == event_sha256 for event in events):
        raise ProspectiveLedgerError(
            "GitHub witness target commit does not contain the target event."
        )


def _validate_event_artifacts_at_commit(
    commit_sha: str,
    event: Mapping[str, object],
) -> None:
    pairs: tuple[tuple[str, str], ...]
    if event.get("event_type") == "prediction_commitment":
        pairs = (
            ("input_path", "input_sha256"),
            ("capture_sidecar_path", "capture_sidecar_sha256"),
            ("prediction_path", "prediction_sha256"),
            ("model_artifact_path", "model_artifact_sha256"),
            ("freeze_manifest_path", "freeze_manifest_sha256"),
        )
    else:
        pairs = (("evidence_path", "evidence_sha256"),)
    for path_field, hash_field in pairs:
        if event.get(path_field) is None:
            continue
        path = Path(str(event[path_field]))
        blob = _git_blob_at_commit(commit_sha, path)
        if hashlib.sha256(blob).hexdigest() != event[hash_field]:
            raise ProspectiveLedgerError(
                f"GitHub witness target commit has a mismatched {path_field}."
            )


def _fetch_github_run_response(url: str) -> bytes:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "cf-difficulty-prospective-v2",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = getattr(response, "status", None)
            if status != 200:
                raise ProspectiveLedgerError(
                    f"GitHub workflow-run API returned HTTP {status}."
                )
            return response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ProspectiveLedgerError(
            f"Cannot fetch the GitHub workflow-run API response: {error}"
        ) from error


def _target_is_default_branch_ancestor(commit_sha: str, branch: str) -> None:
    candidates = [f"refs/remotes/origin/{branch}", f"refs/heads/{branch}"]
    found = False
    for candidate in candidates:
        exists = subprocess.run(
            ["git", "rev-parse", "--verify", f"{candidate}^{{commit}}"],
            capture_output=True,
            check=False,
        )
        if exists.returncode != 0:
            continue
        found = True
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit_sha, candidate],
            capture_output=True,
            check=False,
        )
        if ancestor.returncode == 0:
            return
    if not found:
        raise ProspectiveLedgerError(
            f"Cannot resolve local or origin default branch {branch!r}."
        )
    raise ProspectiveLedgerError(
        "Witness target commit is not an ancestor of the default branch."
    )


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
        raise ProspectiveLedgerError(
            f"Witness API evidence already exists: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def record_github_actions_witness(
    run_id: int,
    *,
    contest_id: object,
    protocol_path: Path,
    ledger_path: Path = DEFAULT_COMMITMENT_LEDGER,
    evidence_dir: Path = DEFAULT_WITNESS_EVIDENCE_DIR,
    fetcher: Callable[[str], bytes] | None = None,
) -> dict[str, object]:
    """Fetch, validate, preserve, and append one external GitHub time witness."""

    protocol, protocol_hash = _load_protocol(protocol_path)
    external = _external_timestamp_policy(protocol)
    normalized_run_id = _require_positive_integer(run_id, "run_id")
    normalized_id = _normalize_contest_id(contest_id)
    events = verify_ledger(
        ledger_path,
        allowed_event_types=COMMITMENT_EVENT_TYPES,
    )
    coverage = _find_coverage_event(events, normalized_id)
    if any(
        event.get("event_type") == "github_actions_witness"
        and event.get("target_event_sha256") == coverage.get("event_sha256")
        for event in events
    ):
        raise ProspectiveLedgerError(
            f"Contest {normalized_id} already has a GitHub witness."
        )

    repository_name = str(external["repository_full_name"])
    api_url = (
        f"https://api.github.com/repos/{repository_name}/actions/runs/"
        f"{normalized_run_id}"
    )
    raw = (fetcher or _fetch_github_run_response)(api_url)
    if not isinstance(raw, bytes):
        raise ProspectiveLedgerError("GitHub run fetcher must return exact bytes.")
    run = _decode_json_object_bytes(raw, "GitHub workflow run response")
    repository = run.get("repository")
    if not isinstance(repository, dict):
        raise ProspectiveLedgerError("Workflow run lacks repository metadata.")
    repository_id = _require_positive_integer(
        repository.get("id"), "run.repository.id"
    )
    workflow_id = _require_positive_integer(
        run.get("workflow_id"), "run.workflow_id"
    )
    response_run_id = _require_positive_integer(run.get("id"), "run.id")
    run_attempt = _require_positive_integer(
        run.get("run_attempt"), "run.run_attempt"
    )
    if response_run_id != normalized_run_id:
        raise ProspectiveLedgerError("GitHub API response run id mismatch.")

    head_sha = _require_commit_sha(run.get("head_sha"), "run.head_sha")
    expected = {
        "repository_id": external["repository_id"],
        "repository_full_name": repository_name,
        "head_branch": external["default_branch"],
        "event": external["trigger_event"],
        "path": external["workflow_path"],
        "name": external["workflow_name"],
        "status": external["required_status"],
        "conclusion": external["required_conclusion"],
        "run_attempt": external["required_run_attempt"],
    }
    actual = {
        "repository_id": repository_id,
        "repository_full_name": repository.get("full_name"),
        "head_branch": run.get("head_branch"),
        "event": run.get("event"),
        "path": run.get("path"),
        "name": run.get("name"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "run_attempt": run_attempt,
    }
    if actual != expected:
        raise ProspectiveLedgerError(
            "Workflow run does not match the frozen witness policy: "
            f"expected={expected!r}, actual={actual!r}."
        )

    created_at = _require_canonical_utc(run.get("created_at"), "run.created_at")
    deadline = _require_canonical_utc(
        coverage.get("lock_deadline_utc"),
        "coverage.lock_deadline_utc",
    )
    target_event_sha = _require_sha256(
        coverage.get("event_sha256"),
        "coverage.event_sha256",
    )
    _git_commit_contains_event(head_sha, ledger_path, target_event_sha)
    _validate_event_artifacts_at_commit(head_sha, coverage)
    workflow_blob = _git_blob_at_commit(
        head_sha, Path(str(external["workflow_path"]))
    )
    workflow_sha = hashlib.sha256(workflow_blob).hexdigest()
    if workflow_sha != external["workflow_file_sha256"]:
        raise ProspectiveLedgerError(
            "Witness target commit contains the wrong workflow bytes."
        )
    _target_is_default_branch_ancestor(
        head_sha, str(external["default_branch"])
    )
    canonical_url = (
        f"https://github.com/{repository_name}/actions/runs/{normalized_run_id}"
    )
    if run.get("html_url") != canonical_url:
        raise ProspectiveLedgerError("Workflow run html_url is not canonical.")

    evidence_path = evidence_dir / f"github-run-{normalized_run_id}.json"
    _repository_relative(evidence_path)
    _write_bytes_exclusive(evidence_path, raw)
    payload = {
        "event_type": "github_actions_witness",
        "contest_id": normalized_id,
        "target_event_sha256": target_event_sha,
        "target_commit_sha": head_sha,
        "run_api_response_path": _repository_relative(evidence_path),
        "run_api_response_sha256": hashlib.sha256(raw).hexdigest(),
        "repository_id": repository_id,
        "repository_full_name": repository_name,
        "head_sha": head_sha,
        "head_branch": str(run["head_branch"]),
        "trigger_event": str(run["event"]),
        "workflow_id": workflow_id,
        "workflow_path": str(run["path"]),
        "workflow_file_sha256": workflow_sha,
        "run_id": normalized_run_id,
        "run_attempt": run_attempt,
        "external_timestamp": {
            "provider": str(external["provider"]),
            "field": str(external["deadline_field"]),
            "value_utc": _format_utc(created_at),
        },
        "status": str(run["status"]),
        "conclusion": str(run["conclusion"]),
        "run_url": canonical_url,
        "timely": created_at <= deadline,
    }
    try:
        return _append_payloads(
            ledger_path,
            [payload],
            protocol=protocol,
            protocol_sha256=protocol_hash,
            allowed_event_types=COMMITMENT_EVENT_TYPES,
        )[0]
    except Exception:
        evidence_path.unlink(missing_ok=True)
        raise


def record_prediction_invalidation(
    *,
    contest_id: object,
    reason_code: str,
    reason_detail: str,
    protocol_path: Path,
    ledger_path: Path = DEFAULT_COMMITMENT_LEDGER,
    evidence_path: Path | None = None,
) -> dict[str, object]:
    protocol, protocol_hash = _load_protocol(protocol_path)
    normalized_id = _normalize_contest_id(contest_id)
    if reason_code not in INVALIDATION_REASON_CODES:
        raise ProspectiveLedgerError(
            f"Invalid prediction invalidation reason: {reason_code}"
        )
    if not reason_detail.strip():
        raise ProspectiveLedgerError("Prediction invalidation requires detail.")
    events = verify_ledger(
        ledger_path,
        allowed_event_types=COMMITMENT_EVENT_TYPES,
    )
    prediction = _find_prediction_event(events, normalized_id)
    if any(
        event.get("event_type") == "prediction_invalidated"
        and event.get("target_event_sha256") == prediction.get("event_sha256")
        for event in events
    ):
        raise ProspectiveLedgerError(
            f"Contest {normalized_id} prediction is already invalidated."
        )
    evidence_text = (
        _repository_relative(evidence_path)
        if evidence_path is not None
        else None
    )
    evidence_sha = (
        sha256_file(evidence_path) if evidence_path is not None else None
    )
    payload = {
        "event_type": "prediction_invalidated",
        "contest_id": normalized_id,
        "target_event_sha256": prediction["event_sha256"],
        "reason_code": reason_code,
        "reason_detail": reason_detail.strip(),
        "evidence_path": evidence_text,
        "evidence_sha256": evidence_sha,
    }
    return _append_payloads(
        ledger_path,
        [payload],
        protocol=protocol,
        protocol_sha256=protocol_hash,
        allowed_event_types=COMMITMENT_EVENT_TYPES,
    )[0]


def build_commitment_state(
    path: Path = DEFAULT_COMMITMENT_LEDGER,
) -> dict[str, dict[str, object]]:
    events = verify_ledger(
        path,
        allowed_event_types=COMMITMENT_EVENT_TYPES,
    )
    bases = _existing_base_events(events)
    by_hash = {
        str(event["event_sha256"]): event
        for event in events
    }
    states: dict[str, dict[str, object]] = {}
    for contest_id, event in bases.items():
        states[contest_id] = {
            "base_event": event,
            "witness": None,
            "invalidation": None,
            "qualified": event.get("event_type") == "prediction_commitment",
        }
    for event in events:
        event_type = event.get("event_type")
        if event_type == "github_actions_witness":
            target = by_hash.get(str(event.get("target_event_sha256")))
            if target is None or target.get("event_type") not in {
                "prediction_commitment",
                "operational_miss",
            }:
                raise ProspectiveLedgerError(
                    "GitHub witness references an unknown coverage event."
                )
            contest_id = str(target["contest_id"])
            if states[contest_id]["witness"] is not None:
                raise ProspectiveLedgerError(
                    f"Contest {contest_id} has duplicate witness events."
                )
            states[contest_id]["witness"] = event
            states[contest_id]["qualified"] = bool(
                target.get("event_type") == "prediction_commitment"
                and event.get("timely")
            )
        elif event_type in {"prediction_invalidated", "operational_miss"}:
            reference = event.get("target_event_sha256")
            if reference is None:
                reference = event.get("invalidates_event_sha256")
            if reference is None:
                continue
            target = by_hash.get(str(reference))
            if target is None or target.get("event_type") != "prediction_commitment":
                raise ProspectiveLedgerError(
                    "Invalidation references an unknown prediction event."
                )
            contest_id = str(target["contest_id"])
            if states[contest_id]["invalidation"] is not None:
                raise ProspectiveLedgerError(
                    f"Contest {contest_id} has duplicate invalidations."
                )
            states[contest_id]["invalidation"] = event
            states[contest_id]["qualified"] = False
    for state in states.values():
        if state["base_event"].get("event_type") == "prediction_commitment":
            if state["witness"] is None:
                state["qualified"] = False
    return states


def _verify_protocol_binding(
    events: Sequence[Mapping[str, object]],
    protocol_path: Path,
) -> None:
    if not events:
        return
    protocol, protocol_sha = _load_protocol(protocol_path)
    if any(
        event.get("protocol_id") != protocol.get("protocol_id")
        or event.get("protocol_sha256") != protocol_sha
        for event in events
    ):
        raise ProspectiveLedgerError(
            "Ledger events do not bind the current frozen protocol."
        )


def _verify_current_event_artifacts(
    events: Sequence[Mapping[str, object]],
    *,
    repository_root: Path = Path("."),
) -> None:
    pairs_by_type = {
        "prediction_commitment": (
            ("input_path", "input_sha256"),
            ("capture_sidecar_path", "capture_sidecar_sha256"),
            ("prediction_path", "prediction_sha256"),
            ("model_artifact_path", "model_artifact_sha256"),
            ("freeze_manifest_path", "freeze_manifest_sha256"),
        ),
        "operational_miss": (("evidence_path", "evidence_sha256"),),
        "prediction_invalidated": (("evidence_path", "evidence_sha256"),),
        "github_actions_witness": (
            ("run_api_response_path", "run_api_response_sha256"),
        ),
        "audit_outcome_snapshot": (
            ("selection_manifest_path", "selection_manifest_sha256"),
            ("raw_snapshot_path", "raw_snapshot_sha256"),
        ),
        "cohort_census_snapshot": (
            ("selection_manifest_path", "selection_manifest_sha256"),
            ("raw_snapshot_path", "raw_snapshot_sha256"),
        ),
        "confirmatory_outcome_snapshot": (
            ("selection_manifest_path", "selection_manifest_sha256"),
            ("raw_snapshot_path", "raw_snapshot_sha256"),
        ),
    }
    root = repository_root.resolve()
    for event in events:
        for path_field, hash_field in pairs_by_type.get(
            str(event.get("event_type")), ()
        ):
            value = event.get(path_field)
            if value is None:
                continue
            relative = Path(_require_repository_path(value, path_field))
            path = root / relative
            try:
                path.resolve().relative_to(root)
            except ValueError as error:
                raise ProspectiveLedgerError(
                    f"Event artifact escapes repository: {value}"
                ) from error
            if sha256_file(path) != event.get(hash_field):
                raise ProspectiveLedgerError(
                    f"Current event artifact hash mismatch: {value}"
                )


def verify_commitment_ledger(
    path: Path = DEFAULT_COMMITMENT_LEDGER,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> dict[str, int]:
    events = verify_ledger(
        path,
        allowed_event_types=COMMITMENT_EVENT_TYPES,
    )
    _verify_protocol_binding(events, protocol_path)
    _verify_current_event_artifacts(events)
    states = build_commitment_state(path)
    return {
        "coverage_contests": len(states),
        "prediction_commitments": sum(
            state["base_event"].get("event_type")
            == "prediction_commitment"
            for state in states.values()
        ),
        "direct_operational_misses": sum(
            state["base_event"].get("event_type") == "operational_miss"
            for state in states.values()
        ),
        "witnessed_predictions": sum(
            state["base_event"].get("event_type") == "prediction_commitment"
            and state["witness"] is not None
            for state in states.values()
        ),
        "timely_coverage_events": sum(
            isinstance(state["witness"], Mapping)
            and state["witness"].get("timely") is True
            for state in states.values()
        ),
        "unwitnessed_coverage_events": sum(
            state["witness"] is None for state in states.values()
        ),
        "qualified_predictions": sum(
            bool(state["qualified"]) for state in states.values()
        ),
        "invalidated_predictions": sum(
            state["invalidation"] is not None for state in states.values()
        ),
    }


def record_snapshot_observation(
    selection_manifest_path: Path,
    *,
    protocol_path: Path,
    observation_ledger_path: Path = DEFAULT_OBSERVATION_LEDGER,
) -> dict[str, object]:
    from cf_diff.prospective_snapshot import (
        ProspectiveSnapshotError,
        verify_snapshot_selection,
    )

    protocol, protocol_hash = _load_protocol(protocol_path)
    manifest = _read_json_object(
        selection_manifest_path,
        "snapshot selection manifest",
    )
    kind = manifest.get("kind")
    event_type_by_kind = {
        "audit_outcome": "audit_outcome_snapshot",
        "cohort_census": "cohort_census_snapshot",
        "confirmatory_outcome": "confirmatory_outcome_snapshot",
    }
    event_type = event_type_by_kind.get(str(kind))
    if event_type is None:
        raise ProspectiveLedgerError(
            f"Unknown snapshot selection kind: {kind!r}"
        )
    if (
        manifest.get("protocol_id") != protocol["protocol_id"]
        or manifest.get("protocol_sha256") != protocol_hash
        or manifest.get("status") != "selected"
    ):
        raise ProspectiveLedgerError(
            "Snapshot selection does not match the frozen protocol."
        )
    if kind in {"cohort_census", "confirmatory_outcome"}:
        try:
            verified = verify_snapshot_selection(
                protocol_path,
                selection_manifest_path.parent,
                expected_kind=str(kind),
            )
        except ProspectiveSnapshotError as error:
            raise ProspectiveLedgerError(
                f"Snapshot selection failed fixed-window verification: {error}"
            ) from error
        if verified != manifest:
            raise ProspectiveLedgerError(
                "Snapshot selection changed during verification."
            )
    raw_path_value = manifest.get("raw_snapshot_path")
    if not isinstance(raw_path_value, str):
        raise ProspectiveLedgerError(
            "Snapshot selection lacks raw_snapshot_path."
        )
    relative_raw = Path(raw_path_value)
    if relative_raw.is_absolute() or ".." in relative_raw.parts:
        raise ProspectiveLedgerError(
            "Snapshot raw response path must stay inside its run directory."
        )
    raw_path = selection_manifest_path.parent / relative_raw
    raw_sha = _require_sha256(
        manifest.get("raw_snapshot_sha256"),
        "selection.raw_snapshot_sha256",
    )
    if sha256_file(raw_path) != raw_sha:
        raise ProspectiveLedgerError("Snapshot raw response hash mismatch.")
    events = verify_ledger(
        observation_ledger_path,
        allowed_event_types=OBSERVATION_EVENT_TYPES,
    )
    if event_type in {
        "cohort_census_snapshot",
        "confirmatory_outcome_snapshot",
    } and any(event.get("event_type") == event_type for event in events):
        raise ProspectiveLedgerError(
            f"Observation chain already has {event_type}."
        )
    payload: dict[str, object] = {
        "event_type": event_type,
        "selection_manifest_path": _repository_relative(
            selection_manifest_path
        ),
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "raw_snapshot_path": _repository_relative(raw_path),
        "raw_snapshot_sha256": raw_sha,
    }
    if event_type == "audit_outcome_snapshot":
        payload["contest_id"] = _normalize_contest_id(
            manifest.get("contest_id")
        )
        if any(
            event.get("event_type") == event_type
            and event.get("contest_id") == payload["contest_id"]
            for event in events
        ):
            raise ProspectiveLedgerError(
                f"Contest {payload['contest_id']} already has an audit snapshot."
            )
    return _append_payloads(
        observation_ledger_path,
        [payload],
        protocol=protocol,
        protocol_sha256=protocol_hash,
        allowed_event_types=OBSERVATION_EVENT_TYPES,
    )[0]


def verify_observation_ledger(
    path: Path = DEFAULT_OBSERVATION_LEDGER,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> dict[str, int]:
    events = verify_ledger(
        path,
        allowed_event_types=OBSERVATION_EVENT_TYPES,
    )
    _verify_protocol_binding(events, protocol_path)
    _verify_current_event_artifacts(events)
    return {
        "events": len(events),
        "audit_outcome_snapshots": sum(
            event.get("event_type") == "audit_outcome_snapshot"
            for event in events
        ),
        "cohort_census_snapshots": sum(
            event.get("event_type") == "cohort_census_snapshot"
            for event in events
        ),
        "confirmatory_outcome_snapshots": sum(
            event.get("event_type") == "confirmatory_outcome_snapshot"
            for event in events
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prediction = subparsers.add_parser(
        "prediction",
        help="append one T0 prediction commitment",
    )
    prediction.add_argument("--predictions", type=Path, required=True)
    prediction.add_argument("--input", type=Path, required=True)
    prediction.add_argument("--capture-sidecar", type=Path, required=True)
    prediction.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    prediction.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    prediction.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    prediction.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_COMMITMENT_LEDGER,
    )

    missed = subparsers.add_parser(
        "miss",
        help="append one operational miss",
    )
    missed.add_argument("--contest-id", required=True)
    missed.add_argument("--contest-start-utc", required=True)
    missed.add_argument("--stage", required=True)
    missed.add_argument("--reason-code", choices=sorted(MISS_REASON_CODES), required=True)
    missed.add_argument("--reason-detail", required=True)
    missed.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    missed.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_COMMITMENT_LEDGER,
    )

    witness = subparsers.add_parser(
        "witness",
        help="fetch and append a GitHub Actions timestamp witness",
    )
    witness.add_argument("--run-id", type=int, required=True)
    witness.add_argument("--contest-id", required=True)
    witness.add_argument(
        "--evidence-dir",
        type=Path,
        default=DEFAULT_WITNESS_EVIDENCE_DIR,
    )
    witness.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    witness.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_COMMITMENT_LEDGER,
    )

    invalid = subparsers.add_parser(
        "invalidate",
        help="append an objective prediction invalidation",
    )
    invalid.add_argument("--contest-id", required=True)
    invalid.add_argument(
        "--reason-code",
        choices=sorted(INVALIDATION_REASON_CODES),
        required=True,
    )
    invalid.add_argument("--reason-detail", required=True)
    invalid.add_argument("--evidence", type=Path, default=None)
    invalid.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    invalid.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_COMMITMENT_LEDGER,
    )

    observation = subparsers.add_parser(
        "observation",
        help="append a verified fixed snapshot selection",
    )
    observation.add_argument("--selection", type=Path, required=True)
    observation.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    observation.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_OBSERVATION_LEDGER,
    )

    verify = subparsers.add_parser("verify", help="verify both event chains")
    verify.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    verify.add_argument(
        "--commitment-ledger",
        type=Path,
        default=DEFAULT_COMMITMENT_LEDGER,
    )
    verify.add_argument(
        "--observation-ledger",
        type=Path,
        default=DEFAULT_OBSERVATION_LEDGER,
    )

    append_only = subparsers.add_parser(
        "check-append-only",
        help="compare both chains with a Git base ref",
    )
    append_only.add_argument("--base-ref", required=True)
    append_only.add_argument(
        "--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH
    )
    append_only.add_argument(
        "--commitment-ledger",
        type=Path,
        default=DEFAULT_COMMITMENT_LEDGER,
    )
    append_only.add_argument(
        "--observation-ledger",
        type=Path,
        default=DEFAULT_OBSERVATION_LEDGER,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "prediction":
            event = record_prediction_commitment(
                args.predictions,
                input_path=args.input,
                capture_sidecar_path=args.capture_sidecar,
                model_path=args.model,
                protocol_path=args.protocol,
                manifest_path=args.manifest,
                ledger_path=args.ledger,
            )
            print(event["event_sha256"])
        elif args.command == "miss":
            event = record_operational_miss(
                contest_id=args.contest_id,
                contest_start_utc=args.contest_start_utc,
                miss_stage=args.stage,
                reason_code=args.reason_code,
                reason_detail=args.reason_detail,
                protocol_path=args.protocol,
                ledger_path=args.ledger,
            )
            print(event["event_sha256"])
        elif args.command == "witness":
            event = record_github_actions_witness(
                args.run_id,
                contest_id=args.contest_id,
                protocol_path=args.protocol,
                ledger_path=args.ledger,
                evidence_dir=args.evidence_dir,
            )
            print(event["event_sha256"])
        elif args.command == "invalidate":
            event = record_prediction_invalidation(
                contest_id=args.contest_id,
                reason_code=args.reason_code,
                reason_detail=args.reason_detail,
                evidence_path=args.evidence,
                protocol_path=args.protocol,
                ledger_path=args.ledger,
            )
            print(event["event_sha256"])
        elif args.command == "observation":
            event = record_snapshot_observation(
                args.selection,
                protocol_path=args.protocol,
                observation_ledger_path=args.ledger,
            )
            print(event["event_sha256"])
        elif args.command == "verify":
            print(
                json.dumps(
                    {
                        "commitments": verify_commitment_ledger(
                            args.commitment_ledger,
                            protocol_path=args.protocol,
                        ),
                        "observations": verify_observation_ledger(
                            args.observation_ledger,
                            protocol_path=args.protocol,
                        ),
                    },
                    sort_keys=True,
                )
            )
        else:
            print(
                json.dumps(
                    check_git_append_only(
                        args.base_ref,
                        [
                            args.commitment_ledger,
                            args.observation_ledger,
                        ],
                        protocol_path=args.protocol,
                    ),
                    sort_keys=True,
                )
            )
    except (ProspectiveLedgerError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
