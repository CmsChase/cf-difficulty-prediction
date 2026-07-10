"""Append-only ledgers for prospective predictions and later rating reveals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import pandas as pd

ZERO_HASH: Final[str] = "0" * 64
PREDICTION_COLUMNS: Final[set[str]] = {
    "contest_id",
    "index",
    "contest_start_utc",
    "primary_prediction",
    "comparator_prediction",
    "feature_row_sha256",
    "protocol_id",
    "model_bundle_id",
    "model_artifact_sha256",
    "prediction_created_at_utc",
}
REVEAL_COLUMNS: Final[set[str]] = {
    "contest_id",
    "index",
    "official_rating",
}
DEFAULT_PREDICTION_LEDGER: Final[Path] = Path(
    "prospective/ledger/predictions.jsonl"
)
DEFAULT_REVEAL_LEDGER: Final[Path] = Path("prospective/ledger/reveals.jsonl")


class ProspectiveLedgerError(RuntimeError):
    """Raised when a prospective ledger operation is unsafe or invalid."""


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON value canonically for hashing."""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProspectiveLedgerError(f"Value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    if not path.is_file():
        raise ProspectiveLedgerError(f"Required file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_append_only_bytes(
    previous: bytes,
    current: bytes,
    *,
    label: str,
) -> None:
    """Require a new ledger to preserve every previously committed byte."""
    if not current.startswith(previous):
        raise ProspectiveLedgerError(
            f"{label} is not append-only: previously committed bytes changed."
        )


def check_git_append_only(
    base_ref: str,
    ledger_paths: list[Path],
) -> dict[str, str]:
    """Compare ledger files with a Git base ref and reject rewrites/deletions."""
    base_check = subprocess.run(
        ["git", "rev-parse", "--verify", f"{base_ref}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if base_check.returncode != 0:
        raise ProspectiveLedgerError(f"Git base ref does not exist: {base_ref}")
    results: dict[str, str] = {}
    for path in ledger_paths:
        repository_path = path.as_posix()
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{base_ref}:{repository_path}"],
            capture_output=True,
            check=False,
        )
        if exists.returncode != 0:
            results[repository_path] = "new_or_absent"
            continue
        previous = subprocess.run(
            ["git", "show", f"{base_ref}:{repository_path}"],
            capture_output=True,
            check=False,
        )
        if previous.returncode != 0:
            raise ProspectiveLedgerError(
                f"Cannot read {repository_path} from Git ref {base_ref}."
            )
        if not path.is_file():
            raise ProspectiveLedgerError(
                f"Previously committed ledger was deleted: {repository_path}"
            )
        assert_append_only_bytes(
            previous.stdout,
            path.read_bytes(),
            label=repository_path,
        )
        results[repository_path] = "append_only"
    return results


def _event_sha256(event_without_hash: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(event_without_hash)).hexdigest()


def _parse_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ProspectiveLedgerError(f"{field} must be a non-empty UTC timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProspectiveLedgerError(f"{field} is not a valid timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise ProspectiveLedgerError(f"{field} must include a timezone.")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ProspectiveLedgerError("Ledger timestamps must include a timezone.")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise ProspectiveLedgerError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveLedgerError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProspectiveLedgerError(f"{label} must contain one JSON object.")
    return payload


def _load_protocol(path: Path) -> tuple[dict[str, object], str]:
    protocol = _read_json_object(path, "protocol")
    if protocol.get("schema_version") != 1:
        raise ProspectiveLedgerError("Protocol schema_version must be 1.")
    if protocol.get("status") != "frozen":
        raise ProspectiveLedgerError("Prospective protocol must be frozen.")
    if not isinstance(protocol.get("protocol_id"), str):
        raise ProspectiveLedgerError("Protocol lacks protocol_id.")
    return protocol, sha256_file(path)


def verify_ledger(
    path: Path,
    *,
    allowed_event_types: set[str] | None = None,
) -> list[dict[str, object]]:
    """Verify a JSONL hash chain and return its events."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        raise ProspectiveLedgerError(f"Ledger has a partial final line: {path}")

    events: list[dict[str, object]] = []
    previous_hash = ZERO_HASH
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        try:
            event = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProspectiveLedgerError(
                f"Ledger line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise ProspectiveLedgerError(
                f"Ledger line {line_number} must be a JSON object."
            )
        if event.get("schema_version") != 1:
            raise ProspectiveLedgerError(
                f"Ledger line {line_number} has unsupported schema_version."
            )
        if event.get("sequence") != line_number:
            raise ProspectiveLedgerError(
                f"Ledger line {line_number} has a non-contiguous sequence."
            )
        if event.get("previous_event_sha256") != previous_hash:
            raise ProspectiveLedgerError(
                f"Ledger line {line_number} breaks the previous-hash chain."
            )
        event_type = event.get("event_type")
        if allowed_event_types is not None and event_type not in allowed_event_types:
            raise ProspectiveLedgerError(
                f"Ledger line {line_number} has unexpected event_type {event_type!r}."
            )
        recorded_hash = event.get("event_sha256")
        if not isinstance(recorded_hash, str):
            raise ProspectiveLedgerError(
                f"Ledger line {line_number} lacks event_sha256."
            )
        unhashed = dict(event)
        del unhashed["event_sha256"]
        expected_hash = _event_sha256(unhashed)
        if recorded_hash != expected_hash:
            raise ProspectiveLedgerError(
                f"Ledger line {line_number} has an invalid event hash."
            )
        previous_hash = recorded_hash
        events.append(event)
    return events


def _append_payloads(
    path: Path,
    payloads: list[dict[str, object]],
    *,
    allowed_event_types: set[str],
) -> list[dict[str, object]]:
    existing = verify_ledger(path, allowed_event_types=allowed_event_types)
    previous_hash = (
        str(existing[-1]["event_sha256"]) if existing else ZERO_HASH
    )
    sequence = len(existing)
    events: list[dict[str, object]] = []
    for payload in payloads:
        sequence += 1
        event = {
            "schema_version": 1,
            **payload,
            "sequence": sequence,
            "previous_event_sha256": previous_hash,
        }
        event["event_sha256"] = _event_sha256(event)
        previous_hash = str(event["event_sha256"])
        events.append(event)

    if events:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = b"".join(
            canonical_json_bytes(event) + b"\n" for event in events
        )
        with path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
    verify_ledger(path, allowed_event_types=allowed_event_types)
    return events


def _normalize_contest_id(value: object) -> str:
    if value is None or pd.isna(value):
        raise ProspectiveLedgerError("contest_id cannot be missing.")
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        numeric = math.nan
    if math.isfinite(numeric) and numeric.is_integer():
        text = str(int(numeric))
    if not text:
        raise ProspectiveLedgerError("contest_id cannot be empty.")
    return text


def _normalize_index(value: object) -> str:
    if value is None or pd.isna(value):
        raise ProspectiveLedgerError("index cannot be missing.")
    text = str(value).strip().upper()
    if not text:
        raise ProspectiveLedgerError("index cannot be empty.")
    return text


def _require_exact_columns(
    frame: pd.DataFrame,
    expected: set[str],
    label: str,
) -> None:
    present = set(str(column) for column in frame.columns)
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing or extra:
        raise ProspectiveLedgerError(
            f"{label} columns do not match the frozen schema; "
            f"missing={missing}, extra={extra}"
        )


def _finite_prediction(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProspectiveLedgerError(f"{field} must be numeric.") from exc
    if not math.isfinite(number):
        raise ProspectiveLedgerError(f"{field} must be finite.")
    return number


def record_prediction_batch(
    prediction_path: Path,
    *,
    protocol_path: Path,
    manifest_path: Path,
    ledger_path: Path,
    recorded_at: datetime | None = None,
) -> dict[str, object]:
    """Validate and append one contest prediction batch."""
    protocol, protocol_hash = _load_protocol(protocol_path)
    manifest = _read_json_object(manifest_path, "model freeze manifest")
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ProspectiveLedgerError("Manifest protocol_id does not match protocol.")
    if manifest.get("protocol_sha256") != protocol_hash:
        raise ProspectiveLedgerError("Manifest protocol hash does not match protocol.")

    frame = pd.read_csv(prediction_path)
    _require_exact_columns(frame, PREDICTION_COLUMNS, "Prediction file")
    if frame.empty:
        raise ProspectiveLedgerError("Prediction file is empty.")

    protocol_ids = set(frame["protocol_id"].astype(str))
    bundle_ids = set(frame["model_bundle_id"].astype(str))
    model_hashes = set(frame["model_artifact_sha256"].astype(str))
    if protocol_ids != {str(protocol["protocol_id"])}:
        raise ProspectiveLedgerError("Prediction protocol_id mismatch.")
    if bundle_ids != {str(manifest.get("model_bundle_id"))}:
        raise ProspectiveLedgerError("Prediction model_bundle_id mismatch.")
    if model_hashes != {str(manifest.get("model_artifact_sha256"))}:
        raise ProspectiveLedgerError("Prediction model artifact hash mismatch.")

    normalized = frame.copy()
    normalized["contest_id"] = normalized["contest_id"].map(_normalize_contest_id)
    normalized["index"] = normalized["index"].map(_normalize_index)
    if normalized.duplicated(["contest_id", "index"]).any():
        raise ProspectiveLedgerError("Prediction batch has duplicate problem keys.")
    contest_ids = set(normalized["contest_id"])
    start_values = set(normalized["contest_start_utc"].astype(str))
    created_values = set(normalized["prediction_created_at_utc"].astype(str))
    if len(contest_ids) != 1 or len(start_values) != 1 or len(created_values) != 1:
        raise ProspectiveLedgerError(
            "One prediction file must contain exactly one contest and timestamp."
        )

    contest_id = next(iter(contest_ids))
    contest_start_text = next(iter(start_values))
    created_text = next(iter(created_values))
    contest_start = _parse_utc(contest_start_text, "contest_start_utc")
    prediction_created = _parse_utc(
        created_text, "prediction_created_at_utc"
    )
    recorded = recorded_at or datetime.now(timezone.utc)
    if recorded.tzinfo is None:
        raise ProspectiveLedgerError("recorded_at must include a timezone.")
    recorded = recorded.astimezone(timezone.utc)

    cohort = protocol.get("cohort")
    timepoint = protocol.get("prediction_timepoint")
    if not isinstance(cohort, dict) or not isinstance(timepoint, dict):
        raise ProspectiveLedgerError("Protocol lacks cohort/timepoint settings.")
    cohort_start = _parse_utc(
        cohort.get("eligibility_start_utc"), "cohort.eligibility_start_utc"
    )
    cohort_end = _parse_utc(
        cohort.get("eligibility_end_utc"), "cohort.eligibility_end_utc"
    )
    if not cohort_start <= contest_start <= cohort_end:
        raise ProspectiveLedgerError("Contest start is outside the frozen cohort.")
    deadline_minutes = timepoint.get("lock_deadline_minutes_after_contest_start")
    if not isinstance(deadline_minutes, int) or deadline_minutes < 1:
        raise ProspectiveLedgerError("Protocol has an invalid prediction deadline.")
    deadline = contest_start + timedelta(minutes=deadline_minutes)
    if prediction_created < contest_start or prediction_created > deadline:
        raise ProspectiveLedgerError(
            "Prediction creation time is outside the frozen lock window."
        )
    if recorded < prediction_created or recorded > deadline:
        raise ProspectiveLedgerError(
            "Ledger recording time is outside the frozen lock window."
        )

    existing = verify_ledger(
        ledger_path,
        allowed_event_types={"prediction_batch", "missed_contest"},
    )
    if any(str(event.get("contest_id")) == contest_id for event in existing):
        raise ProspectiveLedgerError(
            f"Contest {contest_id} already has a prediction or miss event."
        )

    predictions: list[dict[str, object]] = []
    for row in normalized.sort_values("index", kind="mergesort").to_dict(
        orient="records"
    ):
        feature_hash = str(row["feature_row_sha256"])
        if len(feature_hash) != 64:
            raise ProspectiveLedgerError("feature_row_sha256 must have 64 characters.")
        predictions.append(
            {
                "contest_id": contest_id,
                "index": str(row["index"]),
                "primary_prediction": _finite_prediction(
                    row["primary_prediction"], "primary_prediction"
                ),
                "comparator_prediction": _finite_prediction(
                    row["comparator_prediction"], "comparator_prediction"
                ),
                "feature_row_sha256": feature_hash,
            }
        )

    payload = {
        "event_type": "prediction_batch",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "model_bundle_id": manifest["model_bundle_id"],
        "model_artifact_sha256": manifest["model_artifact_sha256"],
        "model_manifest_sha256": sha256_file(manifest_path),
        "source_commit": manifest.get("source_commit"),
        "contest_id": contest_id,
        "contest_start_utc": _format_utc(contest_start),
        "prediction_created_at_utc": _format_utc(prediction_created),
        "recorded_at_utc": _format_utc(recorded),
        "prediction_file_sha256": sha256_file(prediction_path),
        "predictions": predictions,
    }
    return _append_payloads(
        ledger_path,
        [payload],
        allowed_event_types={"prediction_batch", "missed_contest"},
    )[0]


def record_missed_contest(
    *,
    contest_id: str,
    contest_start_utc: str,
    reason: str,
    protocol_path: Path,
    ledger_path: Path,
    noticed_at: datetime | None = None,
) -> dict[str, object]:
    """Append an explicit operational miss instead of silently skipping it."""
    protocol, protocol_hash = _load_protocol(protocol_path)
    normalized_id = _normalize_contest_id(contest_id)
    if not reason.strip():
        raise ProspectiveLedgerError("A missed contest requires a reason.")
    start = _parse_utc(contest_start_utc, "contest_start_utc")
    cohort = protocol.get("cohort")
    if not isinstance(cohort, dict):
        raise ProspectiveLedgerError("Protocol lacks cohort settings.")
    cohort_start = _parse_utc(
        cohort.get("eligibility_start_utc"), "cohort.eligibility_start_utc"
    )
    cohort_end = _parse_utc(
        cohort.get("eligibility_end_utc"), "cohort.eligibility_end_utc"
    )
    if not cohort_start <= start <= cohort_end:
        raise ProspectiveLedgerError("Missed contest is outside the frozen cohort.")
    existing = verify_ledger(
        ledger_path,
        allowed_event_types={"prediction_batch", "missed_contest"},
    )
    if any(str(event.get("contest_id")) == normalized_id for event in existing):
        raise ProspectiveLedgerError(f"Contest {normalized_id} is already recorded.")
    noticed = noticed_at or datetime.now(timezone.utc)
    payload = {
        "event_type": "missed_contest",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "contest_id": normalized_id,
        "contest_start_utc": _format_utc(start),
        "noticed_at_utc": _format_utc(noticed),
        "reason": reason.strip(),
    }
    return _append_payloads(
        ledger_path,
        [payload],
        allowed_event_types={"prediction_batch", "missed_contest"},
    )[0]


def _prediction_by_contest(
    events: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for event in events:
        contest_id = str(event.get("contest_id"))
        if contest_id in output:
            raise ProspectiveLedgerError(f"Duplicate contest event: {contest_id}")
        output[contest_id] = event
    return output


def record_rating_reveal(
    ratings_path: Path,
    *,
    protocol_path: Path,
    prediction_ledger_path: Path,
    reveal_ledger_path: Path,
    revealed_at: datetime | None = None,
) -> dict[str, object]:
    """Append official outcomes without modifying the prediction ledger."""
    protocol, protocol_hash = _load_protocol(protocol_path)
    predictions = verify_ledger(
        prediction_ledger_path,
        allowed_event_types={"prediction_batch", "missed_contest"},
    )
    prediction_map = _prediction_by_contest(predictions)
    before = prediction_ledger_path.read_bytes()

    frame = pd.read_csv(ratings_path)
    _require_exact_columns(frame, REVEAL_COLUMNS, "Rating reveal file")
    if frame.empty:
        raise ProspectiveLedgerError("Rating reveal file is empty.")
    frame["contest_id"] = frame["contest_id"].map(_normalize_contest_id)
    frame["index"] = frame["index"].map(_normalize_index)
    if frame.duplicated(["contest_id", "index"]).any():
        raise ProspectiveLedgerError("Rating reveal has duplicate problem keys.")
    contest_ids = set(frame["contest_id"])
    if len(contest_ids) != 1:
        raise ProspectiveLedgerError("One reveal file must contain one contest.")
    contest_id = next(iter(contest_ids))
    prediction_event = prediction_map.get(contest_id)
    if prediction_event is None or prediction_event.get("event_type") != "prediction_batch":
        raise ProspectiveLedgerError(
            f"Contest {contest_id} has no prediction batch to reveal."
        )

    predicted_rows = prediction_event.get("predictions")
    if not isinstance(predicted_rows, list):
        raise ProspectiveLedgerError("Prediction event has invalid predictions.")
    predicted_keys = {
        (str(row.get("contest_id")), str(row.get("index")))
        for row in predicted_rows
        if isinstance(row, dict)
    }

    existing_reveals = verify_ledger(
        reveal_ledger_path,
        allowed_event_types={"rating_reveal_batch"},
    )
    revealed_keys: set[tuple[str, str]] = set()
    for event in existing_reveals:
        outcomes = event.get("outcomes")
        if not isinstance(outcomes, list):
            raise ProspectiveLedgerError("Reveal event has invalid outcomes.")
        for outcome in outcomes:
            if isinstance(outcome, dict):
                revealed_keys.add(
                    (str(outcome.get("contest_id")), str(outcome.get("index")))
                )

    outcomes: list[dict[str, object]] = []
    for row in frame.sort_values("index", kind="mergesort").to_dict(
        orient="records"
    ):
        key = (contest_id, str(row["index"]))
        if key not in predicted_keys:
            raise ProspectiveLedgerError(f"Reveal key was not predicted: {key}")
        if key in revealed_keys:
            raise ProspectiveLedgerError(f"Rating was already revealed: {key}")
        rating = _finite_prediction(row["official_rating"], "official_rating")
        outcomes.append(
            {
                "contest_id": contest_id,
                "index": str(row["index"]),
                "official_rating": rating,
            }
        )

    reveal_time = (revealed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    contest_start = _parse_utc(
        prediction_event.get("contest_start_utc"), "prediction contest_start_utc"
    )
    outcome_config = protocol.get("outcome")
    if not isinstance(outcome_config, dict):
        raise ProspectiveLedgerError("Protocol lacks outcome settings.")
    delay_hours = outcome_config.get("minimum_reveal_delay_hours_after_contest_start")
    if not isinstance(delay_hours, int) or delay_hours < 0:
        raise ProspectiveLedgerError("Protocol has an invalid reveal delay.")
    if reveal_time < contest_start + timedelta(hours=delay_hours):
        raise ProspectiveLedgerError("Rating reveal is earlier than the frozen delay.")

    payload = {
        "event_type": "rating_reveal_batch",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "contest_id": contest_id,
        "prediction_event_sha256": prediction_event["event_sha256"],
        "source_snapshot_sha256": sha256_file(ratings_path),
        "revealed_at_utc": _format_utc(reveal_time),
        "outcomes": outcomes,
    }
    event = _append_payloads(
        reveal_ledger_path,
        [payload],
        allowed_event_types={"rating_reveal_batch"},
    )[0]
    if prediction_ledger_path.read_bytes() != before:
        raise ProspectiveLedgerError("Reveal operation modified prediction ledger.")
    verify_prospective_ledgers(prediction_ledger_path, reveal_ledger_path)
    return event


def verify_prospective_ledgers(
    prediction_ledger_path: Path,
    reveal_ledger_path: Path,
) -> dict[str, int]:
    """Verify both chains, duplicate keys, and reveal references."""
    prediction_events = verify_ledger(
        prediction_ledger_path,
        allowed_event_types={"prediction_batch", "missed_contest"},
    )
    prediction_map = _prediction_by_contest(prediction_events)
    prediction_hashes = {
        str(event["event_sha256"]): event for event in prediction_events
    }
    prediction_problem_count = 0
    for event in prediction_events:
        if event.get("event_type") != "prediction_batch":
            continue
        rows = event.get("predictions")
        if not isinstance(rows, list) or not rows:
            raise ProspectiveLedgerError("Prediction batch must contain predictions.")
        keys: set[tuple[str, str]] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ProspectiveLedgerError("Prediction row must be an object.")
            key = (str(row.get("contest_id")), str(row.get("index")))
            if key in keys:
                raise ProspectiveLedgerError(f"Duplicate prediction key: {key}")
            keys.add(key)
        prediction_problem_count += len(keys)

    reveal_events = verify_ledger(
        reveal_ledger_path,
        allowed_event_types={"rating_reveal_batch"},
    )
    revealed_keys: set[tuple[str, str]] = set()
    for event in reveal_events:
        reference = str(event.get("prediction_event_sha256"))
        prediction_event = prediction_hashes.get(reference)
        if prediction_event is None or prediction_event.get("event_type") != "prediction_batch":
            raise ProspectiveLedgerError("Reveal references an unknown prediction event.")
        if str(event.get("contest_id")) != str(prediction_event.get("contest_id")):
            raise ProspectiveLedgerError("Reveal contest does not match prediction event.")
        predicted_rows = prediction_event.get("predictions")
        assert isinstance(predicted_rows, list)
        predicted_keys = {
            (str(row.get("contest_id")), str(row.get("index")))
            for row in predicted_rows
            if isinstance(row, dict)
        }
        outcomes = event.get("outcomes")
        if not isinstance(outcomes, list):
            raise ProspectiveLedgerError("Reveal outcomes must be a list.")
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                raise ProspectiveLedgerError("Reveal outcome must be an object.")
            key = (str(outcome.get("contest_id")), str(outcome.get("index")))
            if key not in predicted_keys:
                raise ProspectiveLedgerError(f"Reveal key was not predicted: {key}")
            if key in revealed_keys:
                raise ProspectiveLedgerError(f"Duplicate reveal key: {key}")
            revealed_keys.add(key)

    return {
        "prediction_events": len(prediction_events),
        "prediction_contests": sum(
            event.get("event_type") == "prediction_batch"
            for event in prediction_map.values()
        ),
        "missed_contests": sum(
            event.get("event_type") == "missed_contest"
            for event in prediction_map.values()
        ),
        "predicted_problems": prediction_problem_count,
        "reveal_events": len(reveal_events),
        "revealed_problems": len(revealed_keys),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="append one prediction batch")
    record.add_argument("--predictions", type=Path, required=True)
    record.add_argument("--protocol", type=Path, required=True)
    record.add_argument("--manifest", type=Path, required=True)
    record.add_argument("--ledger", type=Path, required=True)

    missed = subparsers.add_parser("missed", help="record an eligible contest miss")
    missed.add_argument("--contest-id", required=True)
    missed.add_argument("--contest-start-utc", required=True)
    missed.add_argument("--reason", required=True)
    missed.add_argument("--protocol", type=Path, required=True)
    missed.add_argument("--ledger", type=Path, required=True)

    reveal = subparsers.add_parser("reveal", help="append official ratings")
    reveal.add_argument("--ratings", type=Path, required=True)
    reveal.add_argument("--protocol", type=Path, required=True)
    reveal.add_argument("--prediction-ledger", type=Path, required=True)
    reveal.add_argument("--reveal-ledger", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="verify prediction and reveal ledgers")
    verify.add_argument(
        "--prediction-ledger",
        type=Path,
        default=DEFAULT_PREDICTION_LEDGER,
    )
    verify.add_argument(
        "--reveal-ledger",
        type=Path,
        default=DEFAULT_REVEAL_LEDGER,
    )

    append_only = subparsers.add_parser(
        "check-append-only",
        help="compare ledgers with a Git base ref",
    )
    append_only.add_argument("--base-ref", required=True)
    append_only.add_argument(
        "--prediction-ledger",
        type=Path,
        default=DEFAULT_PREDICTION_LEDGER,
    )
    append_only.add_argument(
        "--reveal-ledger",
        type=Path,
        default=DEFAULT_REVEAL_LEDGER,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the prospective ledger CLI."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "record":
            event = record_prediction_batch(
                args.predictions,
                protocol_path=args.protocol,
                manifest_path=args.manifest,
                ledger_path=args.ledger,
            )
            print(event["event_sha256"])
        elif args.command == "missed":
            event = record_missed_contest(
                contest_id=args.contest_id,
                contest_start_utc=args.contest_start_utc,
                reason=args.reason,
                protocol_path=args.protocol,
                ledger_path=args.ledger,
            )
            print(event["event_sha256"])
        elif args.command == "reveal":
            event = record_rating_reveal(
                args.ratings,
                protocol_path=args.protocol,
                prediction_ledger_path=args.prediction_ledger,
                reveal_ledger_path=args.reveal_ledger,
            )
            print(event["event_sha256"])
        elif args.command == "verify":
            print(
                json.dumps(
                    verify_prospective_ledgers(
                        args.prediction_ledger,
                        args.reveal_ledger,
                    ),
                    sort_keys=True,
                )
            )
        else:
            print(
                json.dumps(
                    check_git_append_only(
                        args.base_ref,
                        [args.prediction_ledger, args.reveal_ledger],
                    ),
                    sort_keys=True,
                )
            )
    except ProspectiveLedgerError as exc:
        print(f"Prospective ledger failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
