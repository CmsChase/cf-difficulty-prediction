"""Run the protocol-locked confirmatory prospective analysis."""

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

import numpy as np

from cf_diff.prospective_model import ProspectiveModelError, load_frozen_protocol


DEFAULT_PROTOCOL_PATH: Final[Path] = Path("configs/prospective_protocol_v2.json")
DEFAULT_INPUT_MANIFEST: Final[Path] = Path(
    "prospective/cohort/v2/confirmatory_analysis_input.json"
)
DEFAULT_OUTPUT_PATH: Final[Path] = Path(
    "prospective/results/v2/confirmatory_analysis.json"
)
DEFAULT_COMMITMENT_LEDGER: Final[Path] = Path(
    "prospective/ledger/commitments.jsonl"
)
DEFAULT_OBSERVATION_LEDGER: Final[Path] = Path(
    "prospective/ledger/observations.jsonl"
)

FINALIZED_COLUMNS: Final[tuple[str, ...]] = (
    "contest_id",
    "index",
    "primary_prediction",
    "comparator_prediction",
    "official_problem_present",
    "official_problem_type",
    "official_rating",
    "outcome_status",
    "eligible_for_analysis",
)
OUTCOME_STATUSES: Final[frozenset[str]] = frozenset(
    {"paired", "missing_rating", "non_programming", "absent_from_snapshot"}
)
MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "protocol_id",
        "protocol_sha256",
        "finalized_at_utc",
        "analysis_not_before_utc",
        "finalized_csv",
        "mapping_report",
        "commitment_ledger_sha256",
        "observation_ledger_sha256",
        "cohort_census_selection",
        "confirmatory_outcome_selection",
    }
)
REPORT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "protocol_id",
        "protocol_sha256",
        "census_integrity_status",
        "outcome_integrity_status",
        "census_contest_count",
        "mapped_once_contest_count",
        "unmapped_contest_count",
        "multiply_mapped_contest_count",
        "qualified_prediction_contest_count",
        "operational_miss_contest_count",
        "timely_coverage_event_count",
        "late_coverage_event_count",
        "missing_coverage_witness_count",
        "prespecified_exclusion_contest_count",
        "locked_prediction_contest_count",
        "locked_prediction_row_count",
        "paired_problem_row_count",
        "missing_rating_row_count",
        "non_programming_row_count",
        "absent_from_snapshot_row_count",
        "contest_dispositions",
    }
)
DISPOSITION_FIELDS: Final[frozenset[str]] = frozenset(
    {"contest_id", "disposition", "locked_prediction_row_count", "reason_code"}
)


class ProspectiveAnalysisError(RuntimeError):
    """Raised before or during a confirmatory-analysis integrity gate."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ProspectiveAnalysisError(f"{field} must be a UTC timestamp.")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ProspectiveAnalysisError(f"{field} is not ISO-8601.") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProspectiveAnalysisError(f"{field} must include a UTC offset.")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ProspectiveAnalysisError("Timestamp must be timezone-aware.")
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ProspectiveAnalysisError(f"Required analysis artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ProspectiveAnalysisError(f"{field} must be a lowercase SHA-256.")
    return value


def _read_json(path: Path, label: str) -> dict[str, object]:
    def reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ProspectiveAnalysisError(f"Duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ProspectiveAnalysisError(f"Non-finite JSON value in {label}: {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProspectiveAnalysisError(f"Cannot read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ProspectiveAnalysisError(f"{label} must be a JSON object.")
    return payload


def _safe_relative(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ProspectiveAnalysisError(f"{field} must be a relative POSIX path.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ProspectiveAnalysisError(f"{field} is not a safe relative path.")
    return path


def _positive_int(value: object, field: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        adjective = "non-negative" if allow_zero else "positive"
        raise ProspectiveAnalysisError(f"{field} must be a {adjective} integer.")
    return value


def _canonical_contest_id(value: object) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[1-9][0-9]*", text):
        raise ProspectiveAnalysisError(f"Invalid canonical contest_id: {value!r}")
    return text


def _canonical_index(value: object) -> str:
    text = str(value).strip().upper()
    if not re.fullmatch(r"[A-Z0-9]+", text):
        raise ProspectiveAnalysisError(f"Invalid canonical problem index: {value!r}")
    return text


def _finite_float(value: object, field: str) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as error:
        raise ProspectiveAnalysisError(f"{field} must be numeric.") from error
    if not math.isfinite(number):
        raise ProspectiveAnalysisError(f"{field} must be finite.")
    return number


def _parse_bool(value: str, field: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ProspectiveAnalysisError(f"{field} must be lowercase true or false.")


def _load_protocol(
    protocol_path: Path,
) -> tuple[dict[str, object], str, dict[str, object]]:
    try:
        protocol = load_frozen_protocol(protocol_path)
    except ProspectiveModelError as error:
        raise ProspectiveAnalysisError(str(error)) from error
    analysis = protocol.get("confirmatory_analysis")
    if not isinstance(analysis, dict):
        raise ProspectiveAnalysisError("Protocol lacks confirmatory analysis settings.")
    bootstrap = analysis.get("bootstrap")
    success = analysis.get("success_rule")
    if not isinstance(bootstrap, dict) or not isinstance(success, dict):
        raise ProspectiveAnalysisError("Protocol analysis settings are not structured.")
    if (
        bootstrap.get("resamples") != 10_000
        or bootstrap.get("seed") != 20_260_710
        or bootstrap.get("bit_generator") != "PCG64"
        or bootstrap.get("quantile_probabilities") != [0.025, 0.975]
        or bootstrap.get("quantile_method") != "linear"
        or analysis.get("minimum_distinct_paired_contests") != 30
        or analysis.get("minimum_paired_problem_rows") != 200
        or success
        != {
            "criterion": "lower_confidence_bound_strictly_greater_than_zero",
            "boundary_value": 0,
            "equality_is_success": False,
        }
    ):
        raise ProspectiveAnalysisError("Protocol analysis constants are not frozen.")
    return protocol, _sha256_file(protocol_path), analysis


def _validate_selection_binding(
    value: object,
    *,
    field: str,
    window: Mapping[str, object],
) -> datetime:
    expected_fields = {
        "selection_manifest_sha256",
        "raw_snapshot_sha256",
        "selected_at_utc",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ProspectiveAnalysisError(f"{field} has non-exact fields.")
    _require_sha256(value.get("selection_manifest_sha256"), f"{field}.selection")
    _require_sha256(value.get("raw_snapshot_sha256"), f"{field}.raw")
    selected = _parse_utc(value.get("selected_at_utc"), f"{field}.selected_at_utc")
    anchor = _parse_utc(window.get("anchor_utc"), f"{field}.anchor")
    deadline = _parse_utc(window.get("deadline_utc"), f"{field}.deadline")
    if not anchor <= selected < deadline:
        raise ProspectiveAnalysisError(f"{field} selection is outside its half-open window.")
    return selected


def _load_mapping_report(
    path: Path,
    *,
    protocol_id: str,
    protocol_sha: str,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    report = _read_json(path, "mapping report")
    if frozenset(report) != REPORT_FIELDS:
        raise ProspectiveAnalysisError("Mapping report has non-exact fields.")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "confirmatory_mapping_report"
        or report.get("status") != "finalized"
        or report.get("protocol_id") != protocol_id
        or report.get("protocol_sha256") != protocol_sha
        or report.get("census_integrity_status") != "passed"
        or report.get("outcome_integrity_status") != "passed"
        or report.get("unmapped_contest_count") != 0
        or report.get("multiply_mapped_contest_count") != 0
        or report.get("prespecified_exclusion_contest_count") != 0
    ):
        raise ProspectiveAnalysisError("Mapping integrity gate did not pass.")
    dispositions_raw = report.get("contest_dispositions")
    if not isinstance(dispositions_raw, list):
        raise ProspectiveAnalysisError("Mapping dispositions must be a list.")
    dispositions: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for item in dispositions_raw:
        if not isinstance(item, dict) or frozenset(item) != DISPOSITION_FIELDS:
            raise ProspectiveAnalysisError("Contest disposition has non-exact fields.")
        contest_id = _canonical_contest_id(item.get("contest_id"))
        if contest_id in dispositions:
            raise ProspectiveAnalysisError("Contest dispositions are not unique.")
        disposition = item.get("disposition")
        if disposition not in {
            "qualified_prediction",
            "operational_miss",
            "prespecified_exclusion",
        }:
            raise ProspectiveAnalysisError("Contest disposition is invalid.")
        rows = _positive_int(
            item.get("locked_prediction_row_count"),
            "locked prediction row count",
            allow_zero=True,
        )
        reason = item.get("reason_code")
        if disposition == "qualified_prediction":
            if reason is not None or rows < 1:
                raise ProspectiveAnalysisError("Qualified disposition is malformed.")
        elif not isinstance(reason, str) or not reason:
            raise ProspectiveAnalysisError("Non-qualified disposition needs a reason.")
        if disposition == "prespecified_exclusion":
            raise ProspectiveAnalysisError("Protocol has no exclusion category.")
        dispositions[contest_id] = dict(item)
        order.append(contest_id)
    if order != sorted(order):
        raise ProspectiveAnalysisError("Contest dispositions are not lexicographically sorted.")
    census_count = _positive_int(
        report.get("census_contest_count"), "census contest count", allow_zero=True
    )
    timely = _positive_int(
        report.get("timely_coverage_event_count"),
        "timely coverage count",
        allow_zero=True,
    )
    late = _positive_int(
        report.get("late_coverage_event_count"),
        "late coverage count",
        allow_zero=True,
    )
    missing_witness = _positive_int(
        report.get("missing_coverage_witness_count"),
        "missing coverage witness count",
        allow_zero=True,
    )
    if (
        len(dispositions) != census_count
        or report.get("mapped_once_contest_count") != census_count
        or timely + late + missing_witness != census_count
        or report.get("qualified_prediction_contest_count")
        != sum(item["disposition"] == "qualified_prediction" for item in dispositions.values())
        or report.get("operational_miss_contest_count")
        != sum(item["disposition"] == "operational_miss" for item in dispositions.values())
    ):
        raise ProspectiveAnalysisError("Mapping report contest counts disagree.")
    return report, dispositions


def _load_finalized_rows(
    path: Path,
    dispositions: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FINALIZED_COLUMNS:
                raise ProspectiveAnalysisError("Finalized CSV header is not exact.")
            raw_rows = list(reader)
    except OSError as error:
        raise ProspectiveAnalysisError(f"Cannot read finalized CSV: {error}") from error
    rows: list[dict[str, object]] = []
    keys: set[tuple[str, str]] = set()
    ordered_keys: list[tuple[str, str]] = []
    for raw in raw_rows:
        contest_id = _canonical_contest_id(raw["contest_id"])
        index = _canonical_index(raw["index"])
        key = (contest_id, index)
        if key in keys:
            raise ProspectiveAnalysisError("Finalized CSV has duplicate problem keys.")
        keys.add(key)
        ordered_keys.append(key)
        if contest_id not in dispositions:
            raise ProspectiveAnalysisError("Finalized CSV contest is absent from mapping.")
        primary = _finite_float(raw["primary_prediction"], "primary_prediction")
        comparator = _finite_float(
            raw["comparator_prediction"], "comparator_prediction"
        )
        present = _parse_bool(raw["official_problem_present"], "official_problem_present")
        eligible = _parse_bool(raw["eligible_for_analysis"], "eligible_for_analysis")
        problem_type = raw["official_problem_type"]
        rating_text = raw["official_rating"]
        status = raw["outcome_status"]
        if status not in OUTCOME_STATUSES:
            raise ProspectiveAnalysisError("Outcome status is invalid.")
        rating: float | None = None
        if status == "paired":
            if not present or problem_type != "PROGRAMMING" or not rating_text:
                raise ProspectiveAnalysisError("Paired outcome row is malformed.")
            rating = _finite_float(rating_text, "official_rating")
        elif status == "missing_rating":
            if not present or problem_type != "PROGRAMMING" or rating_text:
                raise ProspectiveAnalysisError("Missing-rating row is malformed.")
        elif status == "non_programming":
            if not present or not problem_type or problem_type == "PROGRAMMING" or rating_text:
                raise ProspectiveAnalysisError("Non-programming row is malformed.")
        else:
            if present or problem_type or rating_text:
                raise ProspectiveAnalysisError("Absent-snapshot row is malformed.")
        qualified = dispositions[contest_id]["disposition"] == "qualified_prediction"
        if eligible != (status == "paired" and qualified):
            raise ProspectiveAnalysisError("eligible_for_analysis is inconsistent.")
        rows.append(
            {
                "contest_id": contest_id,
                "index": index,
                "primary_prediction": primary,
                "comparator_prediction": comparator,
                "official_problem_present": present,
                "official_problem_type": problem_type,
                "official_rating": rating,
                "outcome_status": status,
                "eligible_for_analysis": eligible,
            }
        )
    if ordered_keys != sorted(ordered_keys):
        raise ProspectiveAnalysisError("Finalized CSV keys are not stably sorted.")
    return rows


def _cross_check_report(
    report: Mapping[str, object],
    dispositions: Mapping[str, Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
) -> None:
    per_contest = {contest_id: 0 for contest_id in dispositions}
    counts = {status: 0 for status in OUTCOME_STATUSES}
    for row in rows:
        per_contest[str(row["contest_id"])] += 1
        counts[str(row["outcome_status"])] += 1
    if any(
        per_contest[contest_id] != item["locked_prediction_row_count"]
        for contest_id, item in dispositions.items()
    ):
        raise ProspectiveAnalysisError("Disposition row counts disagree with CSV.")
    expected = {
        "locked_prediction_contest_count": sum(value > 0 for value in per_contest.values()),
        "locked_prediction_row_count": len(rows),
        "paired_problem_row_count": counts["paired"],
        "missing_rating_row_count": counts["missing_rating"],
        "non_programming_row_count": counts["non_programming"],
        "absent_from_snapshot_row_count": counts["absent_from_snapshot"],
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ProspectiveAnalysisError(f"Mapping report {field} disagrees with CSV.")


def _validate_event_chain_mapping(
    protocol_path: Path,
    commitment_ledger_path: Path,
    observation_ledger_path: Path,
    dispositions: Mapping[str, Mapping[str, object]],
    report: Mapping[str, object],
) -> None:
    from cf_diff.prospective_ledger import (
        ProspectiveLedgerError,
        build_commitment_state,
        verify_commitment_ledger,
        verify_observation_ledger,
    )

    try:
        verify_commitment_ledger(
            commitment_ledger_path, protocol_path=protocol_path
        )
        observation = verify_observation_ledger(
            observation_ledger_path, protocol_path=protocol_path
        )
        states = build_commitment_state(commitment_ledger_path)
    except ProspectiveLedgerError as error:
        raise ProspectiveAnalysisError(
            f"Event-chain verification failed: {error}"
        ) from error
    if (
        observation["cohort_census_snapshots"] != 1
        or observation["confirmatory_outcome_snapshots"] != 1
    ):
        raise ProspectiveAnalysisError(
            "Observation chain lacks the fixed census or outcome snapshot."
        )
    if set(states) != set(dispositions):
        raise ProspectiveAnalysisError(
            "Mapping report contests differ from the commitment chain."
        )
    witness_counts = {
        "timely_coverage_event_count": sum(
            isinstance(state["witness"], Mapping)
            and state["witness"].get("timely") is True
            for state in states.values()
        ),
        "late_coverage_event_count": sum(
            isinstance(state["witness"], Mapping)
            and state["witness"].get("timely") is False
            for state in states.values()
        ),
        "missing_coverage_witness_count": sum(
            state["witness"] is None for state in states.values()
        ),
    }
    if any(report.get(field) != value for field, value in witness_counts.items()):
        raise ProspectiveAnalysisError(
            "Mapping report coverage-witness counts differ from the event chain."
        )
    for contest_id, disposition in dispositions.items():
        state = states[contest_id]
        base = state["base_event"]
        assert isinstance(base, Mapping)
        expected_rows = (
            int(base["row_count"])
            if base.get("event_type") == "prediction_commitment"
            else 0
        )
        if disposition["locked_prediction_row_count"] != expected_rows:
            raise ProspectiveAnalysisError(
                "Mapping report row count differs from commitment event."
            )
        if disposition["disposition"] == "qualified_prediction":
            if base.get("event_type") != "prediction_commitment" or not bool(
                state["qualified"]
            ):
                raise ProspectiveAnalysisError(
                    "Qualified mapping is not qualified in the commitment chain."
                )
        elif base.get("event_type") == "prediction_commitment" and bool(
            state["qualified"]
        ):
            raise ProspectiveAnalysisError(
                "Mapping silently excludes a qualified prediction commitment."
            )


def _cluster_bootstrap(
    improvements_by_contest: Mapping[str, Sequence[float]],
    *,
    resamples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use one PCG64 integer draw and expanded problem-level cluster means."""

    try:
        contest_ids = sorted(improvements_by_contest, key=lambda value: int(value))
    except (TypeError, ValueError) as error:
        raise ProspectiveAnalysisError(
            "Bootstrap contest ids must be normalized integers."
        ) from error
    if not contest_ids:
        raise ProspectiveAnalysisError("Bootstrap requires at least one contest.")
    sums = np.asarray(
        [np.asarray(improvements_by_contest[key], dtype=np.float64).sum() for key in contest_ids],
        dtype=np.float64,
    )
    counts = np.asarray(
        [len(improvements_by_contest[key]) for key in contest_ids], dtype=np.int64
    )
    if (counts <= 0).any() or not np.isfinite(sums).all():
        raise ProspectiveAnalysisError("Bootstrap clusters are empty or non-finite.")
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(
        0,
        len(contest_ids),
        size=(resamples, len(contest_ids)),
        endpoint=False,
        dtype=np.int64,
    )
    replicate_sums = sums[draws].sum(axis=1, dtype=np.float64)
    replicate_counts = counts[draws].sum(axis=1, dtype=np.int64)
    return replicate_sums / replicate_counts, draws


def _band(index: str) -> str:
    match = re.match(r"([A-Z])", index)
    if match is None:
        return "unclassified"
    letter = match.group(1)
    if letter <= "B":
        return "A-B"
    if letter <= "D":
        return "C-D"
    return "E+"


def _calculate_results(
    eligible: Sequence[Mapping[str, object]],
    analysis: Mapping[str, object],
) -> dict[str, object]:
    by_contest: dict[str, list[float]] = {}
    primary_errors: list[float] = []
    comparator_errors: list[float] = []
    bands: dict[str, list[tuple[float, float]]] = {
        "A-B": [],
        "C-D": [],
        "E+": [],
        "unclassified": [],
    }
    contest_errors: dict[str, tuple[list[float], list[float]]] = {}
    for row in eligible:
        rating = float(row["official_rating"])
        primary_error = abs(rating - float(row["primary_prediction"]))
        comparator_error = abs(rating - float(row["comparator_prediction"]))
        improvement = comparator_error - primary_error
        contest_id = str(row["contest_id"])
        by_contest.setdefault(contest_id, []).append(improvement)
        primary_errors.append(primary_error)
        comparator_errors.append(comparator_error)
        pair = contest_errors.setdefault(contest_id, ([], []))
        pair[0].append(primary_error)
        pair[1].append(comparator_error)
        bands[_band(str(row["index"]))].append((primary_error, comparator_error))
    bootstrap = analysis["bootstrap"]
    assert isinstance(bootstrap, dict)
    replicates, _ = _cluster_bootstrap(
        by_contest,
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]),
    )
    probabilities = bootstrap["quantile_probabilities"]
    assert isinstance(probabilities, list)
    lower, upper = np.quantile(
        replicates,
        probabilities,
        method=str(bootstrap["quantile_method"]),
    )
    primary_array = np.asarray(primary_errors, dtype=np.float64)
    comparator_array = np.asarray(comparator_errors, dtype=np.float64)
    point = float((comparator_array - primary_array).mean())
    contest_primary = [float(np.mean(values[0])) for values in contest_errors.values()]
    contest_comparator = [float(np.mean(values[1])) for values in contest_errors.values()]
    band_results: dict[str, object] = {}
    for name, values in bands.items():
        if not values:
            band_results[name] = {
                "row_count": 0,
                "primary_mae": None,
                "comparator_mae": None,
                "paired_mae_improvement": None,
            }
            continue
        primary = np.asarray([value[0] for value in values], dtype=np.float64)
        comparator = np.asarray([value[1] for value in values], dtype=np.float64)
        band_results[name] = {
            "row_count": len(values),
            "primary_mae": float(primary.mean()),
            "comparator_mae": float(comparator.mean()),
            "paired_mae_improvement": float((comparator - primary).mean()),
        }
    return {
        "primary": {
            "estimand": "paired_mae_improvement",
            "point_estimate": point,
            "confidence_interval": [float(lower), float(upper)],
            "confidence_level": 0.95,
            "bootstrap_resamples": int(bootstrap["resamples"]),
            "confirmatory_success": bool(float(lower) > 0.0),
        },
        "secondary": {
            "problem_level_mae": {
                "primary": float(primary_array.mean()),
                "comparator": float(comparator_array.mean()),
            },
            "contest_macro_mae": {
                "primary": float(np.mean(contest_primary)),
                "comparator": float(np.mean(contest_comparator)),
            },
            "within_100_accuracy": {
                "primary": float(np.mean(primary_array <= 100.0)),
                "comparator": float(np.mean(comparator_array <= 100.0)),
            },
            "within_200_accuracy": {
                "primary": float(np.mean(primary_array <= 200.0)),
                "comparator": float(np.mean(comparator_array <= 200.0)),
            },
            "index_rank_bands": band_results,
        },
    }


def run_confirmatory_analysis(
    protocol_path: Path,
    input_manifest_path: Path,
    output_path: Path,
    *,
    commitment_ledger_path: Path = DEFAULT_COMMITMENT_LEDGER,
    observation_ledger_path: Path = DEFAULT_OBSERVATION_LEDGER,
    clock: Callable[[], datetime] = _utc_now,
    event_chain_validator: Callable[
        [
            Path,
            Path,
            Path,
            Mapping[str, Mapping[str, object]],
            Mapping[str, object],
        ],
        None,
    ] = _validate_event_chain_mapping,
) -> dict[str, object]:
    """Verify every gate, then calculate and exclusively publish the result."""

    protocol, protocol_sha, analysis = _load_protocol(protocol_path)
    not_before = _parse_utc(
        analysis.get("earliest_execution_utc"),
        "confirmatory_analysis.earliest_execution_utc",
    )
    analyzed_at = clock().astimezone(timezone.utc)
    if analyzed_at < not_before:
        raise ProspectiveAnalysisError(
            "Confirmatory analysis is blocked until the frozen execution time."
        )
    manifest = _read_json(input_manifest_path, "analysis input manifest")
    if frozenset(manifest) != MANIFEST_FIELDS:
        raise ProspectiveAnalysisError("Analysis input manifest has non-exact fields.")
    protocol_id = str(protocol["protocol_id"])
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "confirmatory_analysis_input"
        or manifest.get("status") != "finalized"
        or manifest.get("protocol_id") != protocol_id
        or manifest.get("protocol_sha256") != protocol_sha
        or manifest.get("analysis_not_before_utc") != _format_utc(not_before)
    ):
        raise ProspectiveAnalysisError("Analysis input manifest identity mismatch.")
    windows = protocol.get("fixed_snapshot_windows")
    if not isinstance(windows, dict):
        raise ProspectiveAnalysisError("Protocol snapshot windows are absent.")
    census_selected = _validate_selection_binding(
        manifest.get("cohort_census_selection"),
        field="cohort_census_selection",
        window=windows["cohort_census"],
    )
    outcome_selected = _validate_selection_binding(
        manifest.get("confirmatory_outcome_selection"),
        field="confirmatory_outcome_selection",
        window=windows["confirmatory_outcome"],
    )
    finalized_at = _parse_utc(manifest.get("finalized_at_utc"), "finalized_at_utc")
    if finalized_at < max(census_selected, outcome_selected):
        raise ProspectiveAnalysisError("Analysis input predates a selected snapshot.")
    if manifest.get("commitment_ledger_sha256") != _sha256_file(
        commitment_ledger_path
    ) or manifest.get("observation_ledger_sha256") != _sha256_file(
        observation_ledger_path
    ):
        raise ProspectiveAnalysisError("Current event-chain hashes do not match input.")

    finalized_csv = manifest.get("finalized_csv")
    mapping_binding = manifest.get("mapping_report")
    if not isinstance(finalized_csv, dict) or set(finalized_csv) != {
        "path",
        "sha256",
        "row_count",
    }:
        raise ProspectiveAnalysisError("finalized_csv binding is malformed.")
    if not isinstance(mapping_binding, dict) or set(mapping_binding) != {
        "path",
        "sha256",
    }:
        raise ProspectiveAnalysisError("mapping_report binding is malformed.")
    csv_path = input_manifest_path.parent / _safe_relative(
        finalized_csv.get("path"), "finalized_csv.path"
    )
    report_path = input_manifest_path.parent / _safe_relative(
        mapping_binding.get("path"), "mapping_report.path"
    )
    if finalized_csv.get("sha256") != _sha256_file(csv_path):
        raise ProspectiveAnalysisError("Finalized CSV SHA-256 mismatch.")
    if mapping_binding.get("sha256") != _sha256_file(report_path):
        raise ProspectiveAnalysisError("Mapping report SHA-256 mismatch.")
    report, dispositions = _load_mapping_report(
        report_path, protocol_id=protocol_id, protocol_sha=protocol_sha
    )
    event_chain_validator(
        protocol_path,
        commitment_ledger_path,
        observation_ledger_path,
        dispositions,
        report,
    )
    rows = _load_finalized_rows(csv_path, dispositions)
    if finalized_csv.get("row_count") != len(rows):
        raise ProspectiveAnalysisError("Finalized CSV row count mismatch.")
    _cross_check_report(report, dispositions, rows)
    eligible = [row for row in rows if bool(row["eligible_for_analysis"])]
    eligible_contests = {str(row["contest_id"]) for row in eligible}
    minimum_contests = int(analysis["minimum_distinct_paired_contests"])
    minimum_rows = int(analysis["minimum_paired_problem_rows"])
    if len(eligible_contests) < minimum_contests or len(eligible) < minimum_rows:
        raise ProspectiveAnalysisError(
            "Confirmatory sample thresholds are not met; aggregate metrics are blocked."
        )

    calculated = _calculate_results(eligible, analysis)
    result = {
        "schema_version": 1,
        "artifact_type": "prospective_confirmatory_analysis",
        "status": "complete",
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha,
        "analysis_input_manifest_path": input_manifest_path.as_posix(),
        "analysis_input_manifest_sha256": _sha256_file(input_manifest_path),
        "analyzed_at_utc": _format_utc(analyzed_at),
        "eligible_contest_count": len(eligible_contests),
        "eligible_problem_row_count": len(eligible),
        "coverage": {
            field: report[field]
            for field in (
                "census_contest_count",
                "qualified_prediction_contest_count",
                "operational_miss_contest_count",
                "timely_coverage_event_count",
                "late_coverage_event_count",
                "missing_coverage_witness_count",
                "locked_prediction_contest_count",
                "locked_prediction_row_count",
                "paired_problem_row_count",
                "missing_rating_row_count",
                "non_programming_row_count",
                "absent_from_snapshot_row_count",
            )
        },
        **calculated,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output_path)
    except FileExistsError as error:
        raise ProspectiveAnalysisError(
            f"Analysis output already exists: {output_path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_confirmatory_analysis(
            args.protocol, args.input_manifest, args.output
        )
        print(json.dumps(result, sort_keys=True))
    except (ProspectiveAnalysisError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
