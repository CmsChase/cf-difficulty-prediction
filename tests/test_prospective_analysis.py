"""Tests for the deterministic confirmatory prospective analysis."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from cf_diff import prospective_analysis
from cf_diff.prospective_analysis import (
    FINALIZED_COLUMNS,
    ProspectiveAnalysisError,
    _cluster_bootstrap,
    run_confirmatory_analysis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROTOCOL = PROJECT_ROOT / "configs" / "prospective_protocol_v2.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _frozen_protocol(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    protocol = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    protocol["status"] = "frozen"
    protocol["protocol_frozen_at_utc"] = "2026-07-12T00:00:00Z"
    path = tmp_path / "protocol.json"
    _write_json(path, protocol)
    return path, protocol


def _analysis_package(
    tmp_path: Path,
    *,
    contest_count: int = 30,
    rows_per_contest: int = 7,
    improvement: float = 40.0,
) -> dict[str, object]:
    protocol_path, protocol = _frozen_protocol(tmp_path)
    protocol_sha = _sha(protocol_path)
    csv_path = tmp_path / "confirmatory_outcomes.csv"
    rows: list[dict[str, str]] = []
    dispositions: list[dict[str, object]] = []
    indices = [chr(ord("A") + value) for value in range(rows_per_contest)]
    for offset in range(contest_count):
        contest_id = str(1000 + offset)
        dispositions.append(
            {
                "contest_id": contest_id,
                "disposition": "qualified_prediction",
                "locked_prediction_row_count": rows_per_contest,
                "reason_code": None,
            }
        )
        for number, index in enumerate(indices):
            rating = 1000.0 + number * 100.0
            primary_error = 10.0
            comparator_error = primary_error + improvement
            rows.append(
                {
                    "contest_id": contest_id,
                    "index": index,
                    "primary_prediction": str(rating - primary_error),
                    "comparator_prediction": str(rating - comparator_error),
                    "official_problem_present": "true",
                    "official_problem_type": "PROGRAMMING",
                    "official_rating": str(rating),
                    "outcome_status": "paired",
                    "eligible_for_analysis": "true",
                }
            )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINALIZED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "schema_version": 1,
        "kind": "confirmatory_mapping_report",
        "status": "finalized",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "census_integrity_status": "passed",
        "outcome_integrity_status": "passed",
        "census_contest_count": contest_count,
        "mapped_once_contest_count": contest_count,
        "unmapped_contest_count": 0,
        "multiply_mapped_contest_count": 0,
        "qualified_prediction_contest_count": contest_count,
        "operational_miss_contest_count": 0,
        "timely_coverage_event_count": contest_count,
        "late_coverage_event_count": 0,
        "missing_coverage_witness_count": 0,
        "prespecified_exclusion_contest_count": 0,
        "locked_prediction_contest_count": contest_count,
        "locked_prediction_row_count": len(rows),
        "paired_problem_row_count": len(rows),
        "missing_rating_row_count": 0,
        "non_programming_row_count": 0,
        "absent_from_snapshot_row_count": 0,
        "contest_dispositions": dispositions,
    }
    report_path = tmp_path / "mapping_report.json"
    _write_json(report_path, report)
    commitment = tmp_path / "commitments.jsonl"
    observation = tmp_path / "observations.jsonl"
    commitment.write_bytes(b"commitment-ledger\n")
    observation.write_bytes(b"observation-ledger\n")
    windows = protocol["fixed_snapshot_windows"]
    assert isinstance(windows, dict)
    census_time = datetime.fromisoformat(
        str(windows["cohort_census"]["anchor_utc"]).replace("Z", "+00:00")
    ) + timedelta(minutes=1)
    outcome_time = datetime.fromisoformat(
        str(windows["confirmatory_outcome"]["anchor_utc"]).replace("Z", "+00:00")
    ) + timedelta(minutes=1)
    manifest = {
        "schema_version": 1,
        "kind": "confirmatory_analysis_input",
        "status": "finalized",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "finalized_at_utc": (outcome_time + timedelta(minutes=1))
        .isoformat()
        .replace("+00:00", "Z"),
        "analysis_not_before_utc": protocol["confirmatory_analysis"][
            "earliest_execution_utc"
        ],
        "finalized_csv": {
            "path": csv_path.name,
            "sha256": _sha(csv_path),
            "row_count": len(rows),
        },
        "mapping_report": {"path": report_path.name, "sha256": _sha(report_path)},
        "commitment_ledger_sha256": _sha(commitment),
        "observation_ledger_sha256": _sha(observation),
        "cohort_census_selection": {
            "selection_manifest_sha256": "1" * 64,
            "raw_snapshot_sha256": "2" * 64,
            "selected_at_utc": census_time.isoformat().replace("+00:00", "Z"),
        },
        "confirmatory_outcome_selection": {
            "selection_manifest_sha256": "3" * 64,
            "raw_snapshot_sha256": "4" * 64,
            "selected_at_utc": outcome_time.isoformat().replace("+00:00", "Z"),
        },
    }
    manifest_path = tmp_path / "confirmatory_analysis_input.json"
    _write_json(manifest_path, manifest)
    return {
        "protocol_path": protocol_path,
        "protocol": protocol,
        "csv_path": csv_path,
        "rows": rows,
        "report_path": report_path,
        "report": report,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "commitment": commitment,
        "observation": observation,
    }


def _run(package: dict[str, object], output: Path) -> dict[str, object]:
    return run_confirmatory_analysis(
        package["protocol_path"],
        package["manifest_path"],
        output,
        commitment_ledger_path=package["commitment"],
        observation_ledger_path=package["observation"],
        clock=lambda: datetime(2027, 3, 5, 1, 0, tzinfo=timezone.utc),
        event_chain_validator=lambda *args: None,
    )


def test_cluster_bootstrap_matches_frozen_golden_fixture() -> None:
    clusters = {str(index): [index + index * index / 1000] for index in range(30)}
    replicates, draws = _cluster_bootstrap(
        clusters, resamples=10_000, seed=20_260_710
    )
    assert draws[0, :15].tolist() == [
        23,
        17,
        20,
        27,
        15,
        23,
        22,
        2,
        18,
        6,
        12,
        9,
        18,
        14,
        22,
    ]
    assert [value.hex() for value in replicates[:5]] == [
        "0x1.100a1a7cca9d9p+4",
        "0x1.9e9c779a6b50bp+3",
        "0x1.1749ba5e353f7p+4",
        "0x1.bda8e448a2bf5p+3",
        "0x1.b8f72015d867ep+3",
    ]
    interval = np.quantile(replicates, [0.025, 0.975], method="linear")
    assert [value.hex() for value in interval] == [
        "0x1.741f08b5f8bc7p+3",
        "0x1.1f956ced218f6p+4",
    ]
    baseline = np.arange(10_000, dtype=np.float64)
    assert np.quantile(
        baseline, [0.025, 0.975], method="linear"
    ).tolist() == pytest.approx([249.975, 9749.025])


def test_complete_analysis_uses_strict_positive_interval(tmp_path: Path) -> None:
    package = _analysis_package(tmp_path)
    result = _run(package, tmp_path / "result.json")
    assert result["eligible_contest_count"] == 30
    assert result["eligible_problem_row_count"] == 210
    assert result["primary"]["point_estimate"] == pytest.approx(40.0)
    assert result["primary"]["confidence_interval"] == pytest.approx([40.0, 40.0])
    assert result["primary"]["confirmatory_success"] is True
    assert result["secondary"]["within_100_accuracy"] == {
        "primary": 1.0,
        "comparator": 1.0,
    }


def test_zero_lower_bound_is_not_success(tmp_path: Path) -> None:
    package = _analysis_package(tmp_path, improvement=0.0)
    result = _run(package, tmp_path / "result.json")
    assert result["primary"]["confidence_interval"] == [0.0, 0.0]
    assert result["primary"]["confirmatory_success"] is False


@pytest.mark.parametrize(
    ("contests", "rows_per_contest"), [(29, 7), (30, 6)]
)
def test_minimum_thresholds_fail_before_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contests: int,
    rows_per_contest: int,
) -> None:
    package = _analysis_package(
        tmp_path, contest_count=contests, rows_per_contest=rows_per_contest
    )
    monkeypatch.setattr(
        prospective_analysis,
        "_calculate_results",
        lambda *args, **kwargs: pytest.fail("metrics must remain blocked"),
    )
    with pytest.raises(ProspectiveAnalysisError, match="thresholds"):
        _run(package, tmp_path / "result.json")


def test_time_gate_runs_before_reading_outcomes(tmp_path: Path) -> None:
    protocol_path, _ = _frozen_protocol(tmp_path)
    with pytest.raises(ProspectiveAnalysisError, match="blocked"):
        run_confirmatory_analysis(
            protocol_path,
            tmp_path / "missing-manifest.json",
            tmp_path / "result.json",
            clock=lambda: datetime(2027, 3, 5, 0, 4, 58, tzinfo=timezone.utc),
        )


def test_mapping_integrity_failure_blocks_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _analysis_package(tmp_path)
    report = package["report"]
    assert isinstance(report, dict)
    report["census_integrity_status"] = "failed"
    _write_json(package["report_path"], report)
    manifest = package["manifest"]
    assert isinstance(manifest, dict)
    manifest["mapping_report"]["sha256"] = _sha(package["report_path"])
    _write_json(package["manifest_path"], manifest)
    monkeypatch.setattr(
        prospective_analysis,
        "_calculate_results",
        lambda *args, **kwargs: pytest.fail("metrics must remain blocked"),
    )
    with pytest.raises(ProspectiveAnalysisError, match="integrity"):
        _run(package, tmp_path / "result.json")


@pytest.mark.parametrize("mutation", ["csv", "report", "ledger"])
def test_bound_artifact_tampering_is_rejected(tmp_path: Path, mutation: str) -> None:
    package = _analysis_package(tmp_path)
    if mutation == "csv":
        package["csv_path"].write_bytes(package["csv_path"].read_bytes() + b"\n")
    elif mutation == "report":
        package["report_path"].write_bytes(package["report_path"].read_bytes() + b" ")
    else:
        package["commitment"].write_bytes(b"changed\n")
    with pytest.raises(ProspectiveAnalysisError, match="SHA|hash"):
        _run(package, tmp_path / "result.json")


def test_invalid_boolean_and_duplicate_key_are_rejected(tmp_path: Path) -> None:
    package = _analysis_package(tmp_path)
    rows = package["rows"]
    assert isinstance(rows, list)
    rows[0]["eligible_for_analysis"] = "True"
    rows.append(dict(rows[1]))
    with package["csv_path"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINALIZED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = package["manifest"]
    assert isinstance(manifest, dict)
    manifest["finalized_csv"]["sha256"] = _sha(package["csv_path"])
    manifest["finalized_csv"]["row_count"] = len(rows)
    _write_json(package["manifest_path"], manifest)
    with pytest.raises(ProspectiveAnalysisError, match="lowercase|duplicate"):
        _run(package, tmp_path / "result.json")


def test_cli_exposes_no_analysis_freedom(tmp_path: Path) -> None:
    parser = prospective_analysis._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--seed", "1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--resamples", "1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--now", "2099-01-01T00:00:00Z"])
