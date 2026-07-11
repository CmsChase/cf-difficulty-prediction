"""Cross-file invariants for the prospective v2 research protocol."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "prospective_protocol_v2.json"
WITNESS_WORKFLOW_PATH = (
    PROJECT_ROOT / ".github" / "workflows" / "prospective-witness.yml"
)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo == timezone.utc
    return parsed


def _lf_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return hashlib.sha256(
        text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    ).hexdigest()


def _load_protocol() -> dict[str, object]:
    with PROTOCOL_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_snapshot_windows_have_exact_non_overlapping_schedules() -> None:
    """The one-second boundary and 48-slot policy cannot drift in prose."""

    protocol = _load_protocol()
    cohort = protocol["cohort"]
    windows = protocol["fixed_snapshot_windows"]
    assert isinstance(cohort, dict)
    assert isinstance(windows, dict)

    eligibility_end = _parse_utc(str(cohort["eligibility_end_utc"]))
    expected_offsets = {
        "cohort_census": 300,
        "confirmatory_outcome": 259_500,
    }
    for kind, offset in expected_offsets.items():
        window = windows[kind]
        assert isinstance(window, dict)
        anchor = _parse_utc(str(window["anchor_utc"]))
        deadline = _parse_utc(str(window["deadline_utc"]))
        interval = int(window["retry_interval_seconds"])
        maximum = int(window["maximum_attempts"])

        assert anchor == eligibility_end + timedelta(seconds=offset)
        assert deadline == anchor + timedelta(
            seconds=int(window["window_duration_seconds"])
        )
        assert window["request_window"] == "half_open"
        assert window["missed_scheduled_attempt_policy"] == "invalidate_window"
        assert maximum == 48
        assert interval == 1_800
        assert anchor + (maximum - 1) * timedelta(seconds=interval) < deadline
        assert anchor + maximum * timedelta(seconds=interval) == deadline


def test_external_witness_configuration_binds_exact_workflow() -> None:
    """The public timestamp contract identifies one repository and workflow."""

    external = _load_protocol()["external_timestamp"]
    assert isinstance(external, dict)
    assert external == {
        "provider": "github_actions_workflow_run",
        "repository_id": 1_280_990_637,
        "repository_full_name": "CmsChase/cf-difficulty-prediction",
        "default_branch": "main",
        "workflow_name": "Prospective commitment witness",
        "workflow_path": ".github/workflows/prospective-witness.yml",
        "workflow_file_sha256": _lf_sha256(WITNESS_WORKFLOW_PATH),
        "trigger_event": "push",
        "deadline_field": "created_at",
        "required_status": "completed",
        "required_conclusion": "success",
        "required_run_attempt": 1,
        "target_commit_must_be_default_branch_ancestor": True,
        "commit_message_skip_directives_prohibited": True,
        "local_git_timestamps_are_external_evidence": False,
        "witness_record_policy": external["witness_record_policy"],
    }
    assert "GitHub-created timestamp" in str(external["witness_record_policy"])


def test_witness_workflow_has_no_conditional_trigger_or_write_token() -> None:
    """Every main push gets a read-only run whose dependencies are SHA pinned."""

    workflow = WITNESS_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "branches:\n      - main" in workflow
    assert "paths:" not in workflow
    assert "paths-ignore:" not in workflow
    assert "concurrency:" not in workflow
    assert "contents: read" in workflow
    assert "actions: read" in workflow
    assert "contents: write" not in workflow
    assert (
        "actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10"
        in workflow
    )
    assert (
        "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1"
        in workflow
    )


def test_cohort_mapping_has_no_discretionary_exclusion_bucket() -> None:
    """Every census contest must become a commitment or an explicit miss."""

    cohort = _load_protocol()["cohort"]
    assert isinstance(cohort, dict)
    assert cohort["contest_exclusion_codes"] == []
    assert "no discretionary contest-exclusion" in str(
        cohort["cohort_census_rule"]
    )


def test_confirmatory_analysis_contract_is_fully_numeric() -> None:
    """RNG, threshold, interval, and success boundaries are machine locked."""

    analysis = _load_protocol()["confirmatory_analysis"]
    assert isinstance(analysis, dict)
    bootstrap = analysis["bootstrap"]
    success = analysis["success_rule"]
    assert isinstance(bootstrap, dict)
    assert isinstance(success, dict)

    assert analysis["earliest_execution_utc"] == "2027-03-05T00:04:59Z"
    assert analysis["minimum_distinct_paired_contests"] == 30
    assert analysis["minimum_paired_problem_rows"] == 200
    assert bootstrap["resamples"] == 10_000
    assert bootstrap["seed"] == 20_260_710
    assert bootstrap["bit_generator"] == "PCG64"
    assert bootstrap["quantile_probabilities"] == [0.025, 0.975]
    assert bootstrap["quantile_method"] == "linear"
    assert success == {
        "criterion": "lower_confidence_bound_strictly_greater_than_zero",
        "boundary_value": 0,
        "equality_is_success": False,
    }
    implementation = analysis["implementation_freeze"]
    assert implementation["required_source_sha256_keys"] == [
        "prospective_model",
        "prospective_input",
        "prospective_ledger",
        "prospective_snapshot",
        "prospective_cohort",
        "prospective_analysis",
        "statement_features",
        "witness_workflow",
        "tests_workflow",
        "test_prospective_protocol",
        "test_prospective_input",
        "test_prospective_model",
        "test_prospective_ledger",
        "test_prospective_snapshot",
        "test_prospective_cohort",
        "test_prospective_analysis",
    ]
    golden = implementation["golden_fixture"]
    assert golden["first_draw_first_15"][:5] == [23, 17, 20, 27, 15]
    assert golden["confidence_interval_hex"] == [
        "0x1.741f08b5f8bc7p+3",
        "0x1.1f956ced218f6p+4",
    ]


def test_draft_cannot_be_mistaken_for_an_open_cohort() -> None:
    """Operational code must continue to reject the current public draft."""

    protocol = _load_protocol()
    assert protocol["status"] == "draft"
    assert "protocol_frozen_at_utc" not in protocol
