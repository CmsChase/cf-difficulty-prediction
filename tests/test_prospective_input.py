"""Tests for label-isolated prospective T0 input capture."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import prospective_input
from cf_diff.prospective_input import (
    ProspectiveInputError,
    capture_prospective_input,
)
from cf_diff.prospective_model import (
    PREDICTION_COLUMNS,
    freeze_prospective_model,
    predict_prospective,
)
from cf_diff.statement_features import STATEMENT_FEATURE_COLUMNS, FetchResult


DRAFT_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "prospective_protocol_v2.json"
)
CONTEST_START = datetime(2026, 8, 15, 1, 0, tzinfo=timezone.utc)
CAPTURE_TIME = datetime(2026, 8, 15, 1, 1, tzinfo=timezone.utc)
SYNTHETIC_HTML = """
<html><body>
  <div class="problem-statement">
    <div class="time-limit">time limit per test 1.5 seconds</div>
    <div class="memory-limit">memory limit per test 256 megabytes</div>
    <div class="header"><div class="title">A. Graph Walk</div></div>
    <p>Given a tree with n vertices, answer q queries.</p>
    <div class="input-specification"><p>The first line contains n and q.</p></div>
    <div class="output-specification"><p>Print the shortest path.</p></div>
    <div class="sample-tests">
      <div class="input"><pre>3 1</pre></div>
      <div class="output"><pre>2</pre></div>
    </div>
  </div>
</body></html>
"""


def _frozen_protocol(tmp_path: Path) -> Path:
    protocol = json.loads(DRAFT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["status"] = "frozen"
    protocol["protocol_frozen_at_utc"] = "2026-08-01T00:00:00Z"
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _targets(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "inputs" / "3000_t0_features.csv",
        tmp_path / "inputs" / "3000_t0_features.capture.json",
        tmp_path / "raw" / "3000",
    )


def _successful_fetcher(calls: list[str]):
    def fetcher(**kwargs: object) -> FetchResult:
        url = str(kwargs["url"])
        raw_path = Path(kwargs["raw_path"])
        calls.append(url)
        raw_path.write_text(SYNTHETIC_HTML, encoding="utf-8", newline="\n")
        return FetchResult(
            status="fetched",
            cache_path=raw_path,
            html_text=SYNTHETIC_HTML,
            http_status=200,
        )

    return fetcher


def test_capture_writes_exact_schema_and_audit_sidecar(tmp_path: Path) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    output_path, sidecar_path, raw_dir = _targets(tmp_path)
    calls: list[str] = []

    paths = capture_prospective_input(
        protocol_path=protocol_path,
        contest_id="03000",
        indices=["A", "01"],
        contest_start_utc=CONTEST_START,
        output_path=output_path,
        sidecar_path=sidecar_path,
        raw_dir=raw_dir,
        sleep_seconds=0,
        clock=lambda: CAPTURE_TIME,
        fetcher=_successful_fetcher(calls),
    )

    assert paths == {
        "input": output_path,
        "sidecar": sidecar_path,
        "raw_dir": raw_dir,
    }
    frame = pd.read_csv(
        output_path,
        dtype={"contest_id": "string", "index": "string"},
    )
    assert list(frame.columns) == [
        "contest_id",
        "index",
        *STATEMENT_FEATURE_COLUMNS,
    ]
    assert frame["contest_id"].tolist() == ["3000", "3000"]
    assert frame["index"].tolist() == ["A", "01"]
    assert "url" not in frame.columns
    assert "fetch_status" not in frame.columns
    assert calls == [
        "https://codeforces.com/problemset/problem/3000/A",
        "https://codeforces.com/problemset/problem/3000/01",
    ]

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["status"] == "complete"
    assert sidecar["requested_indices"] == ["A", "01"]
    assert sidecar["request_policy"]["metadata_api_used"] is False
    assert sidecar["output"]["columns"] == list(frame.columns)
    assert sidecar["output"]["row_count"] == 2
    assert sidecar["output"]["sha256"] == hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()
    assert [problem["index"] for problem in sidecar["problems"]] == ["A", "01"]
    assert all(problem["raw_html_sha256"] for problem in sidecar["problems"])


@pytest.mark.parametrize(
    ("contest_id", "indices", "message"),
    [
        ("0", ["A"], "positive integer"),
        ("abc", ["A"], "positive integer"),
        ("3000", [], "At least one"),
        ("3000", ["A", "a"], "unique"),
        ("3000", ["A", "../B"], "Invalid"),
    ],
)
def test_invalid_keys_fail_before_network_or_files(
    tmp_path: Path,
    contest_id: str,
    indices: list[str],
    message: str,
) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    output_path, sidecar_path, raw_dir = _targets(tmp_path)
    calls: list[str] = []

    with pytest.raises(ProspectiveInputError, match=message):
        capture_prospective_input(
            protocol_path=protocol_path,
            contest_id=contest_id,
            indices=indices,
            contest_start_utc=CONTEST_START,
            output_path=output_path,
            sidecar_path=sidecar_path,
            raw_dir=raw_dir,
            sleep_seconds=0,
            clock=lambda: CAPTURE_TIME,
            fetcher=_successful_fetcher(calls),
        )

    assert calls == []
    assert not output_path.exists()
    assert not sidecar_path.exists()
    assert not raw_dir.exists()


@pytest.mark.parametrize("existing_target", ["output", "sidecar", "raw"])
def test_capture_never_overwrites_existing_targets(
    tmp_path: Path,
    existing_target: str,
) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    output_path, sidecar_path, raw_dir = _targets(tmp_path)
    targets = {
        "output": output_path,
        "sidecar": sidecar_path,
        "raw": raw_dir,
    }
    target = targets[existing_target]
    if target == raw_dir:
        target.mkdir(parents=True)
        sentinel = target / "sentinel.txt"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        sentinel = target
    sentinel.write_text("keep", encoding="utf-8")
    calls: list[str] = []

    with pytest.raises(ProspectiveInputError, match="will not be overwritten"):
        capture_prospective_input(
            protocol_path=protocol_path,
            contest_id="3000",
            indices=["A"],
            contest_start_utc=CONTEST_START,
            output_path=output_path,
            sidecar_path=sidecar_path,
            raw_dir=raw_dir,
            sleep_seconds=0,
            clock=lambda: CAPTURE_TIME,
            fetcher=_successful_fetcher(calls),
        )

    assert calls == []
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_draft_protocol_cannot_capture(tmp_path: Path) -> None:
    output_path, sidecar_path, raw_dir = _targets(tmp_path)
    calls: list[str] = []

    with pytest.raises(ProspectiveInputError, match="status must be 'frozen'"):
        capture_prospective_input(
            protocol_path=DRAFT_PROTOCOL_PATH,
            contest_id="3000",
            indices=["A"],
            contest_start_utc=CONTEST_START,
            output_path=output_path,
            sidecar_path=sidecar_path,
            raw_dir=raw_dir,
            sleep_seconds=0,
            clock=lambda: CAPTURE_TIME,
            fetcher=_successful_fetcher(calls),
        )

    assert calls == []
    assert not raw_dir.exists()


@pytest.mark.parametrize(
    ("contest_start", "clock_time", "message"),
    [
        (
            datetime(2026, 8, 14, 23, 59, tzinfo=timezone.utc),
            CAPTURE_TIME,
            "outside",
        ),
        (
            CONTEST_START,
            datetime(2026, 8, 15, 0, 59, tzinfo=timezone.utc),
            "before",
        ),
        (
            CONTEST_START,
            datetime(2026, 8, 15, 1, 30, 1, tzinfo=timezone.utc),
            "deadline",
        ),
    ],
)
def test_invalid_capture_window_fails_before_network(
    tmp_path: Path,
    contest_start: datetime,
    clock_time: datetime,
    message: str,
) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    output_path, sidecar_path, raw_dir = _targets(tmp_path)
    calls: list[str] = []

    with pytest.raises(ProspectiveInputError, match=message):
        capture_prospective_input(
            protocol_path=protocol_path,
            contest_id="3000",
            indices=["A"],
            contest_start_utc=contest_start,
            output_path=output_path,
            sidecar_path=sidecar_path,
            raw_dir=raw_dir,
            sleep_seconds=0,
            clock=lambda: clock_time,
            fetcher=_successful_fetcher(calls),
        )

    assert calls == []
    assert not raw_dir.exists()


def test_feature_allowlist_drift_fails_before_network(tmp_path: Path) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    primary = protocol["model_bundle"]["primary_feature_columns"]
    primary[-1], primary[-2] = primary[-2], primary[-1]
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    output_path, sidecar_path, raw_dir = _targets(tmp_path)
    calls: list[str] = []

    with pytest.raises(ProspectiveInputError, match="exactly match"):
        capture_prospective_input(
            protocol_path=protocol_path,
            contest_id="3000",
            indices=["A"],
            contest_start_utc=CONTEST_START,
            output_path=output_path,
            sidecar_path=sidecar_path,
            raw_dir=raw_dir,
            sleep_seconds=0,
            clock=lambda: CAPTURE_TIME,
            fetcher=_successful_fetcher(calls),
        )

    assert calls == []
    assert not raw_dir.exists()


def test_partial_fetch_failure_writes_failure_sidecar_not_csv(
    tmp_path: Path,
) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    output_path, sidecar_path, raw_dir = _targets(tmp_path)
    calls: list[str] = []

    def fetcher(**kwargs: object) -> FetchResult:
        url = str(kwargs["url"])
        raw_path = Path(kwargs["raw_path"])
        calls.append(url)
        if url.endswith("/B"):
            return FetchResult(
                status="failed",
                cache_path=raw_path,
                html_text=None,
                error="network unavailable",
            )
        raw_path.write_text(SYNTHETIC_HTML, encoding="utf-8")
        return FetchResult(
            status="fetched",
            cache_path=raw_path,
            html_text=SYNTHETIC_HTML,
            http_status=200,
        )

    with pytest.raises(ProspectiveInputError, match="no model input"):
        capture_prospective_input(
            protocol_path=protocol_path,
            contest_id="3000",
            indices=["A", "B", "C"],
            contest_start_utc=CONTEST_START,
            output_path=output_path,
            sidecar_path=sidecar_path,
            raw_dir=raw_dir,
            sleep_seconds=0,
            clock=lambda: CAPTURE_TIME,
            fetcher=fetcher,
        )

    assert not output_path.exists()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["status"] == "failed"
    assert sidecar["output"] is None
    assert [problem["index"] for problem in sidecar["problems"]] == [
        "A",
        "B",
        "C",
    ]
    assert sidecar["problems"][1]["fetch_status"] == "failed"
    assert sidecar["problems"][2]["fetch_status"] == "not_attempted"
    assert len(calls) == 2


def test_non_statement_page_is_an_audited_whole_contest_failure(
    tmp_path: Path,
) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    output_path, sidecar_path, raw_dir = _targets(tmp_path)

    def fetcher(**kwargs: object) -> FetchResult:
        raw_path = Path(kwargs["raw_path"])
        challenge = "<html><body>verification required</body></html>"
        raw_path.write_text(challenge, encoding="utf-8")
        return FetchResult(
            status="fetched",
            cache_path=raw_path,
            html_text=challenge,
            http_status=200,
        )

    with pytest.raises(ProspectiveInputError, match="no model input"):
        capture_prospective_input(
            protocol_path=protocol_path,
            contest_id="3000",
            indices=["A", "B"],
            contest_start_utc=CONTEST_START,
            output_path=output_path,
            sidecar_path=sidecar_path,
            raw_dir=raw_dir,
            sleep_seconds=0,
            clock=lambda: CAPTURE_TIME,
            fetcher=fetcher,
        )

    assert not output_path.exists()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["status"] == "failed"
    assert sidecar["problems"][0]["parse_status"] == "missing_statement"
    assert sidecar["problems"][1]["fetch_status"] == "not_attempted"


def test_deadline_crossing_removes_csv_and_records_failure(tmp_path: Path) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    output_path, sidecar_path, raw_dir = _targets(tmp_path)
    current = {"value": CAPTURE_TIME}

    def clock() -> datetime:
        return current["value"]

    def fetcher(**kwargs: object) -> FetchResult:
        raw_path = Path(kwargs["raw_path"])
        raw_path.write_text(SYNTHETIC_HTML, encoding="utf-8")
        current["value"] = datetime(
            2026,
            8,
            15,
            1,
            30,
            1,
            tzinfo=timezone.utc,
        )
        return FetchResult(
            status="fetched",
            cache_path=raw_path,
            html_text=SYNTHETIC_HTML,
            http_status=200,
        )

    with pytest.raises(ProspectiveInputError, match="no model input"):
        capture_prospective_input(
            protocol_path=protocol_path,
            contest_id="3000",
            indices=["A"],
            contest_start_utc=CONTEST_START,
            output_path=output_path,
            sidecar_path=sidecar_path,
            raw_dir=raw_dir,
            sleep_seconds=0,
            clock=clock,
            fetcher=fetcher,
        )

    assert not output_path.exists()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["status"] == "failed"
    assert "deadline" in sidecar["error"]


def test_cli_smoke_uses_the_same_guarded_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path = _frozen_protocol(tmp_path)
    output_path, sidecar_path, raw_dir = _targets(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        prospective_input,
        "_fetch_fresh_problem_page",
        _successful_fetcher(calls),
    )
    monkeypatch.setattr(prospective_input, "_utc_now", lambda: CAPTURE_TIME)

    exit_code = prospective_input.main(
        [
            "--protocol",
            str(protocol_path),
            "--contest-id",
            "3000",
            "--indices",
            "A",
            "--contest-start-utc",
            "2026-08-15T01:00:00Z",
            "--output",
            str(output_path),
            "--sidecar",
            str(sidecar_path),
            "--raw-dir",
            str(raw_dir),
            "--sleep-seconds",
            "0",
        ]
    )

    assert exit_code == 0
    assert output_path.is_file()
    assert sidecar_path.is_file()


def test_synthetic_capture_freeze_predict_dry_run(tmp_path: Path) -> None:
    """Exercise the complete pre-ledger chain without a metadata API."""
    protocol_path = _frozen_protocol(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    primary = protocol["model_bundle"]["primary_feature_columns"]
    cutoff_seconds = int(
        datetime(2026, 8, 15, tzinfo=timezone.utc).timestamp()
    )
    model_table = pd.DataFrame(
        {
            "contest_id": [100, 101, 102],
            "index": ["A", "B", "C"],
            "rating": [900.0, 1400.0, 1900.0],
            "start_time_seconds": [
                cutoff_seconds - 300,
                cutoff_seconds - 200,
                cutoff_seconds - 100,
            ],
        }
    )
    statements: dict[str, object] = {
        "contest_id": [100, 101, 102],
        "index": ["A", "B", "C"],
    }
    for number, column in enumerate(primary, start=1):
        if column in {"index_rank", "index_number"}:
            continue
        statements[column] = [number, number + 1, number + 2]
    model_table_path = tmp_path / "historical-model.parquet"
    statements_path = tmp_path / "historical-statements.parquet"
    model_table.to_parquet(model_table_path, index=False)
    pd.DataFrame(statements).to_parquet(statements_path, index=False)
    model_path = tmp_path / "bundle.json"
    manifest_path = tmp_path / "manifest.json"
    freeze_prospective_model(
        protocol_path=protocol_path,
        model_table_path=model_table_path,
        statement_features_path=statements_path,
        model_path=model_path,
        manifest_path=manifest_path,
        requirements_path=PROJECT_ROOT / "requirements.txt",
        source_commit="0123456789abcdef0123456789abcdef01234567",
        frozen_at_utc="2026-08-02T00:00:00Z",
    )

    output_path, sidecar_path, raw_dir = _targets(tmp_path)
    capture_prospective_input(
        protocol_path=protocol_path,
        contest_id="3000",
        indices=["A", "B"],
        contest_start_utc=CONTEST_START,
        output_path=output_path,
        sidecar_path=sidecar_path,
        raw_dir=raw_dir,
        sleep_seconds=0,
        clock=lambda: CAPTURE_TIME,
        fetcher=_successful_fetcher([]),
    )
    prediction_path = tmp_path / "predictions" / "3000.csv"
    predict_prospective(
        protocol_path=protocol_path,
        model_path=model_path,
        manifest_path=manifest_path,
        input_path=output_path,
        capture_sidecar_path=sidecar_path,
        output_path=prediction_path,
        contest_start_utc=CONTEST_START,
        prediction_created_at_utc="2026-08-15T01:02:00Z",
    )

    predictions = pd.read_csv(prediction_path)
    assert tuple(predictions.columns) == PREDICTION_COLUMNS
    assert predictions["index"].tolist() == ["A", "B"]
    assert predictions["capture_sidecar_sha256"].nunique() == 1
