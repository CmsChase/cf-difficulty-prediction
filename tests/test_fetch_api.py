"""Unit tests for exact-byte Codeforces raw acquisition."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import fetch_api


class FakeResponse:
    """Minimal requests response containing exact mocked wire bytes."""

    def __init__(
        self,
        raw_bytes: bytes,
        *,
        url: str,
        status_code: int = 200,
    ) -> None:
        self.content = raw_bytes
        self.url = url
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """Raise an HTTP error for non-success status codes."""
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            raise requests.HTTPError(
                f"HTTP {self.status_code}",
                response=response,
            )

    def json(self) -> object:
        """Decode the mocked bytes for reusable helper tests."""
        return json.loads(self.content)


@pytest.fixture
def raw_responses() -> tuple[bytes, bytes]:
    """Return deliberately compact raw JSON bytes for both endpoints."""
    problemset_bytes = (
        b'{"status":"OK","result":{"problems":[{"contestId":1},'
        b'{"contestId":2}],"problemStatistics":[{"solvedCount":10}]}}'
    )
    contest_bytes = (
        b'{"status":"OK","comment":"mock","result":[{"id":1},{"id":2}]}'
    )
    return problemset_bytes, contest_bytes


def _mock_session_get(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[FakeResponse],
) -> None:
    """Patch Session.get to return responses in request order."""
    pending = iter(responses)

    def fake_get(
        _session: requests.Session,
        _url: str,
        *,
        params: dict[str, object],
        timeout: int,
    ) -> FakeResponse:
        assert params
        assert timeout == 30
        return next(pending)

    monkeypatch.setattr(requests.Session, "get", fake_get)


def test_acquisition_creates_snapshot_manifest_latest_and_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_responses: tuple[bytes, bytes],
) -> None:
    """One run writes exact raw files, provenance, latest, and a JSON log."""
    problemset_bytes, contest_bytes = raw_responses
    timestamp = datetime(2026, 6, 25, 4, 15, tzinfo=timezone.utc)
    monkeypatch.setattr(fetch_api, "utc_now", lambda: timestamp)
    responses = [
        FakeResponse(
            problemset_bytes,
            url=(
                "https://codeforces.com/api/problemset.problems?lang=en"
            ),
        ),
        FakeResponse(
            contest_bytes,
            url=(
                "https://codeforces.com/api/contest.list?"
                "gym=false&lang=en"
            ),
        ),
    ]
    _mock_session_get(monkeypatch, responses)
    sleeps: list[float] = []

    output_root = tmp_path / "raw" / "codeforces"
    latest_dir = output_root / "latest"
    log_path = tmp_path / "outputs" / "logs" / "fetch.log"
    snapshot_dir = fetch_api.acquire_snapshot(
        output_root,
        latest_dir=latest_dir,
        log_path=log_path,
        sleep_fn=sleeps.append,
    )

    assert snapshot_dir.name == "20260625T041500Z"
    assert sorted(path.name for path in snapshot_dir.iterdir()) == [
        "contest.list.json",
        "manifest.json",
        "problemset.problems.json",
    ]
    assert (
        snapshot_dir / "problemset.problems.json"
    ).read_bytes() == problemset_bytes
    assert (snapshot_dir / "contest.list.json").read_bytes() == contest_bytes
    assert sleeps == [2.1]

    manifest = json.loads(
        (snapshot_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["snapshot_id"] == "20260625T041500Z"
    assert set(manifest["resources"]) == {
        "problemset.problems",
        "contest.list",
    }
    required_keys = {
        "endpoint",
        "request_url",
        "query_params",
        "fetched_at_utc",
        "http_status",
        "api_status",
        "sha256",
        "bytes",
        "output_path",
        "top_level_keys",
        "record_counts",
    }
    for resource in manifest["resources"].values():
        assert required_keys <= set(resource)

    problemset_manifest = manifest["resources"]["problemset.problems"]
    assert problemset_manifest["sha256"] == hashlib.sha256(
        problemset_bytes
    ).hexdigest()
    assert problemset_manifest["bytes"] == len(problemset_bytes)
    assert problemset_manifest["record_counts"] == {
        "problems": 2,
        "problem_statistics": 1,
    }
    assert manifest["resources"]["contest.list"]["record_counts"] == {
        "contests": 2
    }

    for filename in (
        "problemset.problems.json",
        "contest.list.json",
        "manifest.json",
    ):
        assert (latest_dir / filename).read_bytes() == (
            snapshot_dir / filename
        ).read_bytes()
    assert log_path.is_file()
    log_records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(record.get("event") == "snapshot_completed" for record in log_records)


def test_fixed_response_sha256_is_stable(
    raw_responses: tuple[bytes, bytes],
) -> None:
    """SHA-256 is calculated directly from fixed raw bytes."""
    problemset_bytes, _ = raw_responses
    expected = hashlib.sha256(problemset_bytes).hexdigest()
    assert fetch_api.compute_sha256(problemset_bytes) == expected
    assert fetch_api.compute_sha256(problemset_bytes) == expected


def test_malformed_json_raises_clear_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed response bytes are rejected before any snapshot is written."""
    response = FakeResponse(
        b'{"status":"OK","result":',
        url="https://codeforces.com/api/problemset.problems?lang=en",
    )
    _mock_session_get(monkeypatch, [response])
    spec = fetch_api.ResourceSpec(
        endpoint="problemset.problems",
        query_params={"lang": "en"},
        output_filename="problemset.problems.json",
    )

    with requests.Session() as session:
        with pytest.raises(fetch_api.CodeforcesAPIError, match="malformed JSON"):
            fetch_api.fetch_raw_resource(session, spec, timeout=30)


def test_problemset_count_extraction() -> None:
    """Problem and statistics counts are independently recorded."""
    payload: dict[str, Any] = {
        "status": "OK",
        "result": {
            "problems": [{}, {}, {}],
            "problemStatistics": [{}, {}],
        },
    }
    assert fetch_api.extract_record_counts(
        "problemset.problems",
        payload,
    ) == {
        "problems": 3,
        "problem_statistics": 2,
    }


def test_contest_count_extraction() -> None:
    """Contest count comes from the contest.list result array."""
    payload: dict[str, Any] = {
        "status": "OK",
        "result": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}],
    }
    assert fetch_api.extract_record_counts("contest.list", payload) == {
        "contests": 4
    }


def test_missing_status_is_rejected() -> None:
    """A payload without the mandatory API status fails loudly."""
    with pytest.raises(fetch_api.CodeforcesAPIError, match="status"):
        fetch_api._validate_api_payload({"result": []}, "contest.list")
