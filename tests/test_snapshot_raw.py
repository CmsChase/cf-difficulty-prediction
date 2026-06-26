"""Unit tests for immutable Codeforces raw snapshot creation."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import snapshot_raw


PROBLEMSET_PAYLOAD = {
    "status": "OK",
    "comment": "full raw problemset response",
    "result": {
        "problems": [{"contestId": 1, "index": "A", "name": "Watermelon"}],
        "problemStatistics": [{"contestId": 1, "index": "A", "solvedCount": 42}],
    },
}
CONTEST_PAYLOAD = {
    "status": "OK",
    "comment": "full raw contest response",
    "result": [{"id": 1, "name": "Codeforces Beta Round 1"}],
}


def _load_json(path: Path) -> object:
    """Read one UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def mocked_fetches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace both Codeforces API calls with fixed raw response objects."""
    monkeypatch.setattr(
        snapshot_raw,
        "fetch_problemset_problems",
        lambda lang="en": PROBLEMSET_PAYLOAD,
    )
    monkeypatch.setattr(
        snapshot_raw,
        "fetch_contest_list",
        lambda gym=False, lang="en": CONTEST_PAYLOAD,
    )


def test_cli_creates_snapshots_provenance_and_manifest(
    mocked_fetches: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI creates five files with verifiable hashes and manifest paths."""
    del mocked_fetches
    run_time = datetime(2026, 6, 25, 2, 15, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(snapshot_raw, "utc_now", lambda: run_time)

    output_root = tmp_path / "codeforces"
    exit_code = snapshot_raw.main(
        [
            "--output-root",
            str(output_root),
            "--lang",
            "en",
            "--seed",
            "42",
        ]
    )

    assert exit_code == 0
    timestamp_token = "20260625T021530Z"
    problemset_path = (
        output_root
        / "snapshots"
        / f"problemset_problems_{timestamp_token}.json"
    )
    contest_path = (
        output_root / "snapshots" / f"contest_list_{timestamp_token}.json"
    )
    problemset_meta_path = (
        output_root
        / "provenance"
        / f"problemset_problems_{timestamp_token}.meta.json"
    )
    contest_meta_path = (
        output_root
        / "provenance"
        / f"contest_list_{timestamp_token}.meta.json"
    )
    manifest_path = output_root / "manifest_latest.json"

    assert sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    ) == [
        "manifest_latest.json",
        f"provenance/contest_list_{timestamp_token}.meta.json",
        f"provenance/problemset_problems_{timestamp_token}.meta.json",
        f"snapshots/contest_list_{timestamp_token}.json",
        f"snapshots/problemset_problems_{timestamp_token}.json",
    ]
    assert _load_json(problemset_path) == PROBLEMSET_PAYLOAD
    assert _load_json(contest_path) == CONTEST_PAYLOAD

    for snapshot_path, metadata_path in (
        (problemset_path, problemset_meta_path),
        (contest_path, contest_meta_path),
    ):
        snapshot_bytes = snapshot_path.read_bytes()
        metadata = _load_json(metadata_path)
        assert isinstance(metadata, dict)
        assert metadata["sha256"] == hashlib.sha256(snapshot_bytes).hexdigest()
        assert metadata["byte_count"] == len(snapshot_bytes)
        assert metadata["api_status"] == "OK"
        assert metadata["seed"] == 42
        assert metadata["fetched_at_utc"] == "2026-06-25T02:15:30+00:00"

    manifest = _load_json(manifest_path)
    assert manifest == {
        "created_at_utc": "2026-06-25T02:15:30+00:00",
        "seed": 42,
        "entries": {
            "problemset_problems": {
                "snapshot_path": (
                    f"snapshots/problemset_problems_{timestamp_token}.json"
                ),
                "provenance_path": (
                    "provenance/"
                    f"problemset_problems_{timestamp_token}.meta.json"
                ),
            },
            "contest_list": {
                "snapshot_path": f"snapshots/contest_list_{timestamp_token}.json",
                "provenance_path": (
                    f"provenance/contest_list_{timestamp_token}.meta.json"
                ),
            },
        },
    }
    assert "Created Codeforces snapshot set:" in capsys.readouterr().out


def test_second_timestamp_preserves_first_snapshot_set_and_updates_manifest(
    mocked_fetches: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A later run appends timestamped files and repoints the latest manifest."""
    del mocked_fetches
    run_times = iter(
        [
            datetime(2026, 6, 25, 2, 15, 30, tzinfo=timezone.utc),
            datetime(2026, 6, 25, 2, 16, 30, tzinfo=timezone.utc),
        ]
    )
    monkeypatch.setattr(snapshot_raw, "utc_now", lambda: next(run_times))

    output_root = tmp_path / "codeforces"
    first_manifest = snapshot_raw.create_snapshot_set(output_root)
    first_problemset_path = (
        output_root
        / first_manifest["entries"]["problemset_problems"]["snapshot_path"]
    )
    first_snapshot_bytes = first_problemset_path.read_bytes()

    second_manifest = snapshot_raw.create_snapshot_set(output_root)

    assert first_problemset_path.exists()
    assert first_problemset_path.read_bytes() == first_snapshot_bytes
    assert len(list((output_root / "snapshots").glob("*.json"))) == 4
    assert len(list((output_root / "provenance").glob("*.meta.json"))) == 4
    assert (
        second_manifest["entries"]["problemset_problems"]["snapshot_path"]
        == "snapshots/problemset_problems_20260625T021630Z.json"
    )
    assert _load_json(output_root / "manifest_latest.json") == second_manifest
