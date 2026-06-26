"""Tests for deterministic Codeforces snapshot cleaning."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import clean_data


@pytest.fixture
def raw_snapshot_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write compact raw API fixtures and a compatible latest manifest."""
    problemset_payload = {
        "status": "OK",
        "result": {
            "problems": [
                {
                    "contestId": 100,
                    "index": "A1",
                    "name": "First",
                    "type": "PROGRAMMING",
                    "points": 500.0,
                    "rating": 800,
                    "tags": ["math", "implementation"],
                },
                {
                    "contestId": 100,
                    "index": "AA12",
                    "name": "Second",
                    "type": "PROGRAMMING",
                    "rating": 900,
                    "tags": ["graphs"],
                },
                {
                    "contestId": 100,
                    "index": "B",
                    "name": "Third",
                    "type": "PROGRAMMING",
                    "rating": 1000,
                },
                {
                    "contestId": 100,
                    "index": "C",
                    "name": "Question",
                    "type": "OTHER",
                    "rating": 1100,
                },
                {
                    "contestId": 100,
                    "index": "D",
                    "name": "Unrated",
                    "type": "PROGRAMMING",
                },
                {
                    "contestId": 999,
                    "index": "E2",
                    "name": "Missing contest",
                    "type": "PROGRAMMING",
                    "rating": 1200,
                },
                {
                    "contestId": 100,
                    "index": "F",
                    "name": "Missing statistics",
                    "type": "PROGRAMMING",
                    "rating": 1300,
                },
            ],
            "problemStatistics": [
                {"contestId": 100, "index": "A1", "solvedCount": 10},
                {"contestId": 100, "index": "AA12", "solvedCount": 20},
                {"contestId": 100, "index": "B", "solvedCount": 30},
                {"contestId": 100, "index": "C", "solvedCount": 40},
                {"contestId": 100, "index": "D", "solvedCount": 50},
                {"contestId": 999, "index": "E2", "solvedCount": 60},
            ],
        },
    }
    contest_payload = {
        "status": "OK",
        "result": [
            {
                "id": 100,
                "name": "Test Round",
                "type": "CF",
                "phase": "FINISHED",
                "durationSeconds": 7200,
                "startTimeSeconds": 1_700_000_000,
            }
        ],
    }

    raw_root = tmp_path / "raw" / "codeforces"
    snapshots_dir = raw_root / "snapshots"
    snapshots_dir.mkdir(parents=True)
    problemset_path = snapshots_dir / "problemset_problems_test.json"
    contest_path = snapshots_dir / "contest_list_test.json"
    problemset_path.write_text(
        json.dumps(problemset_payload),
        encoding="utf-8",
    )
    contest_path.write_text(
        json.dumps(contest_payload),
        encoding="utf-8",
    )

    manifest_path = raw_root / "manifest_latest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": "2026-06-25T00:00:00+00:00",
                "seed": 42,
                "entries": {
                    "problemset_problems": {
                        "snapshot_path": (
                            "snapshots/problemset_problems_test.json"
                        ),
                        "provenance_path": "unused.meta.json",
                    },
                    "contest_list": {
                        "snapshot_path": "snapshots/contest_list_test.json",
                        "provenance_path": "unused.meta.json",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return problemset_path, contest_path, manifest_path


def test_manifest_cli_writes_and_cleans_expected_tables(
    raw_snapshot_files: tuple[Path, Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The manifest CLI filters rows, merges counts, and derives index fields."""
    _, _, manifest_path = raw_snapshot_files
    output_root = tmp_path / "data"

    exit_code = clean_data.main(
        [
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--seed",
            "42",
        ]
    )

    assert exit_code == 0
    expected_paths = [
        output_root / "interim" / "problems.parquet",
        output_root / "interim" / "problem_statistics.parquet",
        output_root / "interim" / "contests.parquet",
        output_root / "processed" / "problems_merged.parquet",
        output_root / "processed" / "problems_model_ready.parquet",
        output_root / "processed" / "data_dictionary.md",
    ]
    assert all(path.is_file() for path in expected_paths)

    model = pd.read_parquet(
        output_root / "processed" / "problems_model_ready.parquet"
    )
    assert model["index"].tolist() == ["A1", "AA12", "B"]
    assert model["solved_count"].tolist() == [10, 20, 30]
    assert model["index_letter"].tolist() == ["A", "AA", "B"]
    assert model["index_suffix"].tolist()[:2] == [1, 12]
    assert pd.isna(model.loc[2, "index_suffix"])
    assert model["index_rank"].tolist() == [1, 1, 2]
    assert model["contest_phase"].tolist() == ["FINISHED"] * 3
    assert model["contest_type"].tolist() == ["CF"] * 3
    assert model["tag_count"].tolist() == [2, 1, 0]
    assert model["has_points"].tolist() == [1, 0, 0]

    dictionary = (
        output_root / "processed" / "data_dictionary.md"
    ).read_text(encoding="utf-8")
    for column in (
        "solved_count",
        "contest_start_time_seconds",
        "index_letter",
        "index_suffix",
        "index_rank",
        "log_solved_count",
    ):
        assert f"| {column} |" in dictionary
    assert "Wrote cleaned Codeforces data:" in capsys.readouterr().out


def test_explicit_snapshot_paths_are_supported(
    raw_snapshot_files: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    """The CLI accepts the explicit two-snapshot input mode."""
    problemset_path, contest_path, _ = raw_snapshot_files
    output_root = tmp_path / "explicit"

    exit_code = clean_data.main(
        [
            "--problemset-snapshot",
            str(problemset_path),
            "--contest-snapshot",
            str(contest_path),
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    assert (
        output_root / "processed" / "problems_model_ready.parquet"
    ).is_file()


def test_incomplete_input_selection_fails_clearly(
    raw_snapshot_files: tuple[Path, Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One explicit snapshot without the other is rejected."""
    problemset_path, _, _ = raw_snapshot_files

    exit_code = clean_data.main(
        [
            "--problemset-snapshot",
            str(problemset_path),
            "--output-root",
            str(tmp_path / "invalid"),
        ]
    )

    assert exit_code == 1
    assert "provide both" in capsys.readouterr().err.lower()
