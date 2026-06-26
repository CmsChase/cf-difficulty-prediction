"""Tests for the Codeforces preprocessing pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import preprocess


@pytest.fixture
def raw_snapshot(tmp_path: Path) -> Path:
    """Create a compact fetched-snapshot fixture."""
    raw_dir = tmp_path / "raw" / "latest"
    raw_dir.mkdir(parents=True)
    problemset_payload = {
        "status": "OK",
        "result": {
            "problems": [
                {
                    "contestId": 2,
                    "index": "B",
                    "name": "Rated later",
                    "type": "PROGRAMMING",
                    "rating": 1200,
                    "tags": ["graphs"],
                },
                {
                    "contestId": 1,
                    "index": "A",
                    "name": "Rated earlier",
                    "type": "PROGRAMMING",
                    "points": 500.0,
                    "rating": 800,
                    "tags": ["math", "implementation"],
                },
                {
                    "contestId": 1,
                    "index": "B",
                    "name": "Unrated",
                    "type": "PROGRAMMING",
                    "tags": [],
                },
                {
                    "contestId": 1,
                    "index": "C",
                    "name": "Question",
                    "type": "OTHER",
                    "rating": 900,
                    "tags": ["interactive"],
                },
            ],
            "problemStatistics": [
                {"contestId": 1, "index": "A", "solvedCount": 100},
                {"contestId": 1, "index": "B", "solvedCount": 50},
                {"contestId": 1, "index": "C", "solvedCount": 25},
            ],
        },
    }
    contest_payload = {
        "status": "OK",
        "result": [
            {
                "id": 2,
                "name": "Later Round",
                "type": "CF",
                "phase": "FINISHED",
                "startTimeSeconds": 2000,
                "durationSeconds": 7200,
            },
            {
                "id": 1,
                "name": "Earlier Round",
                "type": "CF",
                "phase": "FINISHED",
                "startTimeSeconds": 1000,
                "durationSeconds": 7200,
            },
        ],
    }
    manifest = {
        "snapshot_id": "test",
        "resources": {
            "problemset.problems": {
                "output_path": "problemset.problems.json"
            },
            "contest.list": {"output_path": "contest.list.json"},
        },
    }
    (raw_dir / "problemset.problems.json").write_text(
        json.dumps(problemset_payload),
        encoding="utf-8",
    )
    (raw_dir / "contest.list.json").write_text(
        json.dumps(contest_payload),
        encoding="utf-8",
    )
    (raw_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return raw_dir


def test_preprocess_outputs_merge_filter_dictionary_and_summary(
    raw_snapshot: Path,
    tmp_path: Path,
) -> None:
    """The pipeline writes documented tables with the required semantics."""
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    log_path = tmp_path / "outputs" / "logs" / "preprocess.log"

    paths = preprocess.preprocess_snapshot(
        raw_snapshot,
        interim_dir,
        processed_dir,
        log_path=log_path,
    )

    expected_files = {
        interim_dir / "problems.parquet",
        interim_dir / "problem_statistics.parquet",
        interim_dir / "contests.parquet",
        interim_dir / "problems_merged.parquet",
        interim_dir / "data_dictionary.csv",
        processed_dir / "rated_programming_problems.parquet",
        processed_dir / "preprocess_summary.json",
    }
    assert expected_files == set(paths.values())
    assert all(path.is_file() for path in expected_files)
    assert log_path.is_file()

    merged = pd.read_parquet(interim_dir / "problems_merged.parquet")
    solved_lookup = merged.set_index(["contest_id", "index"])["solved_count"]
    assert solved_lookup.loc[(1, "A")] == 100
    assert solved_lookup.loc[(1, "B")] == 50
    assert pd.isna(solved_lookup.loc[(2, "B")])
    assert "start_time_seconds" in merged.columns
    assert "contest_name" in merged.columns

    rated = pd.read_parquet(
        processed_dir / "rated_programming_problems.parquet"
    )
    assert rated["name"].tolist() == ["Rated earlier", "Rated later"]
    assert rated["type"].tolist() == ["PROGRAMMING", "PROGRAMMING"]
    assert rated["rating"].tolist() == [800, 1200]
    assert pd.isna(rated.loc[1, "solved_count"])
    assert list(rated.loc[0, "tags"]) == ["math", "implementation"]
    assert list(rated.loc[1, "tags"]) == ["graphs"]

    dictionary = pd.read_csv(interim_dir / "data_dictionary.csv")
    assert dictionary.columns.tolist() == [
        "column_name",
        "table_name",
        "pandas_dtype",
        "source_endpoint",
        "source_object",
        "description",
        "nullable",
    ]
    solved_dictionary = dictionary.loc[
        (dictionary["table_name"] == "problems_merged")
        & (dictionary["column_name"] == "solved_count")
    ].iloc[0]
    assert solved_dictionary["source_endpoint"] == "problemset.problems"
    assert solved_dictionary["source_object"] == "ProblemStatistics"

    summary = json.loads(
        (processed_dir / "preprocess_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["row_counts"] == {
        "problems": 4,
        "problem_statistics": 3,
        "contests": 2,
        "problems_merged": 4,
        "rated_programming_problems": 2,
    }
    assert summary["number_of_rated_programming_problems"] == 2
    assert summary["missing_counts"]["solved_count"] == 1
    assert summary["min_rating"] == 800
    assert summary["max_rating"] == 1200
    assert summary["min_start_time_seconds"] == 1000
    assert summary["max_start_time_seconds"] == 2000

    log_events = [
        json.loads(line)["event"]
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert "tables_normalized" in log_events
    assert "modeling_filter_applied" in log_events
    assert "preprocess_completed" in log_events


def test_cli_runs_with_required_directories(
    raw_snapshot: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public CLI writes the expected processed table."""
    interim_dir = tmp_path / "cli-interim"
    processed_dir = tmp_path / "cli-processed"
    exit_code = preprocess.main(
        [
            "--raw-dir",
            str(raw_snapshot),
            "--interim-dir",
            str(interim_dir),
            "--processed-dir",
            str(processed_dir),
            "--log-path",
            str(tmp_path / "cli.log"),
        ]
    )

    assert exit_code == 0
    assert (
        processed_dir / "rated_programming_problems.parquet"
    ).is_file()
    assert "Wrote rated programming problems:" in capsys.readouterr().out


def test_invalid_numeric_values_are_counted(
    raw_snapshot: Path,
    tmp_path: Path,
) -> None:
    """Invalid numeric API values are made explicit in summary metadata."""
    problemset_path = raw_snapshot / "problemset.problems.json"
    payload = json.loads(problemset_path.read_text(encoding="utf-8"))
    payload["result"]["problems"][0]["rating"] = "not-a-rating"
    problemset_path.write_text(json.dumps(payload), encoding="utf-8")

    preprocess.preprocess_snapshot(
        raw_snapshot,
        tmp_path / "interim",
        tmp_path / "processed",
        log_path=tmp_path / "preprocess.log",
    )
    summary = json.loads(
        (tmp_path / "processed" / "preprocess_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["invalid_numeric_counts"]["problems.rating"] == 1
    assert summary["number_of_rated_programming_problems"] == 1
