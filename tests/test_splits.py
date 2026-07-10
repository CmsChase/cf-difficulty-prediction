"""Tests for leakage-resistant Codeforces split generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import features, splits


def _model_table() -> pd.DataFrame:
    """Return two problems for each of six chronologically ordered contests."""
    rows = []
    for contest_id in range(1, 7):
        for index in ("A", "B"):
            rows.append(
                {
                    "contest_id": contest_id,
                    "index": index,
                    "name": f"Problem {contest_id}{index}",
                    "rating": 800 + contest_id * 100,
                    "start_time_seconds": contest_id * 1000,
                    "index_rank": 1 if index == "A" else 2,
                }
            )
    return pd.DataFrame(rows)


def test_grouped_split_has_zero_contest_overlap() -> None:
    """A contest appears in exactly one randomized partition."""
    config = features.ExperimentConfig(
        random_seed=42,
        grouped_split=features.SplitRatios(0.5, 0.25, 0.25),
    )
    assignment = splits.build_contest_grouped_split(_model_table(), config)
    per_contest = assignment.groupby("contest_id")["split_name"].nunique()

    assert per_contest.max() == 1
    assert set(assignment["split_name"]) == {"train", "valid", "test"}
    repeated = splits.build_contest_grouped_split(_model_table(), config)
    pd.testing.assert_frame_equal(assignment, repeated)


def test_forward_time_split_is_strictly_ordered() -> None:
    """Validation and test contest times are strictly later than training."""
    config = features.ExperimentConfig(
        forward_time_split=features.SplitRatios(0.5, 0.25, 0.25),
    )
    table = _model_table()
    assignment = splits.build_forward_time_split(table, config)
    joined = assignment.merge(
        table[["contest_id", "index", "start_time_seconds"]],
        on=["contest_id", "index"],
        validate="one_to_one",
    )
    ranges = joined.groupby("split_name")["start_time_seconds"].agg(
        ["min", "max"]
    )

    assert ranges.loc["train", "max"] < ranges.loc["valid", "min"]
    assert ranges.loc["valid", "max"] < ranges.loc["test", "min"]


def test_split_outputs_and_summary(tmp_path: Path) -> None:
    """Both assignment files and the combined summary are persisted."""
    input_path = tmp_path / "model_table.parquet"
    output_dir = tmp_path / "splits"
    log_path = tmp_path / "splits.log"
    config_path = tmp_path / "experiment.yaml"
    _model_table().to_parquet(input_path, engine="pyarrow", index=False)
    config_path.write_text(
        (PROJECT_ROOT / "configs" / "experiment.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    paths = splits.generate_splits(
        input_path,
        output_dir,
        config_path=config_path,
        log_path=log_path,
    )

    assert all(path.is_file() for path in paths.values())
    grouped = pd.read_parquet(paths["contest_grouped_split"])
    forward = pd.read_parquet(paths["forward_time_split"])
    required = {
        "contest_id",
        "index",
        "split_name",
        "fold",
        "strategy",
        "config_fingerprint_sha256",
    }
    assert set(grouped.columns) == required
    assert set(forward.columns) == required
    summary = json.loads(paths["split_summary"].read_text(encoding="utf-8"))
    assert summary["contest_grouped"]["zero_contest_overlap"] is True
    assert summary["contest_grouped"]["contest_overlap_count"] == 0
    assert summary["forward_time"]["strictly_ordered"] is True
    assert sum(summary["contest_grouped"]["row_counts"].values()) == 12
    assert log_path.is_file()
    log_record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert log_record["logger"] == "cf_diff.splits"
