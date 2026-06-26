"""Tests for Codeforces robustness experiments."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import robustness


def _model_table() -> pd.DataFrame:
    """Return a small model table with all robustness feature groups."""
    rows = []
    for contest_id in range(1, 7):
        for index, rank, tag_math, tag_dp, points in (
            ("A", 1, 1, 0, 500.0),
            ("B", 2, 0, 1, None),
        ):
            solved_count = 1200 // (contest_id + rank)
            rows.append(
                {
                    "contest_id": contest_id,
                    "index": index,
                    "name": f"Problem {contest_id}{index}",
                    "rating": 800 + contest_id * 80 + rank * 120,
                    "start_time_seconds": 1_600_000_000 + contest_id * 86_400,
                    "index_letter": index,
                    "index_number": 0,
                    "index_rank": rank,
                    "has_points": int(points is not None),
                    "points": 0.0 if points is None else points,
                    "tag_count": tag_math + tag_dp,
                    "tag__math": tag_math,
                    "tag__dp": tag_dp,
                    "solved_count": solved_count,
                    "solved_count_missing": 0,
                    "log_solved_count": math.log1p(solved_count),
                }
            )
    return pd.DataFrame(rows)


def _processed_table() -> pd.DataFrame:
    """Return processed rows needed to construct age-normalized features."""
    frame = _model_table()
    return frame.loc[
        :,
        [
            "contest_id",
            "index",
            "start_time_seconds",
            "solved_count",
        ],
    ].copy()


def _split_assignment(strategy: str) -> pd.DataFrame:
    """Return row-level split assignment with train, valid, and test."""
    split_by_contest = {
        1: "train",
        2: "train",
        3: "train",
        4: "valid",
        5: "test",
        6: "test",
    }
    rows = []
    for contest_id, split_name in split_by_contest.items():
        for index in ("A", "B"):
            rows.append(
                {
                    "contest_id": contest_id,
                    "index": index,
                    "split_name": split_name,
                    "fold": {"train": 0, "valid": 1, "test": 2}[split_name],
                    "strategy": strategy,
                }
            )
    return pd.DataFrame(rows)


def test_age_feature_construction_uses_snapshot_time() -> None:
    """Age-normalized features use elapsed days and clamp age to one day."""
    model = pd.DataFrame(
        {
            "contest_id": [1, 2],
            "index": ["A", "A"],
            "rating": [800, 900],
            "start_time_seconds": [
                int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()),
                int(datetime(2020, 1, 10, 12, tzinfo=timezone.utc).timestamp()),
            ],
            "solved_count": [100, 10],
        }
    )
    snapshot_time = pd.Timestamp("2020-01-11T00:00:00Z")

    enriched = robustness.add_age_normalized_features(
        model,
        model,
        snapshot_time,
    )

    assert enriched.loc[0, "problem_age_days"] == 10.0
    assert enriched.loc[0, "solves_per_day"] == 10.0
    assert enriched.loc[1, "problem_age_days"] == 1.0
    assert enriched.loc[1, "solves_per_day"] == 10.0
    assert enriched.loc[0, "log_solves_per_day"] == math.log1p(10.0)


def test_cold_start_feature_sets_exclude_raw_solved_features() -> None:
    """Cold-start and age-normalized-only sets avoid raw solved features."""
    enriched = robustness.add_age_normalized_features(
        _model_table(),
        _processed_table(),
        pd.Timestamp("2020-10-01T00:00:00Z"),
    )
    feature_columns = [
        "index_letter",
        "index_number",
        "index_rank",
        "has_points",
        "points",
        "tag_count",
        "tag__math",
        "tag__dp",
        "solved_count",
        "solved_count_missing",
        "log_solved_count",
    ]

    groups = robustness.build_robustness_feature_groups(enriched, feature_columns)
    feature_sets = robustness.build_robustness_feature_sets(groups)

    metadata_columns = set(
        feature_sets["metadata_only_cold_start"]["feature_columns"]
    )
    age_only_columns = set(
        feature_sets["age_normalized_solved_only"]["feature_columns"]
    )
    assert metadata_columns.isdisjoint(robustness.RAW_SOLVED_COLUMNS)
    assert age_only_columns == set(robustness.AGE_NORMALIZED_COLUMNS)
    assert set(feature_sets["full_api_reference"]["feature_columns"]) & set(
        robustness.RAW_SOLVED_COLUMNS
    )
    assert set(feature_sets["full_api_plus_age_norm"]["feature_columns"]) & set(
        robustness.RAW_SOLVED_COLUMNS
    )
    assert set(robustness.AGE_NORMALIZED_COLUMNS).issubset(
        set(feature_sets["full_api_plus_age_norm"]["feature_columns"])
    )


def test_metric_calculation() -> None:
    """Robustness uses the shared regression metric definitions."""
    metrics = robustness.compute_regression_metrics(
        [1000, 1200, 1400],
        [900, 1200, 1600],
    )

    assert metrics["MAE"] == 100.0
    assert metrics["RMSE"] == 129.099445
    assert metrics["within_100"] == 0.666667
    assert metrics["within_200"] == 1.0


def test_cold_start_gap_calculation() -> None:
    """Cold-start comparison reports MAE gap from full API reference."""
    test_metrics = pd.DataFrame(
        {
            "strategy": ["contest_grouped", "contest_grouped"],
            "model_name": ["ridge_regression", "ridge_regression"],
            "feature_set_name": [
                "full_api_reference",
                "metadata_only_cold_start",
            ],
            "MAE": [100.0, 125.0],
            "RMSE": [120.0, 140.0],
            "R2": [0.5, 0.4],
            "within_100": [0.7, 0.6],
            "within_200": [0.9, 0.8],
            "feature_count": [10, 7],
            "row_count": [20, 20],
        }
    )

    comparison = robustness.build_cold_start_comparison(test_metrics)
    metadata = comparison.loc[
        comparison["feature_set_name"].eq("metadata_only_cold_start")
    ].iloc[0]

    assert metadata["absolute_MAE_gap_vs_full_api"] == 25.0
    assert metadata["percent_MAE_gap_vs_full_api"] == 25.0


def test_run_robustness_writes_outputs(tmp_path: Path) -> None:
    """Tiny smoke test writes every required robustness artifact."""
    processed_path = tmp_path / "rated_programming_problems.parquet"
    feature_path = tmp_path / "model_table.parquet"
    feature_columns_path = tmp_path / "feature_columns.json"
    contest_split_path = tmp_path / "contest_grouped_split.parquet"
    time_split_path = tmp_path / "forward_time_split.parquet"
    output_dir = tmp_path / "robustness"
    log_path = tmp_path / "logs" / "robustness.log"
    config_path = tmp_path / "experiment.yaml"

    _processed_table().to_parquet(processed_path, engine="pyarrow", index=False)
    _model_table().to_parquet(feature_path, engine="pyarrow", index=False)
    feature_columns_path.write_text(
        json.dumps(
            {
                "feature_columns": [
                    "index_letter",
                    "index_number",
                    "index_rank",
                    "has_points",
                    "points",
                    "tag_count",
                    "tag__math",
                    "tag__dp",
                    "solved_count",
                    "solved_count_missing",
                    "log_solved_count",
                ]
            }
        ),
        encoding="utf-8",
    )
    _split_assignment("contest_grouped").to_parquet(
        contest_split_path,
        engine="pyarrow",
        index=False,
    )
    _split_assignment("forward_time").to_parquet(
        time_split_path,
        engine="pyarrow",
        index=False,
    )
    config_path.write_text("project:\n  random_seed: 7\n", encoding="utf-8")

    paths = robustness.run_robustness(
        config_path=config_path,
        processed_path=processed_path,
        feature_path=feature_path,
        feature_columns_path=feature_columns_path,
        contest_split_path=contest_split_path,
        time_split_path=time_split_path,
        output_dir=output_dir,
        log_path=log_path,
    )

    assert all(path.is_file() for path in paths.values())
    assert log_path.is_file()
    test_metrics = pd.read_csv(paths["robustness_metrics_test"])
    assert set(test_metrics["strategy"]) == {"contest_grouped", "forward_time"}
    assert {
        "metadata_only_cold_start",
        "age_normalized_solved_only",
        "full_api_reference",
    }.issubset(set(test_metrics["feature_set_name"]))
    summary = json.loads(paths["robustness_summary"].read_text(encoding="utf-8"))
    assert "cold_start" in summary["experiment_families"]
    assert "age_normalized" in summary["experiment_families"]
