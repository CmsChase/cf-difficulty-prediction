"""Tests for rolling temporal validation."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import temporal_validation


def _feature_table(contest_count: int = 10) -> pd.DataFrame:
    """Return a small chronological feature table."""
    rows = []
    start = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
    for contest_id in range(1, contest_count + 1):
        for index, rank, tag_dp, tag_math, points in (
            ("A", 1, 0, 1, 500.0),
            ("B", 2, 1, 0, None),
        ):
            solved_count = 1500 // (contest_id + rank)
            rows.append(
                {
                    "contest_id": contest_id,
                    "index": index,
                    "name": f"Problem {contest_id}{index}",
                    "rating": 800 + contest_id * 70 + rank * 110,
                    "start_time_seconds": start + contest_id * 60 * 86400,
                    "index_letter": index,
                    "index_number": 0,
                    "index_rank": rank,
                    "has_points": int(points is not None),
                    "points": 0.0 if points is None else points,
                    "tag_count": tag_dp + tag_math,
                    "tag__dp": tag_dp,
                    "tag__math": tag_math,
                    "solved_count": solved_count,
                    "solved_count_missing": 0,
                    "log_solved_count": math.log1p(solved_count),
                }
            )
    return pd.DataFrame(rows)


def _feature_columns() -> list[str]:
    """Return modeling feature columns for synthetic fixtures."""
    return [
        "index_letter",
        "index_number",
        "index_rank",
        "tag_count",
        "tag__dp",
        "tag__math",
        "has_points",
        "points",
        "solved_count",
        "solved_count_missing",
        "log_solved_count",
    ]


def test_rolling_fold_construction() -> None:
    """Rolling folds use the requested expanding-window boundaries."""
    table = _feature_table(10)
    contest_times = temporal_validation.build_contest_time_table(table)

    folds = temporal_validation.build_rolling_folds(contest_times)

    assert len(folds) == 4
    assert [len(fold.train_contest_ids) for fold in folds] == [5, 6, 7, 8]
    assert [len(fold.test_contest_ids) for fold in folds] == [1, 1, 1, 1]
    assert folds[0].train_contest_ids == (1, 2, 3, 4, 5)
    assert folds[0].test_contest_ids == (6,)


def test_no_contest_overlap_and_no_future_training() -> None:
    """No fold leaks contest ids or future contests into training."""
    contest_times = temporal_validation.build_contest_time_table(_feature_table(10))
    start_by_contest = dict(
        zip(contest_times["contest_id"], contest_times["start_time_seconds"])
    )

    folds = temporal_validation.build_rolling_folds(contest_times)

    for fold in folds:
        assert set(fold.train_contest_ids).isdisjoint(fold.test_contest_ids)
        max_train_time = max(start_by_contest[contest] for contest in fold.train_contest_ids)
        min_test_time = min(start_by_contest[contest] for contest in fold.test_contest_ids)
        assert max_train_time < min_test_time


def test_age_normalized_feature_construction() -> None:
    """Age-normalized exposure features use snapshot time and clamp age."""
    table = pd.DataFrame(
        {
            "contest_id": [1, 2],
            "index": ["A", "B"],
            "rating": [800, 900],
            "start_time_seconds": [
                int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()),
                int(datetime(2020, 1, 10, 12, tzinfo=timezone.utc).timestamp()),
            ],
            "solved_count": [100, 10],
        }
    )

    enriched = temporal_validation.add_age_normalized_features(
        table,
        pd.Timestamp("2020-01-11T00:00:00Z"),
    )

    assert enriched.loc[0, "problem_age_days"] == 10.0
    assert enriched.loc[0, "solves_per_day"] == 10.0
    assert enriched.loc[1, "problem_age_days"] == 1.0
    assert enriched.loc[1, "solves_per_day"] == 10.0
    assert enriched.loc[0, "log_solves_per_day"] == math.log1p(10.0)


def test_metric_calculation_for_rolling_windows() -> None:
    """Rolling-window evaluation emits metrics for every feature setting."""
    table = temporal_validation.add_age_normalized_features(
        _feature_table(10),
        pd.Timestamp("2023-01-01T00:00:00Z"),
    )
    contest_times = temporal_validation.build_contest_time_table(table)
    folds = temporal_validation.build_rolling_folds(contest_times)
    groups = temporal_validation.build_feature_groups(table, _feature_columns())
    feature_sets = temporal_validation.build_feature_sets(groups)

    metrics = temporal_validation.evaluate_rolling_windows(
        table,
        folds,
        feature_sets,
        seed=7,
    )

    assert len(metrics) == 16
    assert set(metrics["feature_set_name"]) == set(
        temporal_validation.FEATURE_SET_ORDER
    )
    assert {"MAE", "RMSE", "R2", "within_100", "within_200"}.issubset(
        metrics.columns
    )


def test_drift_summary_calculation() -> None:
    """Drift summary reports descriptive train-test distribution changes."""
    table = temporal_validation.add_age_normalized_features(
        _feature_table(10),
        pd.Timestamp("2023-01-01T00:00:00Z"),
    )
    folds = temporal_validation.build_rolling_folds(
        temporal_validation.build_contest_time_table(table)
    )

    drift = temporal_validation.build_temporal_drift_summary(table, folds)

    assert not drift.empty
    assert {"rating", "solved_count", "problem_age_days"}.issubset(
        set(drift["column_name"])
    )
    assert {
        "train_mean",
        "test_mean",
        "mean_difference",
        "absolute_standardized_mean_difference",
    }.issubset(drift.columns)


def test_run_temporal_validation_writes_outputs(tmp_path: Path) -> None:
    """Tiny smoke test writes every required temporal-validation artifact."""
    feature_path = tmp_path / "model_table.parquet"
    feature_columns_path = tmp_path / "feature_columns.json"
    output_dir = tmp_path / "temporal_validation"
    log_path = tmp_path / "logs" / "temporal_validation.log"
    config_path = tmp_path / "experiment.yaml"

    table = _feature_table(10)
    table.to_parquet(feature_path, engine="pyarrow", index=False)
    feature_columns_path.write_text(
        json.dumps({"feature_columns": _feature_columns()}),
        encoding="utf-8",
    )
    config_path.write_text("project:\n  random_seed: 7\n", encoding="utf-8")

    paths = temporal_validation.run_temporal_validation(
        config_path=config_path,
        feature_path=feature_path,
        feature_columns_path=feature_columns_path,
        output_dir=output_dir,
        log_path=log_path,
    )

    assert all(path.is_file() for path in paths.values())
    assert log_path.is_file()
    summary = json.loads(
        paths["temporal_validation_summary"].read_text(encoding="utf-8")
    )
    assert summary["total_rows"] == len(table)
    assert summary["rolling_fold_count"] == 4
    assert "Rolling-window validation tests temporal stability" in " ".join(
        summary["conservative_interpretation_notes"]
    )
