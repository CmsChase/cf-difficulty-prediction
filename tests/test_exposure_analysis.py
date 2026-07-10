"""Tests for exposure-aware Codeforces analysis."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import exposure_analysis


def _feature_table() -> pd.DataFrame:
    """Return a small feature table with exposure-analysis inputs."""
    rows = []
    start = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
    for contest_id in range(1, 7):
        for index, rank, tag_dp, tag_math, points in (
            ("A", 1, 0, 1, 500.0),
            ("B", 2, 1, 0, None),
        ):
            solved_count = 1200 // (contest_id + rank)
            rows.append(
                {
                    "contest_id": contest_id,
                    "index": index,
                    "name": f"Problem {contest_id}{index}",
                    "rating": 800 + contest_id * 100 + rank * 120,
                    "start_time_seconds": start + contest_id * 180 * 86400,
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


def test_age_feature_construction() -> None:
    """Exposure features use snapshot time and clamp age to at least one day."""
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

    enriched = exposure_analysis.add_exposure_features(
        table,
        pd.Timestamp("2020-01-11T00:00:00Z"),
    )

    assert enriched.loc[0, "problem_age_days"] == 10.0
    assert enriched.loc[0, "solves_per_day"] == 10.0
    assert enriched.loc[1, "problem_age_days"] == 1.0
    assert enriched.loc[1, "solves_per_day"] == 10.0
    assert enriched.loc[0, "log_solves_per_day"] == math.log1p(10.0)


def test_age_bucket_assignment() -> None:
    """Problem ages are assigned to the expected fixed buckets."""
    buckets = exposure_analysis.assign_age_buckets(
        pd.Series([30, 365, 366, 3 * 365, 3 * 365 + 1, 5 * 365, 5 * 365 + 1])
    )

    assert buckets.astype(str).tolist() == [
        "0-1y",
        "0-1y",
        "1-3y",
        "1-3y",
        "3-5y",
        "3-5y",
        "5y+",
    ]


def test_feature_set_selection_excludes_raw_solved_when_required() -> None:
    """Metadata and metadata-plus-age-normalized sets exclude raw solved fields."""
    table = exposure_analysis.add_exposure_features(
        _feature_table(),
        pd.Timestamp("2024-01-01T00:00:00Z"),
    )
    feature_columns = [
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

    groups = exposure_analysis.build_feature_groups(table, feature_columns)
    feature_sets = exposure_analysis.build_feature_sets(groups)

    assert set(feature_sets["metadata_only"]).isdisjoint(
        exposure_analysis.RAW_SOLVED_COLUMNS
    )
    assert set(feature_sets["metadata_plus_age_norm"]).isdisjoint(
        exposure_analysis.RAW_SOLVED_COLUMNS
    )
    assert set(exposure_analysis.AGE_EXPOSURE_COLUMNS).issubset(
        set(feature_sets["metadata_plus_age_norm"])
    )
    assert set(exposure_analysis.RAW_SOLVED_COLUMNS).issubset(
        set(feature_sets["full_api_plus_age_norm"])
    )


def test_mismatch_group_selection() -> None:
    """Mismatch examples are selected from quantile-defined corners."""
    table = pd.DataFrame(
        {
            "contest_id": range(1, 9),
            "index": ["A"] * 8,
            "name": [f"P{i}" for i in range(1, 9)],
            "rating": [800, 850, 900, 950, 2500, 2600, 2700, 2800],
            "solved_count": [1, 100, 2, 200, 3, 300, 4, 400],
            "problem_age_days": [100] * 8,
            "solves_per_day": [0.01, 1.0, 0.02, 2.0, 0.03, 3.0, 0.04, 4.0],
            "log_solves_per_day": [
                math.log1p(value)
                for value in [0.01, 1.0, 0.02, 2.0, 0.03, 3.0, 0.04, 4.0]
            ],
            "tag__dp": [1] * 8,
        }
    )

    examples = exposure_analysis.select_mismatch_examples(
        table,
        max_examples_per_group=5,
    )

    assert "popular_hard" in set(examples["mismatch_group"])
    assert "underexposed_easy" in set(examples["mismatch_group"])
    assert "tags" in examples.columns


def test_metric_calculation_by_age_bucket() -> None:
    """Age-bucket evaluation emits test metrics by feature setting."""
    table = exposure_analysis.add_exposure_features(
        _feature_table(),
        pd.Timestamp("2024-01-01T00:00:00Z"),
    )
    groups = exposure_analysis.build_feature_groups(table)
    feature_sets = exposure_analysis.build_feature_sets(groups)

    metrics = exposure_analysis.evaluate_age_bucket_metrics(
        table,
        _split_assignment("contest_grouped"),
        strategy="contest_grouped",
        feature_sets=feature_sets,
        seed=7,
    )

    assert not metrics.empty
    assert set(metrics["split_name"]) == {"test"}
    assert "metadata_only" in set(metrics["feature_set_name"])
    assert {"row_count", "MAE", "RMSE", "R2", "within_100", "within_200"}.issubset(
        metrics.columns
    )


def test_exposure_summary_does_not_select_a_winner_from_test_mae() -> None:
    """Test-partition age metrics remain descriptive, even with a clear MAE minimum."""
    snapshot_time = pd.Timestamp("2024-01-01T00:00:00Z")
    frame = exposure_analysis.add_exposure_features(_feature_table(), snapshot_time)
    age_bucket_metrics = pd.DataFrame(
        [
            {
                "strategy": "contest_grouped",
                "model_name": "model_a",
                "feature_set_name": "features_a",
                "age_bucket": "0-1y",
                "MAE": 1.0,
            },
            {
                "strategy": "forward_time",
                "model_name": "model_b",
                "feature_set_name": "features_b",
                "age_bucket": "0-1y",
                "MAE": 999.0,
            },
        ]
    )

    summary = exposure_analysis.build_exposure_summary(
        frame,
        age_bucket_metrics,
        pd.DataFrame(columns=["mismatch_group"]),
        snapshot_time,
    )

    assert "main_finding_from_age_bucket_analysis" not in summary
    descriptive = summary["descriptive_age_bucket_analysis"]
    assert descriptive["analysis_role"] == "exploratory_descriptive_test_comparison"
    assert descriptive["comparison_row_count"] == 2
    assert descriptive["model_selection_performed"] is False
    assert "strategy" not in descriptive
    assert "model_name" not in descriptive
    assert "feature_set_name" not in descriptive
    assert "mean_age_bucket_MAE" not in descriptive


def test_run_exposure_analysis_writes_outputs(tmp_path: Path) -> None:
    """Tiny smoke test writes every required exposure-analysis artifact."""
    feature_path = tmp_path / "model_table.parquet"
    feature_columns_path = tmp_path / "feature_columns.json"
    contest_split_path = tmp_path / "contest_grouped_split.parquet"
    time_split_path = tmp_path / "forward_time_split.parquet"
    output_dir = tmp_path / "exposure"
    log_path = tmp_path / "logs" / "exposure_analysis.log"
    config_path = tmp_path / "experiment.yaml"

    table = _feature_table()
    table.to_parquet(feature_path, engine="pyarrow", index=False)
    feature_columns_path.write_text(
        json.dumps(
            {
                "feature_columns": [
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
    config_path.write_text(
        (PROJECT_ROOT / "configs" / "experiment.yaml")
        .read_text(encoding="utf-8")
        .replace("random_seed: 42", "random_seed: 7"),
        encoding="utf-8",
    )

    paths = exposure_analysis.run_exposure_analysis(
        config_path=config_path,
        feature_path=feature_path,
        feature_columns_path=feature_columns_path,
        contest_split_path=contest_split_path,
        time_split_path=time_split_path,
        output_dir=output_dir,
        log_path=log_path,
    )

    assert all(path.is_file() for path in paths.values())
    assert log_path.is_file()
    summary = json.loads(paths["exposure_summary"].read_text(encoding="utf-8"))
    assert summary["total_rows"] == len(table)
    assert "solved count is useful" in " ".join(
        summary["conservative_interpretation_notes"]
    ).lower()
