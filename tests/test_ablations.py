"""Tests for Codeforces feature-group ablation experiments."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import ablations


def _model_table() -> pd.DataFrame:
    """Return a small model table with all ablation feature groups."""
    rows = []
    split_contests = (1, 2, 3)
    for contest_id in split_contests:
        for index, rank, tag_math, tag_dp, points in (
            ("A", 1, 1, 0, 500.0),
            ("B", 2, 0, 1, 750.0),
            ("C", 3, 1, 1, None),
        ):
            solved_count = 1000 // (rank + contest_id)
            rows.append(
                {
                    "contest_id": contest_id,
                    "index": index,
                    "name": f"Problem {contest_id}{index}",
                    "rating": 800 + contest_id * 100 + rank * 150,
                    "start_time_seconds": contest_id * 1000,
                    "index_letter": index,
                    "index_number": 0,
                    "index_rank": rank,
                    "solved_count": solved_count,
                    "solved_count_missing": 0,
                    "log_solved_count": math.log1p(solved_count),
                    "tag_count": tag_math + tag_dp,
                    "tag__math": tag_math,
                    "tag__dp": tag_dp,
                    "has_points": int(points is not None),
                    "points": 0.0 if points is None else points,
                }
            )
    return pd.DataFrame(rows)


def _split_assignment(strategy: str) -> pd.DataFrame:
    """Return row-level split assignment with train, valid, and test."""
    split_by_contest = {1: "train", 2: "valid", 3: "test"}
    rows = []
    for contest_id, split_name in split_by_contest.items():
        for index in ("A", "B", "C"):
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


def test_feature_group_selection() -> None:
    """Feature groups map to the intended columns."""
    groups = ablations.build_feature_groups(_model_table())
    feature_sets = ablations.build_feature_sets(groups)

    assert groups["index"] == ["index_letter", "index_number", "index_rank"]
    assert groups["solved"] == [
        "log_solved_count",
        "solved_count",
        "solved_count_missing",
    ]
    assert groups["tags"] == ["tag__dp", "tag__math", "tag_count"]
    assert groups["points"] == ["has_points", "points"]
    assert feature_sets["all_without_solved"]["included_groups"] == [
        "index",
        "tags",
        "points",
    ]
    assert "solved_count" not in feature_sets["all_without_solved"][
        "feature_columns"
    ]


def test_metric_calculation() -> None:
    """Ablations expose the shared regression metric definitions."""
    metrics = ablations.compute_regression_metrics(
        [1000, 1200, 1400],
        [900, 1200, 1600],
    )

    assert metrics["MAE"] == 100.0
    assert metrics["RMSE"] == 129.099445
    assert metrics["within_100"] == 0.666667
    assert metrics["within_200"] == 1.0


def test_drop_comparison_calculation() -> None:
    """Drop comparison reports MAE increase relative to all features."""
    test_metrics = pd.DataFrame(
        {
            "strategy": ["contest_grouped"] * 5,
            "model_name": ["ridge_regression"] * 5,
            "feature_set_name": [
                "all_api_features",
                "all_without_index",
                "all_without_solved",
                "all_without_tags",
                "all_without_points",
            ],
            "MAE": [100.0, 110.0, 150.0, 105.0, 95.0],
        }
    )

    comparison = ablations.build_drop_comparison(test_metrics)
    solved = comparison.loc[
        comparison["removed_group"].eq("solved")
    ].iloc[0]
    points = comparison.loc[
        comparison["removed_group"].eq("points")
    ].iloc[0]

    assert solved["MAE_difference"] == 50.0
    assert solved["percent_MAE_change"] == 50.0
    assert points["MAE_difference"] == -5.0


def test_run_ablations_writes_outputs(tmp_path: Path) -> None:
    """Tiny smoke test writes every required ablation artifact."""
    feature_path = tmp_path / "model_table.parquet"
    feature_columns_path = tmp_path / "feature_columns.json"
    contest_split_path = tmp_path / "contest_grouped_split.parquet"
    time_split_path = tmp_path / "forward_time_split.parquet"
    output_dir = tmp_path / "ablations"
    log_path = tmp_path / "logs" / "ablations.log"
    config_path = tmp_path / "experiment.yaml"

    _model_table().to_parquet(feature_path, engine="pyarrow", index=False)
    feature_columns_path.write_text(
        json.dumps({"feature_columns": list(_model_table().columns)}),
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

    paths = ablations.run_ablations(
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
    test_metrics = pd.read_csv(paths["ablation_metrics_test"])
    assert set(test_metrics["strategy"]) == {"contest_grouped", "forward_time"}
    assert set(test_metrics["model_name"]) == {
        "ridge_regression",
        "hist_gradient_boosting_regressor",
    }
    assert len(test_metrics) == 48
    drop_comparison = pd.read_csv(paths["ablation_drop_comparison"])
    assert set(drop_comparison["removed_group"]) == {
        "index",
        "solved",
        "tags",
        "points",
    }
