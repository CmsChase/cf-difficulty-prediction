"""Tests for baseline Codeforces rating models."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import baselines


def _model_table() -> pd.DataFrame:
    """Return a small feature table with three chronological split groups."""
    rows = []
    for contest_id, split_offset in zip((1, 2, 3), (0, 100, 200), strict=True):
        for index, rank, tag_math, tag_dp, solved in (
            ("A", 1, 1, 0, 1000 - split_offset),
            ("B", 2, 0, 1, 200 - split_offset // 2),
        ):
            rows.append(
                {
                    "contest_id": contest_id,
                    "index": index,
                    "name": f"Problem {contest_id}{index}",
                    "rating": 800 + split_offset + rank * 200,
                    "start_time_seconds": contest_id * 1000,
                    "index_letter": index,
                    "index_number": 0,
                    "index_rank": rank,
                    "tag_count": 1,
                    "tag__math": tag_math,
                    "tag__dp": tag_dp,
                    "solved_count": solved,
                    "log_solved_count": math.log1p(max(solved, 0)),
                    "solved_count_missing": 0,
                    "has_points": 0,
                    "points": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _split_assignment(strategy: str) -> pd.DataFrame:
    """Return one row-level assignment containing train, valid, and test."""
    split_by_contest = {1: "train", 2: "valid", 3: "test"}
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


def _feature_columns() -> list[str]:
    """Return the feature metadata used by tests."""
    return [
        "index_letter",
        "index_number",
        "index_rank",
        "tag_count",
        "tag__math",
        "tag__dp",
        "solved_count",
        "log_solved_count",
        "solved_count_missing",
        "has_points",
        "points",
    ]


def test_compute_regression_metrics() -> None:
    """Regression metrics use the expected definitions."""
    metrics = baselines.compute_regression_metrics(
        [1000, 1200, 1400],
        [900, 1300, 1400],
    )

    assert metrics["MAE"] == 66.666667
    assert metrics["RMSE"] == 81.649658
    assert metrics["within_100"] == 1.0
    assert metrics["within_200"] == 1.0
    assert metrics["R2"] == 0.75


def test_feature_subset_selectors_are_strict() -> None:
    """Each baseline receives only its intended feature subset."""
    table = _model_table()
    feature_columns = _feature_columns()

    assert baselines.select_index_features(table, feature_columns) == [
        "index_letter",
        "index_number",
        "index_rank",
    ]
    assert baselines.select_solved_count_features(table, feature_columns) == [
        "solved_count",
        "log_solved_count",
        "solved_count_missing",
    ]
    assert baselines.select_tag_features(table, feature_columns) == [
        "tag_count",
        "tag__dp",
        "tag__math",
    ]
    assert baselines.select_no_features(table, feature_columns) == []


def test_run_baselines_writes_outputs(tmp_path: Path) -> None:
    """A small end-to-end baseline run writes metrics and predictions."""
    feature_path = tmp_path / "model_table.parquet"
    feature_columns_path = tmp_path / "feature_columns.json"
    contest_split_path = tmp_path / "contest_grouped_split.parquet"
    time_split_path = tmp_path / "forward_time_split.parquet"
    output_dir = tmp_path / "baselines"
    log_path = tmp_path / "logs" / "baselines.log"
    config_path = tmp_path / "experiment.yaml"

    _model_table().to_parquet(feature_path, engine="pyarrow", index=False)
    feature_columns_path.write_text(
        json.dumps({"feature_columns": _feature_columns()}),
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
        "project:\n  random_seed: 7\n",
        encoding="utf-8",
    )

    paths = baselines.run_baselines(
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
    metrics = pd.read_csv(paths["contest_grouped_metrics_csv"])
    assert len(metrics) == 21
    assert set(metrics["split_name"]) == {"train", "valid", "test"}
    assert set(metrics["model_name"]) == {
        "mean_baseline",
        "index_only_baseline",
        "tag_only_baseline",
        "solved_count_only_baseline",
        "ridge_regression",
        "random_forest_regressor",
        "hist_gradient_boosting_regressor",
    }
    predictions = pd.read_parquet(paths["contest_grouped_predictions"])
    required_prediction_columns = {
        "contest_id",
        "index",
        "name",
        "start_time_seconds",
        "actual_rating",
        "predicted_rating",
        "model_name",
        "split_name",
    }
    assert required_prediction_columns <= set(predictions.columns)
    assert len(predictions) == 42
