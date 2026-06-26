"""Tests for baseline result analysis helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import analysis


def _metrics_frame() -> pd.DataFrame:
    """Return synthetic metrics where solved-count beats index/tag."""
    rows = []
    test_mae = {
        "mean_baseline": 500.0,
        "index_only_baseline": 410.0,
        "tag_only_baseline": 390.0,
        "solved_count_only_baseline": 240.0,
        "ridge_regression": 200.0,
        "random_forest_regressor": 230.0,
    }
    train_mae = {
        "mean_baseline": 490.0,
        "index_only_baseline": 395.0,
        "tag_only_baseline": 380.0,
        "solved_count_only_baseline": 220.0,
        "ridge_regression": 180.0,
        "random_forest_regressor": 80.0,
    }
    for strategy in ("contest_grouped", "forward_time"):
        for model_name, mae in test_mae.items():
            for split_name, split_mae in (
                ("train", train_mae[model_name]),
                ("valid", mae + 10.0),
                ("test", mae),
            ):
                rows.append(
                    {
                        "strategy": strategy,
                        "model_name": model_name,
                        "split_name": split_name,
                        "row_count": 10,
                        "feature_count": 3,
                        "MAE": split_mae,
                        "RMSE": split_mae + 20.0,
                        "R2": 0.5,
                        "within_100": 0.2,
                        "within_200": 0.6,
                    }
                )
    return pd.DataFrame(rows)


def _predictions_frame() -> pd.DataFrame:
    """Return synthetic predictions for top-error extraction."""
    return pd.DataFrame(
        {
            "contest_id": [1, 1, 2],
            "index": ["A", "B", "A"],
            "name": ["Easy", "Medium", "Hard"],
            "start_time_seconds": [1000, 1000, 2000],
            "actual_rating": [800, 1200, 2000],
            "predicted_rating": [900, 1800, 1500],
            "model_name": ["ridge_regression"] * 3,
            "split_name": ["test"] * 3,
        }
    )


def _feature_frame() -> pd.DataFrame:
    """Return synthetic feature metadata for enriching predictions."""
    return pd.DataFrame(
        {
            "contest_id": [1, 1, 2],
            "index": ["A", "B", "A"],
            "solved_count": [1000, 200, 10],
            "log_solved_count": [6.9, 5.3, 2.4],
            "tag_count": [1, 2, 1],
            "index_rank": [1, 2, 1],
            "tag__math": [1, 0, 1],
            "tag__dp": [0, 1, 0],
        }
    )


def test_model_ranking_uses_test_mae_only() -> None:
    """Ranking ignores train/valid metrics and orders by test MAE."""
    ranking = analysis.build_model_ranking(_metrics_frame())
    contest = ranking.loc[ranking["strategy"].eq("contest_grouped")]

    assert contest.iloc[0]["model_name"] == "ridge_regression"
    assert contest.iloc[0]["rank_by_MAE"] == 1
    solved = contest.loc[
        contest["model_name"].eq("solved_count_only_baseline")
    ].iloc[0]
    index = contest.loc[contest["model_name"].eq("index_only_baseline")].iloc[0]
    tag = contest.loc[contest["model_name"].eq("tag_only_baseline")].iloc[0]
    assert solved["MAE"] < index["MAE"]
    assert solved["MAE"] < tag["MAE"]


def test_baseline_improvement_calculation() -> None:
    """Best full model improvement is computed from test MAE values."""
    ranking = analysis.build_model_ranking(_metrics_frame())
    improvements = analysis.build_baseline_improvements(ranking)
    row = improvements.loc[
        improvements["strategy"].eq("contest_grouped")
        & improvements["comparison_model"].eq("solved_count_only_baseline")
    ].iloc[0]

    assert row["best_full_model"] == "ridge_regression"
    assert row["absolute_MAE_improvement"] == 40.0
    assert round(row["percent_MAE_improvement"], 6) == 16.666667


def test_top_error_extraction_enriches_features_and_sorts() -> None:
    """Top-error table joins feature context and sorts by absolute error."""
    enriched = analysis.enrich_predictions(
        _predictions_frame(),
        _feature_frame(),
        {
            "tag_feature_map": {
                "math": "tag__math",
                "dp": "tag__dp",
            }
        },
    )
    top_errors = analysis.build_top_error_cases(
        {"contest_grouped": enriched},
        {"contest_grouped": "ridge_regression"},
        top_n=2,
    )

    assert top_errors["abs_error"].tolist() == [600, 500]
    assert top_errors.iloc[0]["name"] == "Medium"
    assert top_errors.iloc[0]["rating"] == 1200
    assert top_errors.iloc[0]["prediction"] == 1800
    assert top_errors.iloc[0]["solved_count"] == 200
    assert top_errors.iloc[0]["tags"] == ["dp"]


def test_run_analysis_writes_required_artifacts(tmp_path: Path) -> None:
    """The analysis CLI backend writes required tables and figures."""
    metrics_dir = tmp_path / "metrics"
    predictions_dir = tmp_path / "predictions"
    output_dir = tmp_path / "analysis"
    metrics_dir.mkdir()
    predictions_dir.mkdir()
    feature_path = tmp_path / "model_table.parquet"
    feature_columns_path = tmp_path / "feature_columns.json"

    metrics = _metrics_frame()
    for strategy in ("contest_grouped", "forward_time"):
        metrics.loc[metrics["strategy"].eq(strategy)].to_csv(
            metrics_dir / f"{strategy}_metrics.csv",
            index=False,
        )
        _predictions_frame().to_parquet(
            predictions_dir / f"{strategy}_predictions.parquet",
            engine="pyarrow",
            index=False,
        )
    _feature_frame().to_parquet(feature_path, engine="pyarrow", index=False)
    feature_columns_path.write_text(
        (
            '{"tag_feature_map": {"math": "tag__math", "dp": "tag__dp"}, '
            '"feature_columns": ["tag__math", "tag__dp"]}'
        ),
        encoding="utf-8",
    )

    paths = analysis.run_analysis(
        config_path=tmp_path / "experiment.yaml",
        feature_path=feature_path,
        feature_columns_path=feature_columns_path,
        contest_split_path=tmp_path / "contest.parquet",
        time_split_path=tmp_path / "time.parquet",
        baseline_metrics_dir=metrics_dir,
        baseline_predictions_dir=predictions_dir,
        output_dir=output_dir,
        log_path=tmp_path / "analysis.log",
    )

    assert all(path.is_file() for path in paths.values())
    assert (output_dir / "tables" / "error_by_index_rank.csv").is_file()
    assert (output_dir / "figures" / "error_by_index_rank.png").is_file()


def test_forward_time_gap_is_not_labeled_overfitting() -> None:
    """Forward-time gaps are described as temporal generalization gaps."""
    metrics = _metrics_frame()
    ranking = analysis.build_model_ranking(metrics)
    improvements = analysis.build_baseline_improvements(ranking)
    train_test_gap = analysis.build_train_test_gap_summary(metrics)

    summary = analysis.build_analysis_summary(
        ranking,
        improvements,
        train_test_gap,
    )

    forward_notes = [
        note
        for note in summary["generalization_gap_notes"]
        if note.get("strategy") == "forward_time"
    ]
    assert forward_notes
    assert all("overfitting" not in note["note"] for note in forward_notes)
    assert any(
        note.get("gap_type") == "temporal_generalization_gap"
        for note in forward_notes
    )
    assert all(
        note.get("strategy") != "forward_time"
        for note in summary["overfitting_notes"]
    )
