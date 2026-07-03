"""Tests for v6 semantic TF-IDF cold-start experiments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import semantic_tfidf


def _model_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for contest_id in range(1, 7):
        for index, rank in (("A", 1), ("B", 2)):
            solved_count = 1000.0 / (contest_id + rank)
            rows.append(
                {
                    "contest_id": contest_id,
                    "index": index,
                    "name": f"Problem {contest_id}{index}",
                    "rating": float(800 + contest_id * 80 + rank * 130),
                    "start_time_seconds": contest_id * 1000,
                    "index_letter": index,
                    "index_number": 0.0,
                    "index_rank": float(rank),
                    "has_points": 1,
                    "points": 500.0 * rank,
                    "tag_count": 2.0,
                    "tag__dp": int(index == "A"),
                    "tag__graphs": int(index == "B"),
                    "solved_count": solved_count,
                    "log_solved_count": float(np.log1p(solved_count)),
                    "solved_count_missing": 0,
                    "accepted_count": solved_count + 1,
                }
            )
    return pd.DataFrame(rows)


def _statement_features() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in _model_table().iterrows():
        rows.append(
            {
                "contest_id": str(row["contest_id"]),
                "index": row["index"],
                "statement_available": 1,
                "statement_char_len": 1000 + int(row["contest_id"]),
                "statement_word_count": 180 + int(row["contest_id"]),
                "sample_count": 2,
                "number_count": 12,
                "math_symbol_count": 5,
                "constraint_keyword_count": 2,
                "time_limit_ms": 2000,
                "memory_limit_mb": 256,
            }
        )
    return pd.DataFrame(rows)


def _statement_text() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in _model_table().iterrows():
        available = not (row["contest_id"] == 6 and row["index"] == "B")
        text = (
            f"graph dynamic programming shortest path contest {row['contest_id']} {row['index']}"
            if available
            else ""
        )
        rows.append(
            {
                "contest_id": row["contest_id"],
                "index": row["index"],
                "text_extract_status": "parsed" if available else "missing_cache",
                "html_cache_found": available,
                "statement_text_available": available,
                "combined_text": text,
            }
        )
    return pd.DataFrame(rows)


def _split_assignment(strategy: str) -> pd.DataFrame:
    split_by_contest = {
        1: "train",
        2: "train",
        3: "train",
        4: "valid",
        5: "test",
        6: "test",
    }
    table = _model_table()
    return pd.DataFrame(
        {
            "contest_id": table["contest_id"],
            "index": table["index"],
            "split_name": table["contest_id"].map(split_by_contest),
            "fold": 0,
            "strategy": strategy,
        }
    )


def _joined_table() -> tuple[pd.DataFrame, list[str]]:
    model = _model_table()
    statement_features = _statement_features()
    statement_text = _statement_text()
    table, _ = semantic_tfidf._join_auxiliary_table(
        model,
        statement_features,
        columns=["contest_id", "index", "statement_char_len", "statement_word_count"],
        suffix="_statement_feature",
    )
    table, _ = semantic_tfidf._join_auxiliary_table(
        table,
        statement_text,
        columns=["contest_id", "index", "combined_text", "statement_text_available"],
        suffix="_statement_text",
    )
    return semantic_tfidf.prepare_text_column(table, "combined_text"), []


def test_text_availability_filtering() -> None:
    frame = pd.DataFrame({"combined_text": ["alpha beta", "", None]})
    summary = semantic_tfidf.text_availability_summary(frame, "combined_text")
    assert summary["text_available_count"] == 1
    assert summary["text_available_rate"] == 0.333333


def test_split_loading(tmp_path: Path) -> None:
    path = tmp_path / "split.parquet"
    _split_assignment("contest_grouped").to_parquet(path, engine="pyarrow", index=False)
    loaded = semantic_tfidf.load_split_assignment(path, "contest_grouped")
    assert set(loaded["split_name"]) == {"train", "valid", "test"}
    assert loaded["strategy"].eq("contest_grouped").all()


def test_leakage_feature_filtering_excludes_solved_related_columns() -> None:
    table, feature_columns = _joined_table()
    columns = semantic_tfidf.select_metadata_features(table, feature_columns)
    assert "solved_count" not in columns
    assert "log_solved_count" not in columns
    assert "accepted_count" not in columns
    assert "tag_count" in columns


def test_full_api_reference_keeps_solved_related_columns() -> None:
    table, feature_columns = _joined_table()
    columns = semantic_tfidf.select_full_api_reference_features(table, feature_columns)
    assert "solved_count" in columns
    assert "log_solved_count" in columns


def test_tfidf_pipeline_fits_only_on_train_data() -> None:
    train = pd.DataFrame({"combined_text": ["alpha train", "alpha only"]})
    test = pd.DataFrame({"combined_text": ["forbidden_token"]})
    pipeline = semantic_tfidf.build_semantic_pipeline(
        train,
        text_column="combined_text",
        use_tfidf=True,
        tfidf_config=semantic_tfidf.TfidfConfig(min_df=1, max_df=1.0, max_features=100),
    )
    pipeline.fit(train, [1000.0, 1200.0])
    vocabulary = pipeline.named_steps["features"].named_transformers_["tfidf"].vocabulary_
    assert "alpha" in vocabulary
    assert "forbidden_token" not in vocabulary
    pipeline.predict(test)


def test_metrics_calculation() -> None:
    metrics = semantic_tfidf.compute_regression_metrics([1000, 1200], [900, 1300])
    assert metrics["MAE"] == 100.0
    assert metrics["RMSE"] == 100.0
    assert metrics["within_100"] == 1.0
    assert metrics["within_200"] == 1.0


def test_improvement_calculation() -> None:
    metrics = pd.DataFrame(
        [
            {"strategy": "contest_grouped", "feature_setting": "metadata_only", "MAE": 300.0},
            {"strategy": "contest_grouped", "feature_setting": "metadata_plus_tfidf", "MAE": 250.0},
        ]
    )
    rows = semantic_tfidf.improvement_table(
        metrics,
        baseline_setting="metadata_only",
        comparison_setting="metadata_plus_tfidf",
    )
    assert rows[0]["absolute_MAE_improvement"] == 50.0
    assert rows[0]["percent_MAE_improvement"] == 16.666667


def test_output_table_schema() -> None:
    table, feature_columns = _joined_table()
    feature_sets = semantic_tfidf.build_feature_sets(table, feature_columns)
    metrics = semantic_tfidf.evaluate_strategy(
        table,
        _split_assignment("contest_grouped"),
        strategy="contest_grouped",
        feature_sets=feature_sets,
        text_column="combined_text",
        tfidf_config=semantic_tfidf.TfidfConfig(min_df=1, max_df=1.0, max_features=100),
    )
    expected = {
        "strategy",
        "split_name",
        "feature_setting",
        "model_name",
        "MAE",
        "RMSE",
        "R2",
        "within_100",
        "within_200",
    }
    assert expected.issubset(metrics.columns)
    assert set(metrics["feature_setting"]) == set(semantic_tfidf.FEATURE_SETTINGS)


def test_missing_statement_text_handling() -> None:
    frame = pd.DataFrame({"combined_text": [None], "statement_text_available": [None]})
    prepared = semantic_tfidf.prepare_text_column(frame, "combined_text")
    assert prepared.loc[0, "combined_text"] == ""
    assert prepared.loc[0, "statement_text_available"] is False


def test_cli_smoke_on_tiny_synthetic_parquet_files(tmp_path: Path) -> None:
    feature_path = tmp_path / "model_table.parquet"
    statement_feature_path = tmp_path / "statement_features.parquet"
    statement_text_path = tmp_path / "statement_text.parquet"
    contest_split_path = tmp_path / "contest_split.parquet"
    time_split_path = tmp_path / "time_split.parquet"
    output_dir = tmp_path / "semantic_tfidf"
    log_path = tmp_path / "logs" / "semantic_tfidf.log"

    _model_table().to_parquet(feature_path, engine="pyarrow", index=False)
    _statement_features().to_parquet(statement_feature_path, engine="pyarrow", index=False)
    _statement_text().to_parquet(statement_text_path, engine="pyarrow", index=False)
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

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cf_diff.semantic_tfidf",
            "--feature-path",
            str(feature_path),
            "--statement-feature-path",
            str(statement_feature_path),
            "--statement-text-path",
            str(statement_text_path),
            "--contest-split-path",
            str(contest_split_path),
            "--time-split-path",
            str(time_split_path),
            "--output-dir",
            str(output_dir),
            "--log-path",
            str(log_path),
            "--tfidf-min-df",
            "1",
            "--tfidf-max-df",
            "1.0",
            "--tfidf-max-features",
            "100",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "summary" / "semantic_tfidf_summary.json").exists()
    assert (output_dir / "tables" / "semantic_tfidf_metrics.csv").exists()
    assert (output_dir / "tables" / "semantic_tfidf_best_by_setting.csv").exists()
    assert (output_dir / "figures" / "semantic_tfidf_mae_by_setting.png").exists()
    prediction_files = list((output_dir / "predictions").glob("*.csv"))
    assert prediction_files
    summary = json.loads(
        (output_dir / "summary" / "semantic_tfidf_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["input_model_table_rows"] == 12
    assert summary["text_available_count"] == 11
