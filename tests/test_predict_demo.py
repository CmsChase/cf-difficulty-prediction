from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import predict_demo


def _synthetic_model_table(row_count: int = 18) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    indices = ["A", "B", "C", "D", "E", "F"]
    for idx in range(row_count):
        index = indices[idx % len(indices)]
        index_rank = float(indices.index(index) + 1)
        solved_count = float(9000 - idx * 320)
        rows.append(
            {
                "contest_id": 1000 + idx // 3,
                "index": index,
                "name": f"Problem {idx}",
                "rating": float(900 + index_rank * 260 + idx * 15),
                "start_time_seconds": 1_600_000_000 + idx * 1000,
                "index_letter": index,
                "index_number": 0.0,
                "index_rank": index_rank,
                "has_points": 1,
                "points": float(500 * index_rank),
                "tag_count": 2.0,
                "tag__dp": int(idx % 2 == 0),
                "tag__graphs": int(idx % 3 == 0),
                "tag__binary_search": int(idx % 4 == 0),
                "tag__shortest_paths": int(idx % 5 == 0),
                "tag__math": int(idx % 2 == 1),
                "tag__greedy": int(idx % 3 == 1),
                "solved_count": solved_count,
                "log_solved_count": float(np.log1p(solved_count)),
                "solved_count_missing": 0,
                "tags": ["dp", "graphs"] if idx % 2 == 0 else ["math"],
            }
        )
    return pd.DataFrame(rows)


def _synthetic_statement_features(model_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, row in model_table.iterrows():
        rows.append(
            {
                "contest_id": row["contest_id"],
                "index": row["index"],
                "name": row["name"],
                "statement_available": 1,
                "statement_char_len": 1000 + idx * 50,
                "statement_word_count": 180 + idx * 5,
                "statement_line_count": 20 + idx,
                "statement_paragraph_count": 6,
                "sample_count": 2 + idx % 3,
                "number_count": 15 + idx,
                "math_symbol_count": 8 + idx,
                "constraint_keyword_count": 3 + idx % 2,
                "time_limit_ms": 2000,
                "memory_limit_mb": 256,
                "has_time_limit": 1,
                "has_memory_limit": 1,
            }
        )
    return pd.DataFrame(rows)


def test_tag_normalization() -> None:
    assert predict_demo.normalize_tag_to_column("binary search") == "tag__binary_search"
    assert predict_demo.normalize_tag_to_column("binary-search") == "tag__binary_search"
    assert predict_demo.normalize_tag_to_column("binary_search") == "tag__binary_search"
    assert (
        predict_demo.normalize_tag_to_column("shortest paths")
        == "tag__shortest_paths"
    )
    assert predict_demo.normalize_tag_to_column("2-sat") == "tag__2_sat"


def test_manual_row_construction() -> None:
    table = _synthetic_model_table()
    row = predict_demo.build_manual_feature_row(
        table,
        problem_index="D2",
        tags=["dp", "graphs"],
        points=2000,
        solved_count=4200,
        statement_values={"statement_char_len": 2600, "sample_count": 3},
    )
    assert row["index"] == "D2"
    assert row["index_letter"] == "D"
    assert row["index_number"] == 2.0
    assert row["index_rank"] == 4.0
    assert row["tag__dp"] == 1
    assert row["tag__graphs"] == 1
    assert row["points"] == 2000
    assert row["solved_count"] == 4200


def test_known_problem_selection() -> None:
    table = _synthetic_model_table()
    row = predict_demo.select_known_problem(table, 1000, "A")
    assert row["contest_id"] == 1000
    assert row["index"] == "A"


def test_excluding_target_problem_from_training() -> None:
    table = _synthetic_model_table()
    training = predict_demo.exclude_known_problem_from_training(table, 1000, "A")
    assert len(training) == len(table) - 1
    assert not (
        (training["contest_id"].astype(str) == "1000")
        & (training["index"].str.upper() == "A")
    ).any()


def test_cold_start_feature_selection_excludes_solved_leakage() -> None:
    table = _synthetic_model_table()
    columns = predict_demo.select_features_for_scenario(
        table,
        "cold_start_metadata",
        [],
    )
    assert columns
    assert all("solved" not in column.lower() for column in columns)


def test_post_publication_feature_selection_allows_solved_features() -> None:
    table = _synthetic_model_table()
    columns = predict_demo.select_features_for_scenario(
        table,
        "post_publication_reference",
        [],
    )
    assert "solved_count" in columns
    assert "log_solved_count" in columns


def test_prediction_range_calculation_from_residuals() -> None:
    summary = predict_demo.ResidualSummary(
        residual_mae_validation=120.0,
        residual_q80_validation=180.0,
        range_method="validation_residual_q80",
    )
    low, high, method = predict_demo.prediction_range_from_residual(1760.0, summary)
    assert (low, high, method) == (1600, 1900, "validation_residual_q80")


def test_fallback_prediction_range_behavior() -> None:
    summary = predict_demo.ResidualSummary(
        residual_mae_validation=None,
        residual_q80_validation=None,
        range_method="fallback_pm250_not_enough_validation_rows",
    )
    low, high, method = predict_demo.prediction_range_from_residual(1760.0, summary)
    assert (low, high) == (1500, 2000)
    assert method.startswith("fallback_pm250")


def test_output_formatting_as_table_ready_records() -> None:
    records = [
        {
            "scenario": "cold_start_metadata",
            "model": "ridge",
            "predicted_rating": 1500.0,
            "predicted_rating_rounded": 1500,
            "tags": ["dp"],
        }
    ]
    frame = predict_demo.records_to_frame(records)
    assert list(frame["scenario"]) == ["cold_start_metadata"]
    assert isinstance(frame.loc[0, "tags"], list)


def test_csv_output(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    records = [{"scenario": "cold_start_metadata", "predicted_rating": 1500.0}]
    predict_demo.write_records(records, path, "table")
    loaded = pd.read_csv(path)
    assert loaded.loc[0, "scenario"] == "cold_start_metadata"


def test_json_output(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    records = [{"scenario": "cold_start_metadata", "predicted_rating": 1500.0}]
    predict_demo.write_records(records, path, "json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded[0]["predicted_rating"] == 1500.0


def test_cli_smoke_with_tiny_synthetic_parquet(tmp_path: Path) -> None:
    model_table = _synthetic_model_table(12)
    statement_features = _synthetic_statement_features(model_table)
    feature_path = tmp_path / "model_table.parquet"
    statement_path = tmp_path / "statement_features.parquet"
    output_path = tmp_path / "demo.json"
    model_table.to_parquet(feature_path, engine="pyarrow", index=False)
    statement_features.to_parquet(statement_path, engine="pyarrow", index=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cf_diff.predict_demo",
            "--feature-path",
            str(feature_path),
            "--statement-feature-path",
            str(statement_path),
            "--manual",
            "--problem-index",
            "C",
            "--tags",
            "dp",
            "graphs",
            "--statement-char-len",
            "2600",
            "--sample-count",
            "3",
            "--model",
            "ridge",
            "--output-path",
            str(output_path),
            "--format",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    records = json.loads(output_path.read_text(encoding="utf-8"))
    scenarios = {record["scenario"] for record in records}
    assert "cold_start_metadata" in scenarios
    assert "cold_start_metadata_plus_text_light" in scenarios
