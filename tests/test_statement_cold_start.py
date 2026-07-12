"""Tests for statement text-light cold-start experiments."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import statement_cold_start


def _model_table() -> pd.DataFrame:
    """Return a small model table with metadata and solved features."""
    rows = []
    for contest_id in range(1, 7):
        for index, rank, tag_dp, tag_math, points in (
            ("A", 1, 0, 1, 500.0),
            ("B", 2, 1, 0, None),
        ):
            solved_count = 1000 // (contest_id + rank)
            rows.append(
                {
                    "contest_id": contest_id,
                    "index": index,
                    "name": f"Problem {contest_id}{index}",
                    "rating": 800 + contest_id * 90 + rank * 120,
                    "start_time_seconds": contest_id * 1000,
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


def _statement_features(include_unmatched: bool = False) -> pd.DataFrame:
    """Return statement features with one missing-statement row."""
    rows = []
    for contest_id in range(1, 7):
        for index in ("A", "B"):
            parsed = not (contest_id == 6 and index == "B")
            rows.append(
                {
                    "contest_id": str(contest_id),
                    "index": index,
                    "name": f"Statement {contest_id}{index}",
                    "url": f"https://example.test/{contest_id}/{index}",
                    "statement_fetch_status": "cached",
                    "statement_parse_status": (
                        "parsed" if parsed else "missing_statement"
                    ),
                    "statement_error": (
                        "" if parsed else "problem-statement block was not found"
                    ),
                    "statement_available": int(parsed),
                    "statement_char_len": 100 + contest_id if parsed else pd.NA,
                    "statement_word_count": 20 + contest_id if parsed else pd.NA,
                    "statement_line_count": 3 if parsed else pd.NA,
                    "statement_paragraph_count": 2 if parsed else pd.NA,
                    "input_section_char_len": 10 if parsed else pd.NA,
                    "output_section_char_len": 8 if parsed else pd.NA,
                    "note_section_char_len": 4 if parsed else pd.NA,
                    "has_input_section": int(parsed),
                    "has_output_section": int(parsed),
                    "has_note_section": int(parsed),
                    "sample_count": 1 if parsed else pd.NA,
                    "sample_input_output_block_count": 2 if parsed else pd.NA,
                    "number_count": contest_id if parsed else pd.NA,
                    "integer_count": contest_id if parsed else pd.NA,
                    "float_count": 0 if parsed else pd.NA,
                    "inequality_symbol_count": 1 if parsed else pd.NA,
                    "math_symbol_count": 2 if parsed else pd.NA,
                    "constraint_keyword_count": 1 if parsed else pd.NA,
                    "big_o_like_count": 1 if parsed else pd.NA,
                    "uppercase_token_count": 0 if parsed else pd.NA,
                    "single_letter_variable_count": 2 if parsed else pd.NA,
                    "latex_like_token_count": 0 if parsed else pd.NA,
                    "code_like_token_count": 1 if parsed else pd.NA,
                    "kw_graph": int(index == "A" and parsed),
                    "kw_tree": int(index == "B" and parsed),
                    "kw_array": 1 if parsed else pd.NA,
                    "kw_string": 0 if parsed else pd.NA,
                    "kw_dp": 0 if parsed else pd.NA,
                    "kw_geometry": 0 if parsed else pd.NA,
                    "kw_greedy": 0 if parsed else pd.NA,
                    "kw_probability": 0 if parsed else pd.NA,
                    "kw_interactive": 0 if parsed else pd.NA,
                    "kw_permutation": 0 if parsed else pd.NA,
                    "kw_binary": 0 if parsed else pd.NA,
                    "kw_shortest_path": 0 if parsed else pd.NA,
                    "kw_query": 1 if parsed else pd.NA,
                    "time_limit_ms": 1000 if parsed else pd.NA,
                    "memory_limit_mb": 256 if parsed else pd.NA,
                    "has_time_limit": int(parsed),
                    "has_memory_limit": int(parsed),
                }
            )
    if include_unmatched:
        extra = dict(rows[0])
        extra["contest_id"] = "999"
        extra["index"] = "Z"
        rows.append(extra)
    return pd.DataFrame(rows)


def _feature_columns() -> list[str]:
    """Return original API feature columns."""
    return [
        "index_letter",
        "index_number",
        "index_rank",
        "has_points",
        "points",
        "tag_count",
        "tag__dp",
        "tag__math",
        "solved_count",
        "solved_count_missing",
        "log_solved_count",
    ]


def _split_assignment(strategy: str) -> pd.DataFrame:
    """Return a split assignment with train, valid, and test rows."""
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


def test_joining_model_table_and_statement_features() -> None:
    """Statement features join by contest_id and index while preserving rows."""
    joined, counts = statement_cold_start.join_statement_features(
        _model_table(),
        _statement_features(include_unmatched=True),
    )

    assert len(joined) == len(_model_table())
    assert counts["matched_rows"] == len(_model_table())
    assert counts["unmatched_model_rows"] == 0
    assert counts["unmatched_statement_rows"] == 1
    assert "statement_word_count" in joined.columns


def test_preserving_rows_with_missing_statement_features() -> None:
    """Rows remain available when statement parse status is missing_statement."""
    joined, _ = statement_cold_start.join_statement_features(
        _model_table(),
        _statement_features(),
    )
    missing = joined.loc[
        joined["statement_parse_status"].eq("missing_statement")
    ]

    assert len(missing) == 1
    assert missing.iloc[0]["contest_id"] == 6
    assert missing.iloc[0]["index"] == "B"


def test_metadata_only_features_without_solved_leakage() -> None:
    """Cold-start metadata features exclude solved-count behavior."""
    joined, _ = statement_cold_start.join_statement_features(
        _model_table(),
        _statement_features(),
    )
    columns = statement_cold_start.select_metadata_features(
        joined,
        _feature_columns(),
    )

    assert "tag_count" in columns
    assert "has_points" in columns
    assert not any(statement_cold_start.has_solved_leakage(column) for column in columns)


def test_text_light_only_excludes_identifier_status_error_columns() -> None:
    """Text-light feature selector keeps numeric statement features only."""
    joined, _ = statement_cold_start.join_statement_features(
        _model_table(),
        _statement_features(),
    )
    columns = statement_cold_start.select_text_light_features(joined)

    assert "statement_word_count" in columns
    assert "sample_count" in columns
    assert "kw_graph" in columns
    assert "has_points" not in columns
    assert "contest_id" not in columns
    assert "index" not in columns
    assert "statement_parse_status" not in columns
    assert "statement_error" not in columns


def test_metadata_plus_text_light_features_without_leakage() -> None:
    """Combined cold-start features exclude solved-count leakage."""
    joined, _ = statement_cold_start.join_statement_features(
        _model_table(),
        _statement_features(),
    )
    feature_sets = statement_cold_start.build_feature_sets(joined, _feature_columns())
    columns = feature_sets["metadata_plus_text_light"]

    assert "statement_word_count" in columns
    assert "tag_count" in columns
    assert not any(statement_cold_start.has_solved_leakage(column) for column in columns)


def test_full_api_reference_allows_solved_features() -> None:
    """Full API reference uses original API features including solved count."""
    joined, _ = statement_cold_start.join_statement_features(
        _model_table(),
        _statement_features(),
    )
    columns = statement_cold_start.select_full_api_reference_features(
        joined,
        _feature_columns(),
    )

    assert "solved_count" in columns
    assert "log_solved_count" in columns


def test_metric_calculation() -> None:
    """Shared metrics are exposed for statement cold-start experiments."""
    metrics = statement_cold_start.compute_regression_metrics(
        [1000, 1200, 1400],
        [900, 1200, 1600],
    )

    assert metrics["MAE"] == 100.0
    assert metrics["RMSE"] == 129.099445
    assert metrics["within_100"] == 0.666667


def test_best_by_setting_summary_generation() -> None:
    """Best-by-setting uses validation even when test favors another model."""
    metrics = pd.DataFrame(
        {
            "strategy": ["contest_grouped"] * 4,
            "split_name": ["valid", "valid", "test", "test"],
            "feature_setting": ["metadata_only"] * 4,
            "model_name": [
                "ridge_regression",
                "hist_gradient_boosting_regressor",
                "ridge_regression",
                "hist_gradient_boosting_regressor",
            ],
            "MAE": [240.0, 260.0, 300.0, 250.0],
            "RMSE": [1.0, 1.0, 1.0, 1.0],
            "R2": [0.0, 0.0, 0.0, 0.0],
            "within_100": [0.0, 0.0, 0.0, 0.0],
            "within_200": [0.0, 0.0, 0.0, 0.0],
            "row_count": [10, 10, 10, 10],
            "feature_count": [3, 3, 3, 3],
            "is_cold_start": [True, True, True, True],
        }
    )

    best = statement_cold_start.build_best_by_setting(metrics)

    assert len(best) == 1
    assert best.iloc[0]["model_name"] == "ridge_regression"
    assert best.iloc[0]["validation_MAE"] == 240.0
    assert best.iloc[0]["MAE"] == 300.0


def test_cli_smoke_with_missing_statement_row(tmp_path: Path) -> None:
    """Tiny smoke test writes all required statement cold-start outputs."""
    feature_path = tmp_path / "model_table.parquet"
    feature_columns_path = tmp_path / "feature_columns.json"
    statement_path = tmp_path / "statement_features.parquet"
    contest_split_path = tmp_path / "contest_grouped_split.parquet"
    time_split_path = tmp_path / "forward_time_split.parquet"
    output_dir = tmp_path / "statement_cold_start"
    log_path = tmp_path / "logs" / "statement_cold_start.log"
    config_path = tmp_path / "experiment.yaml"

    _model_table().to_parquet(feature_path, engine="pyarrow", index=False)
    _statement_features().to_parquet(statement_path, engine="pyarrow", index=False)
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
        (PROJECT_ROOT / "configs" / "experiment.yaml")
        .read_text(encoding="utf-8")
        .replace("random_seed: 42", "random_seed: 7"),
        encoding="utf-8",
    )

    paths = statement_cold_start.run_statement_cold_start(
        config_path=config_path,
        feature_path=feature_path,
        feature_columns_path=feature_columns_path,
        statement_feature_path=statement_path,
        contest_split_path=contest_split_path,
        time_split_path=time_split_path,
        output_dir=output_dir,
        log_path=log_path,
    )

    assert all(path.is_file() for path in paths.values())
    assert log_path.is_file()
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["input_model_table_rows"] == len(_model_table())
    assert summary["statement_missing_count"] == 1
    assert "full_api_reference" in summary["feature_counts"]
