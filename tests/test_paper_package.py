"""Tests for paper artifact packaging."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import paper_package


def _synthetic_json_results() -> dict[str, dict[str, object]]:
    """Return minimal JSON-like result objects for section generation."""
    return {
        "preprocess_summary": {
            "number_of_rated_programming_problems": 10979,
        },
        "feature_summary": {
            "feature_count": 46,
            "tag_feature_count": 37,
        },
        "feature_columns": {
            "feature_columns": ["index_rank", "solved_count"],
        },
        "split_summary": {
            "contest_grouped": {
                "contest_overlap_count": 0,
            },
            "forward_time": {
                "strictly_ordered": True,
            },
        },
        "dataset_summary": {
            "unique_contests": {"processed": 1948},
            "rating_range": {"min": 800, "max": 3500},
            "solved_count_quantiles": {
                "p50": 4167.0,
                "p75": 13659.5,
                "p90": 25146.8,
                "p95": 33251.6,
                "p99": 73912.26,
                "max": 700377.0,
            },
            "top_tags": [
                {"tag": "greedy", "count": 3485},
                {"tag": "math", "count": 3406},
            ],
        },
        "analysis_summary": {
            "best_model_by_strategy": {
                "contest_grouped": {
                    "model_name": "hist_gradient_boosting_regressor",
                    "test_MAE": 166.894671,
                    "within_200": 0.696677,
                },
                "forward_time": {
                    "model_name": "random_forest_regressor",
                    "test_MAE": 152.540918,
                    "within_200": 0.711620,
                },
            }
        },
        "ablation_summary": {
            "best_overall_ablation_by_test_MAE": {
                "strategy": "forward_time",
                "model_name": "hist_gradient_boosting_regressor",
                "feature_set_name": "all_api_features",
                "test_MAE": 153.016985,
            }
        },
    }


def _synthetic_csv_results() -> dict[str, pd.DataFrame]:
    """Return minimal CSV-like result tables for paper packaging."""
    ranking = pd.DataFrame(
        {
            "strategy": [
                "contest_grouped",
                "contest_grouped",
                "contest_grouped",
                "forward_time",
                "forward_time",
                "forward_time",
            ],
            "model_name": [
                "solved_count_only_baseline",
                "index_only_baseline",
                "tag_only_baseline",
                "solved_count_only_baseline",
                "index_only_baseline",
                "tag_only_baseline",
            ],
            "MAE": [274.4, 409.2, 482.9, 227.2, 461.2, 579.0],
            "RMSE": [1, 1, 1, 1, 1, 1],
            "R2": [0.5, 0.4, 0.3, 0.6, 0.4, 0.2],
            "within_100": [0.2, 0.1, 0.1, 0.3, 0.1, 0.1],
            "within_200": [0.4, 0.2, 0.2, 0.5, 0.2, 0.2],
            "rank_by_MAE": [1, 2, 3, 1, 2, 3],
        }
    )
    return {
        "model_ranking_test": ranking,
        "baseline_improvements": pd.DataFrame(
            {
                "strategy": ["contest_grouped"],
                "best_full_model": ["hist_gradient_boosting_regressor"],
                "comparison_model": ["solved_count_only_baseline"],
                "absolute_MAE_improvement": [107.5],
            }
        ),
        "ablation_drop_comparison": pd.DataFrame(
            {
                "strategy": ["contest_grouped", "contest_grouped"],
                "model_name": ["ridge_regression", "ridge_regression"],
                "removed_group": ["solved", "tags"],
                "MAE_difference": [151.0, 30.0],
            }
        ),
        "error_by_tag": pd.DataFrame(
            {
                "strategy": ["contest_grouped"],
                "tag": ["graphs"],
                "count": [40],
                "mean_abs_error": [180.0],
            }
        ),
        "error_by_index_rank": pd.DataFrame(
            {
                "strategy": ["contest_grouped"],
                "index_rank": [1],
                "count": [100],
                "mean_abs_error": [150.0],
            }
        ),
    }


def test_load_json_and_csv_result_files(tmp_path: Path) -> None:
    """JSON and CSV loaders read expected result shapes."""
    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "result.csv"
    json_path.write_text('{"rows": 3}', encoding="utf-8")
    pd.DataFrame({"a": [1, 2]}).to_csv(csv_path, index=False)

    json_results, csv_results = paper_package.load_result_files(
        {"sample_json": json_path},
        {"sample_csv": csv_path},
    )

    assert json_results["sample_json"] == {"rows": 3}
    assert csv_results["sample_csv"]["a"].tolist() == [1, 2]


def test_dataframe_to_markdown_formats_table() -> None:
    """Markdown table formatting is deterministic and pipe-safe."""
    frame = pd.DataFrame(
        {
            "metric": ["rating range", "note"],
            "value": ["800|3500", 1.23456],
        }
    )

    markdown = paper_package.dataframe_to_markdown(frame)

    assert "| metric | value |" in markdown
    assert "800\\|3500" in markdown
    assert "1.235" in markdown


def test_create_output_directories(tmp_path: Path) -> None:
    """Paper output directories are created automatically."""
    directories = paper_package.create_output_directories(
        tmp_path / "paper",
        tmp_path / "paper_tables",
    )

    assert directories["sections"].is_dir()
    assert directories["figures"].is_dir()
    assert directories["tables"].is_dir()
    assert directories["results_tables"].is_dir()


def test_generate_section_file_from_synthetic_inputs(tmp_path: Path) -> None:
    """Synthetic results can generate bilingual sections and combined papers."""
    directories = paper_package.create_output_directories(
        tmp_path / "paper",
        tmp_path / "paper_tables",
    )
    sections = paper_package.build_section_texts(
        _synthetic_json_results(),
        _synthetic_csv_results(),
        ["test_mae_by_model.png"],
    )

    paper_package.write_sections_and_papers(sections, directories)

    abstract = directories["sections"] / "01_abstract_en.md"
    assert abstract.is_file()
    assert "10,979 rated programming problems" in abstract.read_text(
        encoding="utf-8"
    )
    paper_en = directories["paper"] / "paper_en.md"
    paper_cn = directories["paper"] / "paper_cn.md"
    assert "# Abstract" in paper_en.read_text(encoding="utf-8")
    assert "# 摘要" in paper_cn.read_text(encoding="utf-8")


def test_build_paper_tables_writes_markdown_and_csv(tmp_path: Path) -> None:
    """Paper tables are emitted to paper and results-output directories."""
    directories = paper_package.create_output_directories(
        tmp_path / "paper",
        tmp_path / "paper_tables",
    )
    csv_results = _synthetic_csv_results()
    csv_results.update(
        {
            "main_results": pd.DataFrame(),
            "contest_grouped_metrics": pd.DataFrame(),
            "forward_time_metrics": pd.DataFrame(),
            "top_error_cases": pd.DataFrame(),
            "ablation_metrics_test": pd.DataFrame(),
            "feature_group_definitions": pd.DataFrame(),
        }
    )

    tables = paper_package.build_paper_tables(
        _synthetic_json_results(),
        csv_results,
        directories,
    )

    assert "dataset_summary" in tables
    assert (directories["tables"] / "dataset_summary.md").is_file()
    assert (directories["results_tables"] / "dataset_summary.csv").is_file()
