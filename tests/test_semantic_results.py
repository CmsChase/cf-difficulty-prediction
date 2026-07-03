"""Tests for v6 semantic TF-IDF results summarization."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import semantic_results


def _metrics_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = {
        "contest_grouped": {
            "metadata_only": 340.0,
            "text_light_only": 470.0,
            "tfidf_text_only": 450.0,
            "metadata_plus_text_light": 310.0,
            "metadata_plus_tfidf": 312.0,
            "metadata_plus_text_light_plus_tfidf": 298.0,
            "full_api_reference": 190.0,
        },
        "forward_time": {
            "metadata_only": 365.0,
            "text_light_only": 520.0,
            "tfidf_text_only": 525.0,
            "metadata_plus_text_light": 336.0,
            "metadata_plus_tfidf": 325.0,
            "metadata_plus_text_light_plus_tfidf": 316.0,
            "full_api_reference": 180.0,
        },
    }
    for strategy, setting_values in values.items():
        for setting, mae in setting_values.items():
            rows.append(
                {
                    "strategy": strategy,
                    "split_name": "test",
                    "feature_setting": setting,
                    "model_name": "ridge_regression",
                    "feature_count": 10,
                    "uses_tfidf": "tfidf" in setting,
                    "row_count": 100,
                    "MAE": mae,
                    "RMSE": mae + 40,
                    "R2": 0.5,
                    "within_100": 0.2,
                    "within_200": 0.4,
                }
            )
    return pd.DataFrame(rows)


def _summary_payload() -> dict[str, object]:
    return {
        "input_model_table_rows": 12,
        "matched_statement_text_rows": 12,
        "matched_statement_feature_rows": 12,
        "text_available_count": 11,
        "text_available_rate": 0.916667,
        "tfidf_max_features": 100,
        "tfidf_ngram_range": [1, 2],
    }


def test_loading_metrics_csv(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    _metrics_frame().to_csv(path, index=False)
    loaded = semantic_results.load_metrics_csv(path)
    assert len(loaded) == 14
    assert set(loaded["split_name"]) == {"test"}


def test_loading_summary_json(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_summary_payload()), encoding="utf-8")
    loaded = semantic_results.load_summary_json(path)
    assert loaded["text_available_count"] == 11


def test_improvement_calculation() -> None:
    improvements = semantic_results.compute_improvements(_metrics_frame())
    row = improvements.loc[
        (improvements["strategy"].eq("forward_time"))
        & improvements["baseline_setting"].eq("metadata_only")
        & improvements["comparison_setting"].eq("metadata_plus_tfidf")
    ].iloc[0]
    assert row["absolute_MAE_improvement"] == 40.0
    assert round(row["percent_MAE_improvement"], 6) == 10.958904


def test_markdown_generation_contains_key_findings() -> None:
    metrics = _metrics_frame()
    improvements = semantic_results.compute_improvements(metrics)
    markdown = semantic_results.render_markdown_summary(
        metrics,
        _summary_payload(),
        improvements,
    )
    assert "# v6 Semantic Statement Text Modeling Results" in markdown
    assert "TF-IDF alone is weak." in markdown
    assert "Metadata + TF-IDF improves over metadata only." in markdown


def test_markdown_generation_contains_conservative_limitations() -> None:
    metrics = _metrics_frame()
    improvements = semantic_results.compute_improvements(metrics)
    markdown = semantic_results.render_markdown_summary(
        metrics,
        _summary_payload(),
        improvements,
    )
    assert "TF-IDF is classical bag-of-words modeling" in markdown
    assert "ridge regression for comparison consistency" in markdown


def test_table_output_schema() -> None:
    table = semantic_results.build_results_table(_metrics_frame())
    expected = {
        "strategy",
        "feature_setting",
        "model_name",
        "MAE",
        "within_200",
    }
    assert expected.issubset(table.columns)


def test_key_findings_json_schema() -> None:
    metrics = _metrics_frame()
    improvements = semantic_results.compute_improvements(metrics)
    payload = semantic_results.build_key_findings_payload(
        metrics,
        _summary_payload(),
        improvements,
    )
    assert "research_question" in payload
    assert "key_findings" in payload
    assert "conservative_limitations" in payload
    assert "coverage" in payload
    assert "improvements" in payload


def test_figure_generation_smoke_test(tmp_path: Path) -> None:
    metrics = _metrics_frame()
    improvements = semantic_results.compute_improvements(metrics)
    mae_path = tmp_path / "mae.png"
    improvement_path = tmp_path / "improvement.png"
    semantic_results.plot_mae_comparison(metrics, mae_path)
    semantic_results.plot_improvement_comparison(improvements, improvement_path)
    assert mae_path.exists()
    assert mae_path.stat().st_size > 0
    assert improvement_path.exists()
    assert improvement_path.stat().st_size > 0


def test_cli_smoke_with_tiny_synthetic_files(tmp_path: Path) -> None:
    metrics_path = tmp_path / "semantic_tfidf_best_by_setting.csv"
    summary_path = tmp_path / "semantic_tfidf_summary.json"
    output_dir = tmp_path / "results_summary"
    log_path = tmp_path / "logs" / "semantic_results.log"
    _metrics_frame().to_csv(metrics_path, index=False)
    summary_path.write_text(json.dumps(_summary_payload()), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cf_diff.semantic_results",
            "--metrics-path",
            str(metrics_path),
            "--summary-path",
            str(summary_path),
            "--output-dir",
            str(output_dir),
            "--log-path",
            str(log_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "v6_semantic_results_summary.md").exists()
    assert (output_dir / "v6_semantic_results_table.csv").exists()
    assert (output_dir / "v6_semantic_improvements.csv").exists()
    assert (output_dir / "v6_semantic_key_findings.json").exists()
    assert (output_dir / "figures" / "v6_semantic_mae_comparison.png").exists()
    assert (output_dir / "figures" / "v6_semantic_improvement_comparison.png").exists()
    assert log_path.exists()
