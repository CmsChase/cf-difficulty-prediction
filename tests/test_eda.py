"""Tests for Codeforces exploratory data analysis outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import eda


def _processed_frame() -> pd.DataFrame:
    """Return a small processed problem fixture."""
    return pd.DataFrame(
        {
            "contest_id": pd.Series([1, 1, 2, 3], dtype="Int64"),
            "index": ["A", "B", "C1", "A2"],
            "name": ["Intro", "Middle", "Hard", "Variant"],
            "type": ["PROGRAMMING"] * 4,
            "points": [500.0, None, 1000.0, None],
            "rating": pd.Series([800, 1200, 1600, 1000], dtype="Int64"),
            "tags": [
                ["math", "implementation"],
                ["math"],
                ["graphs", "dp"],
                ["implementation"],
            ],
            "solved_count": pd.Series([1000, 100, 10, 500], dtype="Int64"),
            "start_time_seconds": pd.Series(
                [1000, 1000, 2000, 3000],
                dtype="Int64",
            ),
        }
    )


def _feature_frame() -> pd.DataFrame:
    """Return a small model table fixture."""
    frame = _processed_frame().loc[
        :,
        [
            "contest_id",
            "index",
            "name",
            "rating",
            "start_time_seconds",
            "points",
            "solved_count",
        ],
    ].copy()
    frame["index_letter"] = ["A", "B", "C", "A"]
    frame["index_rank"] = pd.Series([1, 2, 3, 1], dtype="Int64")
    frame["log_solved_count"] = [6.908755, 4.615121, 2.397895, 6.216606]
    return frame


def _split_frame(strategy: str) -> pd.DataFrame:
    """Return one row-level split assignment fixture."""
    return pd.DataFrame(
        {
            "contest_id": pd.Series([1, 1, 2, 3], dtype="Int64"),
            "index": ["A", "B", "C1", "A2"],
            "split_name": ["train", "train", "valid", "test"],
            "fold": pd.Series([0, 0, 1, 2], dtype="Int8"),
            "strategy": [strategy] * 4,
        }
    )


def test_dataset_summary_contains_required_counts() -> None:
    """Summary metadata captures counts, ratings, top tags, and splits."""
    processed = _processed_frame()
    features = _feature_frame()
    contest_split = _split_frame("contest_grouped")
    time_split = _split_frame("forward_time")
    tag_frequency = eda.build_tag_frequency(processed)

    summary = eda.build_dataset_summary(
        processed,
        features,
        contest_split,
        time_split,
        tag_frequency,
        config_path=Path("configs/experiment.yaml"),
        processed_path=Path("processed.parquet"),
        feature_path=Path("features.parquet"),
        contest_split_path=Path("contest.parquet"),
        time_split_path=Path("time.parquet"),
    )

    assert summary["row_counts"]["processed"] == 4
    assert summary["row_counts"]["feature_table"] == 4
    assert summary["rating_range"] == {"min": 800, "max": 1600}
    assert summary["points_missing_count"] == 2
    assert summary["solved_count_quantiles"] == {
        "p50": 300.0,
        "p75": 625.0,
        "p90": 850.0,
        "p95": 925.0,
        "p99": 985.0,
        "max": 1000.0,
    }
    assert summary["top_tags"][0] == {"tag": "implementation", "count": 2}
    assert summary["split_sizes"]["contest_grouped"]["rows"] == {
        "train": 2,
        "valid": 1,
        "test": 1,
    }


def test_prepare_top_tags_plot_data_is_sorted_and_limited() -> None:
    """Plot-preparation returns a deterministic top-N bar order."""
    tag_frequency = pd.DataFrame(
        {
            "tag": ["dp", "math", "graphs", "implementation"],
            "count": [2, 3, 1, 3],
            "problem_share": [0.5, 0.75, 0.25, 0.75],
        }
    )

    plot_data = eda.prepare_top_tags_plot_data(tag_frequency, top_n=3)

    assert plot_data["tag"].tolist() == ["dp", "math", "implementation"]
    assert plot_data["count"].tolist() == [2, 3, 3]


def test_run_eda_writes_required_outputs(tmp_path: Path) -> None:
    """The full EDA pipeline writes summaries, figures, and logs."""
    processed_path = tmp_path / "rated.parquet"
    feature_path = tmp_path / "model_table.parquet"
    contest_split_path = tmp_path / "contest_grouped_split.parquet"
    time_split_path = tmp_path / "forward_time_split.parquet"
    output_dir = tmp_path / "eda"
    log_path = tmp_path / "logs" / "eda.log"

    _processed_frame().to_parquet(processed_path, engine="pyarrow", index=False)
    _feature_frame().to_parquet(feature_path, engine="pyarrow", index=False)
    _split_frame("contest_grouped").to_parquet(
        contest_split_path,
        engine="pyarrow",
        index=False,
    )
    _split_frame("forward_time").to_parquet(
        time_split_path,
        engine="pyarrow",
        index=False,
    )

    paths = eda.run_eda(
        config_path=tmp_path / "experiment.yaml",
        processed_path=processed_path,
        feature_path=feature_path,
        contest_split_path=contest_split_path,
        time_split_path=time_split_path,
        output_dir=output_dir,
        log_path=log_path,
    )

    assert all(path.is_file() for path in paths.values())
    assert all(path.stat().st_size > 0 for path in paths.values())
    assert log_path.is_file()
    summary = json.loads(paths["dataset_summary"].read_text(encoding="utf-8"))
    assert summary["row_counts"]["processed"] == 4
    assert summary["top_tags"]
    assert summary["solved_count_quantiles"]["p99"] == 985.0
    assert paths["solved_count_hist_log"].is_file()
    assert paths["solved_count_hist_p99"].is_file()
