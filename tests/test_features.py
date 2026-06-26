"""Tests for lightweight Codeforces feature extraction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import features


def _source_frame() -> pd.DataFrame:
    """Return a small preprocessed problem fixture."""
    return pd.DataFrame(
        {
            "contest_id": pd.Series([2, 1, 3], dtype="Int64"),
            "index": ["AA12", "A", "B2"],
            "name": ["Advanced", "Intro", "Middle"],
            "rating": pd.Series([1600, 800, 1200], dtype="Int64"),
            "start_time_seconds": pd.Series(
                [2000, 1000, 3000],
                dtype="Int64",
            ),
            "points": [None, 500.0, 750.0],
            "tags": [
                ["graphs", "math"],
                ["math", "implementation"],
                ["graphs"],
            ],
            "solved_count": [None, 100, 20],
        }
    )


def test_feature_derivation_and_tag_encoding() -> None:
    """Index features, missing handling, and tag indicators are correct."""
    config = features.ExperimentConfig(min_tag_frequency=2)
    model, metadata = features.build_model_table(_source_frame(), config)

    assert model["index"].tolist() == ["A", "AA12", "B2"]
    assert model["index_letter"].tolist() == ["A", "AA", "B"]
    assert model["index_number"].tolist() == [0, 12, 2]
    assert model["index_rank"].tolist() == [1, 1, 2]
    assert model["has_points"].tolist() == [1, 0, 1]
    assert model["points"].tolist() == [500.0, 0.0, 750.0]
    assert model["solved_count"].tolist() == [100, 0, 20]
    assert model["solved_count_missing"].tolist() == [0, 1, 0]
    assert model["tag__graphs"].tolist() == [0, 1, 1]
    assert model["tag__math"].tolist() == [1, 1, 0]
    assert "tag__implementation" not in model.columns
    assert metadata["target_column"] == "rating"


def test_feature_outputs_and_config_file(
    tmp_path: Path,
) -> None:
    """The feature CLI artifacts are written with auditable metadata."""
    input_path = tmp_path / "rated.parquet"
    output_dir = tmp_path / "features"
    config_path = tmp_path / "experiment.yaml"
    log_path = tmp_path / "features.log"
    _source_frame().to_parquet(input_path, engine="pyarrow", index=False)
    config_path.write_text(
        "\n".join(
            [
                "random_seed: 7",
                "include_points: false",
                "include_tags: true",
                "min_tag_frequency: 2",
                "grouped_split:",
                "  train: 0.6",
                "  valid: 0.2",
                "  test: 0.2",
                "forward_time_split:",
                "  train: 0.6",
                "  valid: 0.2",
                "  test: 0.2",
            ]
        ),
        encoding="utf-8",
    )

    paths = features.generate_features(
        input_path,
        output_dir,
        config_path=config_path,
        log_path=log_path,
    )

    assert all(path.is_file() for path in paths.values())
    model = pd.read_parquet(paths["model_table"])
    assert {"contest_id", "rating", "index_rank", "tag__math"} <= set(
        model.columns
    )
    columns = json.loads(
        paths["feature_columns"].read_text(encoding="utf-8")
    )
    assert "points" not in columns["feature_columns"]
    assert columns["config"]["random_seed"] == 7
    summary = json.loads(
        paths["feature_summary"].read_text(encoding="utf-8")
    )
    assert summary["row_count"] == 3
    assert summary["tag_feature_count"] == 2
    assert log_path.is_file()
