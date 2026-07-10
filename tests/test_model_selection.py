"""Tests for validation-only model selection governance."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import model_selection


def _metrics() -> pd.DataFrame:
    rows = []
    for split_name, values in {
        "valid": {"model_a": 100.0, "model_b": 120.0},
        "test": {"model_a": 140.0, "model_b": 90.0},
    }.items():
        for model_name, mae in values.items():
            rows.append(
                {
                    "strategy": "forward_time",
                    "model_name": model_name,
                    "split_name": split_name,
                    "MAE": mae,
                    "RMSE": mae + 10.0,
                    "R2": 0.5,
                    "within_100": 0.4,
                    "within_200": 0.7,
                }
            )
    return pd.DataFrame(rows)


def test_validation_ranking_ignores_better_test_score() -> None:
    report = model_selection.build_validation_ranked_report(
        _metrics(),
        group_columns=("strategy",),
        candidate_columns=("model_name",),
    )
    selected = model_selection.select_rank_one(report).iloc[0]
    assert selected["model_name"] == "model_a"
    assert selected["validation_MAE"] == 100.0
    assert selected["MAE"] == 140.0
    assert selected["selection_split"] == "valid"
    assert selected["report_split"] == "test"


def test_missing_test_candidate_is_rejected() -> None:
    metrics = _metrics()
    incomplete = metrics.loc[
        ~(
            metrics["split_name"].eq("test")
            & metrics["model_name"].eq("model_b")
        )
    ]
    with pytest.raises(model_selection.ModelSelectionError, match="candidates differ"):
        model_selection.build_validation_ranked_report(
            incomplete,
            group_columns=("strategy",),
            candidate_columns=("model_name",),
        )


def test_tied_validation_scores_use_stable_candidate_order() -> None:
    """A declared lexical tiebreak prevents run-order-dependent winners."""
    metrics = _metrics()
    metrics.loc[metrics["split_name"].eq("valid"), "MAE"] = 100.0
    report = model_selection.build_validation_ranked_report(
        metrics.sample(frac=1.0, random_state=7),
        group_columns=("strategy",),
        candidate_columns=("model_name",),
    )
    assert model_selection.select_rank_one(report).iloc[0]["model_name"] == "model_a"


def test_duplicate_candidate_row_is_rejected() -> None:
    """A non-unique validation key cannot silently influence ranking."""
    metrics = _metrics()
    duplicate = pd.concat(
        [metrics, metrics.iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(model_selection.ModelSelectionError, match="duplicate"):
        model_selection.build_validation_ranked_report(
            duplicate,
            group_columns=("strategy",),
            candidate_columns=("model_name",),
        )


def test_non_finite_selection_metric_is_rejected() -> None:
    """NaN cannot drift to a winner through dataframe sort behavior."""
    metrics = _metrics()
    metrics.loc[
        metrics["split_name"].eq("valid")
        & metrics["model_name"].eq("model_a"),
        "MAE",
    ] = float("nan")
    with pytest.raises(model_selection.ModelSelectionError, match="non-finite MAE"):
        model_selection.build_validation_ranked_report(
            metrics,
            group_columns=("strategy",),
            candidate_columns=("model_name",),
        )
