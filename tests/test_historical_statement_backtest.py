"""Tests for the frozen historical statement-only backtest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from cf_diff import historical_statement_backtest as backtest
from cf_diff import statement_archive
from cf_diff.statement_features import STATEMENT_FEATURE_COLUMNS

CONFIG_PATH = Path("configs/historical_statement_backtest.json")


def _copy_config(tmp_path: Path) -> Path:
    path = tmp_path / "historical_statement_backtest.json"
    path.write_bytes(CONFIG_PATH.read_bytes())
    return path


def _model_frame(count: int = 10) -> pd.DataFrame:
    rows = []
    for offset in range(count):
        rows.append(
            {
                "contest_id": 1000 + offset,
                "index": "A",
                "rating": 800 + offset * 100,
                "start_time_seconds": 1_700_000_000 + offset * 100,
                "index_rank": 1,
                "index_number": 1,
                "tags": ["graphs"],
                "solved": 100_000 - offset,
                "submission_count": 200_000 - offset,
            }
        )
    return pd.DataFrame(rows)


def _valid_html(marker: str = "array") -> bytes:
    return (
        '<html><div class="problem-statement">'
        '<div class="time-limit">time limit per test 1 second</div>'
        '<div class="memory-limit">memory limit per test 256 megabytes</div>'
        f"<p>Given an {marker} of 10 integers, answer 2 queries.</p>"
        '<div class="input-specification"><p>Input contains n.</p></div>'
        '<div class="output-specification"><p>Print the answer.</p></div>'
        '<div class="sample-test"><div class="input">1</div>'
        '<div class="output">1</div></div></div></html>'
    ).encode("utf-8")


def _write_inputs(
    tmp_path: Path, count: int = 10
) -> tuple[Path, Path, Path, pd.DataFrame]:
    model = _model_frame(count)
    model_path = tmp_path / "model_table.parquet"
    model.to_parquet(model_path, engine="pyarrow", index=False)
    cache_dir = tmp_path / "pages"
    cache_dir.mkdir()
    for row in model.itertuples(index=False):
        cache_path = backtest.cache_path_for_problem(
            cache_dir, row.contest_id, row.index
        )
        cache_path.write_bytes(_valid_html(str(row.contest_id)))
    manifest_path = tmp_path / "cache_manifest.csv"
    statement_archive.create_archive(
        cache_dir,
        manifest_path,
        tmp_path / "cache_manifest_summary.json",
    )
    return model_path, cache_dir, manifest_path, model


def _prepared_frame(count: int = 10) -> pd.DataFrame:
    model = _model_frame(count).loc[:, list(backtest.MODEL_TABLE_COLUMNS)].copy()
    for column in STATEMENT_FEATURE_COLUMNS:
        model[column] = 0.0
    return model


def test_config_enforces_exact_feature_allowlists(tmp_path: Path) -> None:
    config = backtest.load_backtest_config(CONFIG_PATH)
    assert config.comparator_features == ("index_rank", "index_number")
    assert len(config.comparator_features) == 2
    assert config.primary_features == (
        "index_rank",
        "index_number",
        *STATEMENT_FEATURE_COLUMNS,
    )
    assert len(config.primary_features) == 43

    changed = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    changed["feature_sets"]["primary"].append("tags")
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(backtest.HistoricalBacktestError, match="exact 41"):
        backtest.load_backtest_config(changed_path)


def test_preparation_never_admits_tags_solved_or_submission_columns(
    tmp_path: Path,
) -> None:
    model_path, cache_dir, _cache_manifest, model = _write_inputs(tmp_path, count=2)
    prepared, manifest = backtest.build_prepared_dataset(model_path, cache_dir)

    assert tuple(prepared.columns) == (
        *backtest.MODEL_TABLE_COLUMNS,
        *STATEMENT_FEATURE_COLUMNS,
    )
    assert not {"tags", "solved", "submission_count"} & set(prepared.columns)
    config = backtest.load_backtest_config(CONFIG_PATH)
    for setting in backtest.SETTING_NAMES:
        assert not {"tags", "solved", "submission_count"} & set(
            backtest.feature_columns(config, setting)
        )
    first_page = backtest.cache_path_for_problem(
        cache_dir, model.iloc[0]["contest_id"], model.iloc[0]["index"]
    )
    assert manifest.loc[0, "stored_byte_size"] == first_page.stat().st_size
    assert manifest.loc[0, "stored_byte_sha256"] == hashlib.sha256(
        first_page.read_bytes()
    ).hexdigest()
    assert manifest.loc[0, "cache_relpath"] == first_page.name
    assert str(tmp_path) not in manifest.loc[0, "cache_relpath"]


def test_pdf_empty_and_parser_failures_are_retained_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model_frame(4)
    model_path = tmp_path / "model.parquet"
    model.to_parquet(model_path, engine="pyarrow", index=False)
    cache_dir = tmp_path / "pages"
    cache_dir.mkdir()
    payloads = (_valid_html(), b"%PDF-1.7 fake", b"", b"explode parser")
    for row, payload in zip(model.itertuples(index=False), payloads, strict=True):
        backtest.cache_path_for_problem(cache_dir, row.contest_id, row.index).write_bytes(
            payload
        )

    real_parser = backtest.parse_problem_statement

    def raising_parser(text: str):  # type: ignore[no-untyped-def]
        if "explode parser" in text:
            raise ValueError("synthetic parse failure")
        return real_parser(text)

    monkeypatch.setattr(backtest, "parse_problem_statement", raising_parser)
    prepared, manifest = backtest.build_prepared_dataset(model_path, cache_dir)

    assert len(prepared) == 4
    assert manifest["parse_status"].tolist() == [
        "parsed",
        "pdf_not_html",
        "empty_file",
        "parse_failed",
    ]
    assert prepared["statement_available"].tolist() == [1, 0, 0, 0]
    assert manifest.loc[2, "stored_byte_size"] == 0
    assert manifest.loc[2, "stored_byte_sha256"] == hashlib.sha256(b"").hexdigest()


def test_complete_equal_timestamp_buckets_never_cross_splits() -> None:
    config = backtest.load_backtest_config(CONFIG_PATH)
    prepared = _prepared_frame(10)
    duplicate = prepared.iloc[[6]].copy()
    duplicate["contest_id"] = 9999
    duplicate["index"] = "B"
    prepared = pd.concat([prepared, duplicate], ignore_index=True)

    split = backtest.build_frozen_split(prepared, config)
    at_boundary = split.loc[
        split["start_time_seconds"].eq(prepared.iloc[6]["start_time_seconds"])
    ]
    assert len(at_boundary) == 2
    assert at_boundary["split_name"].nunique() == 1
    maxima = split.groupby("split_name")["start_time_seconds"].agg(["min", "max"])
    assert maxima.loc["train", "max"] < maxima.loc["valid", "min"]
    assert maxima.loc["valid", "max"] < maxima.loc["test", "min"]


def test_validation_winner_is_locked_even_when_another_alpha_would_win_test() -> None:
    config = backtest.load_backtest_config(CONFIG_PATH)
    prepared = _prepared_frame(10)
    split = backtest.build_frozen_split(prepared, config)
    validation_calls: list[tuple[int, float]] = []

    def validation_predictor(
        train: pd.DataFrame,
        evaluate: pd.DataFrame,
        features: Sequence[str],
        alpha: float,
    ) -> np.ndarray:
        del train
        validation_calls.append((len(features), alpha))
        valid_times = set(
            split.loc[split["split_name"].eq("valid"), "start_time_seconds"]
        )
        assert set(evaluate["start_time_seconds"]) == valid_times
        penalty = abs(np.log10(alpha)) * 10.0
        return evaluate["rating"].to_numpy(dtype=float) + penalty

    _metrics, selected = backtest.select_validation_alphas(
        prepared, split, config, predictor=validation_predictor
    )
    assert selected == {"comparator": 1.0, "primary": 1.0}
    assert len(validation_calls) == 2 * len(config.alpha_candidates)

    test_calls: list[tuple[int, float]] = []

    def test_predictor(
        train: pd.DataFrame,
        evaluate: pd.DataFrame,
        features: Sequence[str],
        alpha: float,
    ) -> np.ndarray:
        del train
        test_calls.append((len(features), alpha))
        # alpha=10 would have zero test error, but it must never be tried.
        penalty = 0.0 if alpha == 10.0 else 100.0
        return evaluate["rating"].to_numpy(dtype=float) + penalty

    test_metrics, _predictions = backtest.evaluate_locked_test(
        prepared, split, config, selected, predictor=test_predictor
    )
    assert test_calls == [(2, 1.0), (43, 1.0)]
    assert test_metrics["settings"]["comparator"]["mae"] == 100.0
    assert test_metrics["settings"]["primary"]["mae"] == 100.0


def test_selection_artifact_hash_tampering_is_rejected(tmp_path: Path) -> None:
    config_path = _copy_config(tmp_path)
    model_path, cache_dir, cache_manifest, _model = _write_inputs(tmp_path)
    selection_dir = tmp_path / "selection"
    backtest.run_selection(
        config_path=config_path,
        model_table_path=model_path,
        cache_dir=cache_dir,
        cache_manifest_path=cache_manifest,
        output_dir=selection_dir,
    )
    metrics_path = selection_dir / backtest.VALIDATION_METRICS_FILENAME
    with metrics_path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(backtest.HistoricalBacktestError, match="hash mismatch"):
        backtest.run_test(
            config_path=config_path,
            selection_dir=selection_dir,
            output_dir=tmp_path / "test-output",
        )


def test_selection_rejects_cache_that_changed_after_manifest(tmp_path: Path) -> None:
    config_path = _copy_config(tmp_path)
    model_path, cache_dir, cache_manifest, model = _write_inputs(tmp_path)
    changed_page = backtest.cache_path_for_problem(
        cache_dir, model.iloc[0]["contest_id"], model.iloc[0]["index"]
    )
    changed_page.write_bytes(changed_page.read_bytes() + b"changed")

    with pytest.raises(backtest.HistoricalBacktestError, match="frozen manifest"):
        backtest.run_selection(
            config_path=config_path,
            model_table_path=model_path,
            cache_dir=cache_dir,
            cache_manifest_path=cache_manifest,
            output_dir=tmp_path / "selection",
        )


def test_selection_lock_hash_tampering_is_rejected(tmp_path: Path) -> None:
    config_path = _copy_config(tmp_path)
    model_path, cache_dir, cache_manifest, _model = _write_inputs(tmp_path)
    selection_dir = tmp_path / "selection"
    backtest.run_selection(
        config_path=config_path,
        model_table_path=model_path,
        cache_dir=cache_dir,
        cache_manifest_path=cache_manifest,
        output_dir=selection_dir,
    )
    lock_path = selection_dir / backtest.SELECTION_LOCK_FILENAME
    lock_path.write_bytes(lock_path.read_bytes() + b" ")

    with pytest.raises(backtest.HistoricalBacktestError, match="lock hash mismatch"):
        backtest.run_test(
            config_path=config_path,
            selection_dir=selection_dir,
            output_dir=tmp_path / "test-output",
        )


def test_locked_alpha_must_still_be_the_validation_winner(tmp_path: Path) -> None:
    config_path = _copy_config(tmp_path)
    model_path, cache_dir, cache_manifest, _model = _write_inputs(tmp_path)
    selection_dir = tmp_path / "selection"
    backtest.run_selection(
        config_path=config_path,
        model_table_path=model_path,
        cache_dir=cache_dir,
        cache_manifest_path=cache_manifest,
        output_dir=selection_dir,
    )
    lock_path = selection_dir / backtest.SELECTION_LOCK_FILENAME
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    current = lock["feature_sets"]["comparator"]["selected_alpha"]
    replacement = next(
        alpha
        for alpha in backtest.load_backtest_config(config_path).alpha_candidates
        if alpha != current
    )
    lock["feature_sets"]["comparator"]["selected_alpha"] = replacement
    lock_path.write_bytes(
        (json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
    )
    lock_hash = backtest.sha256_file(lock_path)
    (selection_dir / backtest.SELECTION_HASH_FILENAME).write_text(
        f"{lock_hash}  {backtest.SELECTION_LOCK_FILENAME}\n",
        encoding="ascii",
        newline="\n",
    )

    with pytest.raises(backtest.HistoricalBacktestError, match="validation winner"):
        backtest.run_test(
            config_path=config_path,
            selection_dir=selection_dir,
            output_dir=tmp_path / "test-output",
        )


def test_primary_top10_is_automatic_and_stably_tie_broken() -> None:
    rows = []
    for contest_id, index, timestamp, error in (
        (5, "B", 30, 10.0),
        (4, "C", 20, 10.0),
        (4, "A", 20, 10.0),
        (9, "A", 10, 9.0),
        (8, "A", 40, 8.0),
    ):
        rows.append(
            {
                "setting": "primary",
                "contest_id": contest_id,
                "index": index,
                "start_time_seconds": timestamp,
                "rating": 1000.0,
                "prediction": 1000.0 + error,
                "absolute_error": error,
            }
        )
    predictions = pd.DataFrame(rows)
    top = backtest.primary_top_errors(predictions, 3)
    assert list(top[["contest_id", "index"]].itertuples(index=False, name=None)) == [
        (4, "A"),
        (4, "C"),
        (5, "B"),
    ]
    assert top["error_rank"].tolist() == [1, 2, 3]


def test_paired_cluster_bootstrap_is_deterministic() -> None:
    rows = []
    for setting, errors in (
        ("comparator", (10.0, 20.0, 30.0)),
        ("primary", (5.0, 25.0, 15.0)),
    ):
        for (contest_id, index), error in zip(
            ((1, "A"), (1, "B"), (2, "A")), errors, strict=True
        ):
            rows.append(
                {
                    "setting": setting,
                    "contest_id": contest_id,
                    "index": index,
                    "absolute_error": error,
                }
            )
    predictions = pd.DataFrame(rows)
    first = backtest.paired_contest_cluster_bootstrap(
        predictions, resamples=1000, seed=42, confidence_level=0.95
    )
    second = backtest.paired_contest_cluster_bootstrap(
        predictions, resamples=1000, seed=42, confidence_level=0.95
    )
    assert first == second
    assert first["point_estimate"] == pytest.approx(-5.0)
    assert first["rng"] == "numpy_pcg64"
    assert first["quantile_method"] == "linear"


def test_cli_select_and_test_smoke_with_exclusive_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _copy_config(tmp_path)
    model_path, cache_dir, cache_manifest, _model = _write_inputs(tmp_path)
    selection_dir = tmp_path / "selection"
    test_dir = tmp_path / "test"
    select_args = [
        "select",
        "--config",
        str(config_path),
        "--model-table",
        str(model_path),
        "--cache-dir",
        str(cache_dir),
        "--cache-manifest",
        str(cache_manifest),
        "--output-dir",
        str(selection_dir),
    ]
    assert backtest.main(select_args) == 0
    assert backtest.main(select_args) == 1
    assert "Refusing to overwrite" in capsys.readouterr().err

    test_args = [
        "test",
        "--config",
        str(config_path),
        "--selection-dir",
        str(selection_dir),
        "--output-dir",
        str(test_dir),
    ]
    assert backtest.main(test_args) == 0
    assert backtest.main(test_args) == 1
    captured = capsys.readouterr()
    assert "Refusing to overwrite" in captured.err
    metrics = json.loads(
        (test_dir / backtest.TEST_METRICS_FILENAME).read_text(encoding="utf-8")
    )
    assert set(metrics["settings"]) == {"comparator", "primary"}
    assert len(pd.read_csv(test_dir / backtest.TEST_PREDICTIONS_FILENAME)) == 4
    assert len(pd.read_csv(test_dir / backtest.TOP10_FILENAME)) == 2
