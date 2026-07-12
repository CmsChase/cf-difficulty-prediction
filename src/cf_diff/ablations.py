"""Run feature-group ablations for Codeforces rating prediction."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cf_diff.baselines import (
    compute_regression_metrics,
    load_baseline_config,
    make_preprocessed_estimator,
)
from cf_diff.features import write_json
from cf_diff.model_selection import (
    DEFAULT_METRIC_COLUMNS,
    build_validation_ranked_report,
    select_rank_one,
)

DEFAULT_CONFIG_PATH: Final[Path] = Path("configs/experiment.yaml")
DEFAULT_FEATURE_PATH: Final[Path] = Path(
    "data/processed/features/model_table.parquet"
)
DEFAULT_FEATURE_COLUMNS_PATH: Final[Path] = Path(
    "data/processed/features/feature_columns.json"
)
DEFAULT_CONTEST_SPLIT_PATH: Final[Path] = Path(
    "data/processed/splits/contest_grouped_split.parquet"
)
DEFAULT_TIME_SPLIT_PATH: Final[Path] = Path(
    "data/processed/splits/forward_time_split.parquet"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path("outputs/ablations")
DEFAULT_LOG_PATH: Final[Path] = Path("outputs/logs/ablations.log")
TARGET_COLUMN: Final[str] = "rating"
STRATEGIES: Final[tuple[str, str]] = ("contest_grouped", "forward_time")
SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "valid", "test")
DROP_FEATURE_SETS: Final[tuple[tuple[str, str], ...]] = (
    ("index", "all_without_index"),
    ("solved", "all_without_solved"),
    ("tags", "all_without_tags"),
    ("points", "all_without_points"),
)
METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "MAE",
    "RMSE",
    "R2",
    "within_100",
    "within_200",
)


class AblationError(RuntimeError):
    """Raised when ablation training cannot proceed safely."""


class JsonLogFormatter(logging.Formatter):
    """Format ablation logs as JSON Lines."""

    def format(self, record: logging.LogRecord) -> str:
        """Return one machine-readable log record."""
        payload: dict[str, object] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
        details = getattr(record, "details", None)
        if isinstance(details, dict):
            payload.update(details)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class AblationModelSpec:
    """Define one deterministic ablation model."""

    model_name: str
    estimator_factory: Callable[[int], object]


def configure_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """Create the dedicated structured ablation logger."""
    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.ablations")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(resolved_path, encoding="utf-8")
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    """Flush and close all handlers attached to a dedicated logger."""
    for handler in tuple(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    table_name: str,
) -> None:
    """Raise a clear error when required columns are absent."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise AblationError(f"{table_name} lacks required columns: {missing}")


def _finite_float(value: object, digits: int = 6) -> float | None:
    """Convert numeric output to a rounded JSON-safe float."""
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _load_feature_metadata(path: Path) -> Mapping[str, object]:
    """Load feature metadata if present."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_feature_groups(
    model_table: pd.DataFrame,
    feature_metadata: Mapping[str, object] | None = None,
) -> dict[str, list[str]]:
    """Select available columns for each predefined feature group."""
    del feature_metadata
    groups = {
        "index": [
            column
            for column in ("index_letter", "index_number", "index_rank")
            if column in model_table.columns
        ],
        "solved": [
            column
            for column in (
                "solved_count",
                "solved_count_missing",
                "log_solved_count",
            )
            if column in model_table.columns
        ],
        "tags": [
            column
            for column in model_table.columns
            if column == "tag_count" or column.startswith("tag__")
        ],
        "points": [
            column
            for column in ("has_points", "points")
            if column in model_table.columns
        ],
    }
    if not groups["index"]:
        raise AblationError("No index feature columns are available.")
    if not groups["solved"]:
        raise AblationError("No solved-count feature columns are available.")
    if not groups["tags"]:
        raise AblationError("No tag feature columns are available.")
    return {name: sorted(columns) for name, columns in groups.items()}


def _dedupe_preserve_order(columns: Sequence[str]) -> list[str]:
    """Return unique columns while preserving first appearance."""
    seen: set[str] = set()
    result: list[str] = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            result.append(column)
    return result


def _combine_groups(
    feature_groups: Mapping[str, Sequence[str]],
    group_names: Sequence[str],
) -> list[str]:
    """Return all columns from a list of feature groups."""
    columns: list[str] = []
    for group_name in group_names:
        columns.extend(feature_groups.get(group_name, []))
    return _dedupe_preserve_order(columns)


def build_feature_sets(
    feature_groups: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, object]]:
    """Build named feature-set definitions for ablation experiments."""
    all_groups = ("index", "solved", "tags", "points")
    definitions = {
        "index_only": ("index",),
        "solved_only": ("solved",),
        "tags_only": ("tags",),
        "index_plus_tags": ("index", "tags"),
        "index_plus_solved": ("index", "solved"),
        "tags_plus_solved": ("tags", "solved"),
        "index_tags_solved": ("index", "tags", "solved"),
        "all_api_features": all_groups,
        "all_without_index": ("solved", "tags", "points"),
        "all_without_solved": ("index", "tags", "points"),
        "all_without_tags": ("index", "solved", "points"),
        "all_without_points": ("index", "solved", "tags"),
    }
    feature_sets: dict[str, dict[str, object]] = {}
    for name, groups in definitions.items():
        columns = _combine_groups(feature_groups, groups)
        if not columns:
            raise AblationError(f"Feature set {name!r} has no columns.")
        feature_sets[name] = {
            "included_groups": list(groups),
            "feature_columns": columns,
            "feature_count": len(columns),
        }
    return feature_sets


def feature_group_definitions_table(
    feature_sets: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """Create a serializable feature-set definition table."""
    rows = []
    for name, definition in feature_sets.items():
        rows.append(
            {
                "feature_set_name": name,
                "included_groups": json.dumps(
                    definition["included_groups"],
                    ensure_ascii=False,
                ),
                "feature_count": int(definition["feature_count"]),
                "feature_columns": json.dumps(
                    definition["feature_columns"],
                    ensure_ascii=False,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["feature_set_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def _ridge_estimator(seed: int) -> Ridge:
    """Build a deterministic ridge regressor."""
    del seed
    return Ridge(alpha=1.0)


def _hist_gradient_boosting_estimator(
    seed: int,
) -> HistGradientBoostingRegressor:
    """Build a deterministic histogram gradient boosting regressor."""
    return HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=160,
        l2_regularization=0.01,
        random_state=seed,
    )


def build_model_specs() -> list[AblationModelSpec]:
    """Return deterministic ablation models."""
    return [
        AblationModelSpec("ridge_regression", _ridge_estimator),
        AblationModelSpec(
            "hist_gradient_boosting_regressor",
            _hist_gradient_boosting_estimator,
        ),
    ]


def join_split_assignments(
    model_table: pd.DataFrame,
    split_assignment: pd.DataFrame,
) -> pd.DataFrame:
    """Join model rows with row-level split assignments."""
    _require_columns(
        model_table,
        ("contest_id", "index", TARGET_COLUMN),
        "model table",
    )
    _require_columns(
        split_assignment,
        ("contest_id", "index", "split_name"),
        "split assignment",
    )
    joined = model_table.merge(
        split_assignment.loc[:, ["contest_id", "index", "split_name"]],
        on=["contest_id", "index"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(model_table):
        raise AblationError(
            "Split assignment did not match every model-table row: "
            f"{len(joined)} of {len(model_table)} rows matched."
        )
    missing_splits = [
        split_name
        for split_name in SPLIT_NAMES
        if not joined["split_name"].eq(split_name).any()
    ]
    if missing_splits:
        raise AblationError(f"Split assignment has empty splits: {missing_splits}")
    return joined


def _fit_predict(
    joined: pd.DataFrame,
    feature_columns: Sequence[str],
    model_spec: AblationModelSpec,
    seed: int,
) -> pd.Series:
    """Fit one ablation model on train rows and predict all rows."""
    train = joined.loc[joined["split_name"].eq("train")].copy()
    train_x = train.loc[:, list(feature_columns)]
    train_y = pd.to_numeric(train[TARGET_COLUMN], errors="raise").to_numpy(
        dtype=float
    )
    model = make_preprocessed_estimator(
        model_spec.estimator_factory(seed),
        train_x,
    )
    model.fit(train_x, train_y)
    predictions = model.predict(joined.loc[:, list(feature_columns)])
    return pd.Series(predictions, index=joined.index, dtype=float)


def evaluate_ablation_strategy(
    model_table: pd.DataFrame,
    split_assignment: pd.DataFrame,
    *,
    strategy: str,
    feature_sets: Mapping[str, Mapping[str, object]],
    model_specs: Sequence[AblationModelSpec],
    seed: int,
) -> pd.DataFrame:
    """Evaluate every model and feature set for one split strategy."""
    joined = join_split_assignments(model_table, split_assignment)
    rows: list[dict[str, object]] = []
    for model_spec in model_specs:
        for feature_set_name, definition in feature_sets.items():
            feature_columns = list(definition["feature_columns"])
            predictions = _fit_predict(
                joined,
                feature_columns,
                model_spec,
                seed,
            )
            for split_name in SPLIT_NAMES:
                mask = joined["split_name"].eq(split_name)
                metrics = compute_regression_metrics(
                    joined.loc[mask, TARGET_COLUMN],
                    predictions.loc[mask],
                )
                rows.append(
                    {
                        "strategy": strategy,
                        "model_name": model_spec.model_name,
                        "feature_set_name": feature_set_name,
                        "split_name": split_name,
                        **metrics,
                        "feature_count": len(feature_columns),
                        "row_count": int(mask.sum()),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["strategy", "model_name", "feature_set_name", "split_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_drop_comparison(test_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare all features against removing each feature group."""
    rows: list[dict[str, object]] = []
    for (strategy, model_name), group in test_metrics.groupby(
        ["strategy", "model_name"],
        sort=True,
    ):
        full = group.loc[group["feature_set_name"].eq("all_api_features")]
        if full.empty:
            continue
        full_mae = float(full.iloc[0]["MAE"])
        for removed_group, feature_set_name in DROP_FEATURE_SETS:
            ablated = group.loc[group["feature_set_name"].eq(feature_set_name)]
            if ablated.empty:
                continue
            ablated_mae = float(ablated.iloc[0]["MAE"])
            difference = ablated_mae - full_mae
            percent = (difference / full_mae * 100.0) if full_mae else np.nan
            rows.append(
                {
                    "strategy": strategy,
                    "model_name": model_name,
                    "removed_group": removed_group,
                    "full_feature_set": "all_api_features",
                    "ablated_feature_set": feature_set_name,
                    "all_api_features_MAE": round(full_mae, 6),
                    "ablated_MAE": round(ablated_mae, 6),
                    "MAE_difference": round(float(difference), 6),
                    "percent_MAE_change": round(float(percent), 6),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["strategy", "model_name", "removed_group"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_locked_ablation_report(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select one algorithm on full-feature validation MAE and lock it."""
    reference = metrics.loc[
        metrics["feature_set_name"].eq("all_api_features")
    ]
    reference_ranking = build_validation_ranked_report(
        reference,
        group_columns=("strategy",),
        candidate_columns=("model_name",),
        metric_columns=DEFAULT_METRIC_COLUMNS,
    )
    selected_models = select_rank_one(reference_ranking).loc[
        :, ["strategy", "model_name"]
    ]
    locked = metrics.merge(
        selected_models,
        on=["strategy", "model_name"],
        how="inner",
        validate="many_to_one",
    )
    report = build_validation_ranked_report(
        locked,
        group_columns=("strategy", "feature_set_name"),
        candidate_columns=("model_name",),
        metric_columns=DEFAULT_METRIC_COLUMNS,
    )
    return select_rank_one(report).sort_values(
        ["strategy", "feature_set_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_ablation_summary(
    test_metrics: pd.DataFrame,
    drop_comparison: pd.DataFrame,
) -> dict[str, object]:
    """Build machine-readable ablation summary."""
    validation_selected_by_strategy: dict[str, dict[str, object]] = {}
    for strategy, group in test_metrics.groupby("strategy", sort=True):
        selection_column = (
            "validation_MAE" if "validation_MAE" in group.columns else "MAE"
        )
        row = group.sort_values(
            [selection_column, "feature_set_name"],
            kind="mergesort",
        ).iloc[0]
        validation_selected_by_strategy[str(strategy)] = {
            "strategy": strategy,
            "model_name": row["model_name"],
            "feature_set_name": row["feature_set_name"],
            "validation_MAE": _finite_float(row.get("validation_MAE")),
            "test_MAE": _finite_float(row["MAE"]),
            "within_200": _finite_float(row["within_200"]),
            "feature_count": int(row["feature_count"]),
        }
    importance_notes = []
    for (strategy, model_name), group in drop_comparison.groupby(
        ["strategy", "model_name"],
        sort=True,
    ):
        row = group.sort_values(
            ["MAE_difference", "removed_group"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        if float(row["MAE_difference"]) > 0:
            note = (
                f"Removing {row['removed_group']} has the largest descriptively "
                "observed test-MAE increase among the pre-specified one-group "
                "drops for this strategy/model; this comparison is exploratory."
            )
        else:
            note = (
                "No pre-specified one-group drop descriptively increased test "
                "MAE relative to all API features for this strategy/model; "
                "this comparison is exploratory."
            )
        importance_notes.append(
            {
                "strategy": strategy,
                "model_name": model_name,
                "most_important_removed_group": row["removed_group"],
                "MAE_difference": _finite_float(row["MAE_difference"]),
                "note": note,
            }
        )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validation_selected_feature_set_test_report": (
            validation_selected_by_strategy
        ),
        "drop_comparison": drop_comparison.to_dict(orient="records"),
        "feature_group_importance_notes": importance_notes,
    }


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Persist a matplotlib figure with stable output settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "cf-diff-ablations"},
    )
    plt.close(fig)


def _empty_figure(title: str) -> plt.Figure:
    """Create a readable placeholder figure for empty inputs."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title(title)
    ax.set_axis_off()
    return fig


def plot_ablation_metric(
    test_metrics: pd.DataFrame,
    strategy: str,
    metric: str,
    path: Path,
) -> None:
    """Save grouped bar chart for one ablation metric and strategy."""
    subset = test_metrics.loc[test_metrics["strategy"].eq(strategy)].copy()
    title = f"{metric} by ablation feature set: {strategy}"
    if subset.empty:
        _save_figure(_empty_figure(title), path)
        return
    order_metric = "validation_MAE" if "validation_MAE" in subset else "MAE"
    feature_sets = (
        subset.sort_values([order_metric, "feature_set_name"])["feature_set_name"]
        .drop_duplicates()
        .tolist()
    )
    models = sorted(subset["model_name"].unique())
    x = np.arange(len(feature_sets))
    width = min(0.8 / max(len(models), 1), 0.38)
    fig, ax = plt.subplots(figsize=(13, 6))
    for model_index, model_name in enumerate(models):
        values = (
            subset.loc[subset["model_name"].eq(model_name)]
            .set_index("feature_set_name")
            .reindex(feature_sets)[metric]
        )
        offset = (model_index - (len(models) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=model_name)
    ax.set_title(title)
    ax.set_xlabel("Feature set")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_sets, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save_figure(fig, path)


def plot_drop_comparison(drop_comparison: pd.DataFrame, path: Path) -> None:
    """Save MAE change from dropping each feature group."""
    title = "Test MAE change when removing feature groups"
    if drop_comparison.empty:
        _save_figure(_empty_figure(title), path)
        return
    labels = (
        drop_comparison["strategy"]
        + "\n"
        + drop_comparison["model_name"]
        + "\n- "
        + drop_comparison["removed_group"]
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(np.arange(len(drop_comparison)), drop_comparison["MAE_difference"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Strategy / model / removed group")
    ax.set_ylabel("Ablated MAE - all_api_features MAE")
    ax.set_xticks(np.arange(len(drop_comparison)))
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, path)


def run_ablations(
    *,
    config_path: Path,
    feature_path: Path,
    feature_columns_path: Path,
    contest_split_path: Path,
    time_split_path: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Path]:
    """Run all ablation experiments and write tables, figures, and summary."""
    logger = configure_logger(log_path)
    try:
        config = load_baseline_config(config_path)
        model_table = pd.read_parquet(feature_path, engine="pyarrow")
        feature_metadata = _load_feature_metadata(feature_columns_path)
        feature_groups = build_feature_groups(model_table, feature_metadata)
        feature_sets = build_feature_sets(feature_groups)
        model_specs = build_model_specs()
        contest_split = pd.read_parquet(contest_split_path, engine="pyarrow")
        time_split = pd.read_parquet(time_split_path, engine="pyarrow")

        metrics = pd.concat(
            [
                evaluate_ablation_strategy(
                    model_table,
                    contest_split,
                    strategy="contest_grouped",
                    feature_sets=feature_sets,
                    model_specs=model_specs,
                    seed=config.random_seed,
                ),
                evaluate_ablation_strategy(
                    model_table,
                    time_split,
                    strategy="forward_time",
                    feature_sets=feature_sets,
                    model_specs=model_specs,
                    seed=config.random_seed,
                ),
            ],
            ignore_index=True,
        ).sort_values(
            ["strategy", "model_name", "feature_set_name", "split_name"],
            kind="mergesort",
        )
        test_metrics = metrics.loc[metrics["split_name"].eq("test")].copy()
        test_metrics = test_metrics.sort_values(
            ["strategy", "model_name", "MAE", "feature_set_name"],
            kind="mergesort",
        ).reset_index(drop=True)
        locked_test_report = build_locked_ablation_report(metrics)
        drop_comparison = build_drop_comparison(locked_test_report)
        feature_definitions = feature_group_definitions_table(feature_sets)
        summary = build_ablation_summary(locked_test_report, drop_comparison)

        output_dir = output_dir.resolve()
        summary_dir = output_dir / "summary"
        tables_dir = output_dir / "tables"
        figures_dir = output_dir / "figures"
        for directory in (summary_dir, tables_dir, figures_dir):
            directory.mkdir(parents=True, exist_ok=True)

        paths = {
            "ablation_summary": summary_dir / "ablation_summary.json",
            "ablation_metrics_all": tables_dir / "ablation_metrics_all.csv",
            "ablation_metrics_test": tables_dir / "ablation_metrics_test.csv",
            "locked_test_results": (
                tables_dir / "ablation_validation_locked_test.csv"
            ),
            "feature_group_definitions": (
                tables_dir / "feature_group_definitions.csv"
            ),
            "ablation_drop_comparison": (
                tables_dir / "ablation_drop_comparison.csv"
            ),
            "ablation_mae_by_feature_set_contest_grouped": (
                figures_dir / "ablation_mae_by_feature_set_contest_grouped.png"
            ),
            "ablation_mae_by_feature_set_forward_time": (
                figures_dir / "ablation_mae_by_feature_set_forward_time.png"
            ),
            "ablation_within_200_by_feature_set_contest_grouped": (
                figures_dir
                / "ablation_within_200_by_feature_set_contest_grouped.png"
            ),
            "ablation_within_200_by_feature_set_forward_time": (
                figures_dir / "ablation_within_200_by_feature_set_forward_time.png"
            ),
            "feature_drop_mae_change": figures_dir / "feature_drop_mae_change.png",
        }

        write_json(paths["ablation_summary"], summary)
        metrics.to_csv(paths["ablation_metrics_all"], index=False)
        test_metrics.to_csv(paths["ablation_metrics_test"], index=False)
        locked_test_report.to_csv(paths["locked_test_results"], index=False)
        feature_definitions.to_csv(paths["feature_group_definitions"], index=False)
        drop_comparison.to_csv(paths["ablation_drop_comparison"], index=False)
        for strategy in STRATEGIES:
            plot_ablation_metric(
                locked_test_report,
                strategy,
                "MAE",
                paths[f"ablation_mae_by_feature_set_{strategy}"],
            )
            plot_ablation_metric(
                locked_test_report,
                strategy,
                "within_200",
                paths[f"ablation_within_200_by_feature_set_{strategy}"],
            )
        plot_drop_comparison(drop_comparison, paths["feature_drop_mae_change"])

        logger.info(
            "Completed Codeforces ablation study",
            extra={
                "event": "ablations_completed",
                "details": {
                    "output_dir": output_dir.as_posix(),
                    "random_seed": config.random_seed,
                    "experiment_count": len(test_metrics),
                },
            },
        )
        return paths
    except Exception:
        logger.exception(
            "Codeforces ablation study failed",
            extra={"event": "ablations_failed"},
        )
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the ablation command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run Codeforces feature-group ablation experiments."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument(
        "--feature-columns-path",
        type=Path,
        default=DEFAULT_FEATURE_COLUMNS_PATH,
    )
    parser.add_argument(
        "--contest-split-path",
        type=Path,
        default=DEFAULT_CONTEST_SPLIT_PATH,
    )
    parser.add_argument(
        "--time-split-path",
        type=Path,
        default=DEFAULT_TIME_SPLIT_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ablation CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        paths = run_ablations(
            config_path=args.config,
            feature_path=args.feature_path,
            feature_columns_path=args.feature_columns_path,
            contest_split_path=args.contest_split_path,
            time_split_path=args.time_split_path,
            output_dir=args.output_dir,
            log_path=args.log_path,
        )
    except (AblationError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote ablation summary: {paths['ablation_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
