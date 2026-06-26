"""Exploratory data analysis outputs for Codeforces difficulty prediction."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_CONFIG_PATH: Final[Path] = Path("configs/experiment.yaml")
DEFAULT_PROCESSED_PATH: Final[Path] = Path(
    "data/processed/rated_programming_problems.parquet"
)
DEFAULT_FEATURE_PATH: Final[Path] = Path(
    "data/processed/features/model_table.parquet"
)
DEFAULT_CONTEST_SPLIT_PATH: Final[Path] = Path(
    "data/processed/splits/contest_grouped_split.parquet"
)
DEFAULT_TIME_SPLIT_PATH: Final[Path] = Path(
    "data/processed/splits/forward_time_split.parquet"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path("outputs/eda")
DEFAULT_LOG_PATH: Final[Path] = Path("outputs/logs/eda.log")
SUMMARY_DIR_NAME: Final[str] = "summary"
FIGURE_DIR_NAME: Final[str] = "figures"
TOP_N_TAGS: Final[int] = 20


class EDAError(RuntimeError):
    """Raised when the EDA pipeline cannot safely produce outputs."""


class JsonLogFormatter(logging.Formatter):
    """Format EDA audit logs as JSON Lines."""

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


def configure_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """Create the dedicated structured EDA logger."""
    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.eda")
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


def write_json(path: Path, payload: object) -> None:
    """Write deterministic pretty UTF-8 JSON."""
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    table_name: str,
) -> None:
    """Fail loudly when a required local input column is missing."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise EDAError(f"{table_name} lacks required columns: {missing}")


def _normalize_tags(value: object) -> list[str]:
    """Normalize a Parquet list-like tags value into unique strings."""
    if value is None or value is pd.NA:
        return []
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, (list, tuple, set)):
        return []
    return sorted({str(tag) for tag in value if str(tag)})


def _finite_int(value: object) -> int | None:
    """Convert numeric aggregate output to int, preserving empty results."""
    if value is None or pd.isna(value):
        return None
    return int(value)


def _finite_float(value: object, digits: int = 6) -> float | None:
    """Convert numeric aggregate output to rounded float."""
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def build_solved_count_quantiles(frame: pd.DataFrame) -> dict[str, float | None]:
    """Summarize important solved-count distribution quantiles."""
    _require_columns(frame, ("solved_count",), "processed problem table")
    solved_count = pd.to_numeric(frame["solved_count"], errors="coerce").dropna()
    if solved_count.empty:
        return {
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "p50": _finite_float(solved_count.quantile(0.50), digits=3),
        "p75": _finite_float(solved_count.quantile(0.75), digits=3),
        "p90": _finite_float(solved_count.quantile(0.90), digits=3),
        "p95": _finite_float(solved_count.quantile(0.95), digits=3),
        "p99": _finite_float(solved_count.quantile(0.99), digits=3),
        "max": _finite_float(solved_count.max(), digits=3),
    }


def build_tag_frequency(frame: pd.DataFrame) -> pd.DataFrame:
    """Count tag occurrences across rated programming problems."""
    _require_columns(frame, ("tags",), "processed problem table")
    normalized_tags = frame["tags"].map(_normalize_tags)
    counter = Counter(tag for tags in normalized_tags for tag in tags)
    denominator = max(len(frame), 1)
    rows = [
        {
            "tag": tag,
            "count": int(count),
            "problem_share": round(count / denominator, 6),
        }
        for tag, count in counter.items()
    ]
    result = pd.DataFrame(rows, columns=["tag", "count", "problem_share"])
    if result.empty:
        return result
    return result.sort_values(
        ["count", "tag"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _ensure_index_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return index_letter and index_rank, deriving them when absent."""
    result = frame.copy()
    if "index_letter" not in result.columns:
        _require_columns(result, ("index",), "problem table")
        result["index_letter"] = (
            result["index"]
            .astype("string")
            .str.extract(r"^([A-Za-z]+)", expand=False)
            .str.upper()
            .fillna("")
        )
    if "index_rank" not in result.columns:
        mapping = {
            letter: rank
            for rank, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ", start=1)
        }
        result["index_rank"] = (
            result["index_letter"]
            .astype("string")
            .str[0]
            .map(mapping)
            .fillna(0)
            .astype(int)
        )
    return result


def build_index_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize problem counts and ratings by leading index letter."""
    _require_columns(frame, ("rating",), "problem table")
    with_index = _ensure_index_features(frame)
    grouped = (
        with_index.assign(index_letter=with_index["index_letter"].replace("", ""))
        .groupby(["index_rank", "index_letter"], dropna=False)
        .agg(
            problem_count=("index_letter", "size"),
            rating_mean=("rating", "mean"),
            rating_median=("rating", "median"),
        )
        .reset_index()
    )
    if grouped.empty:
        return pd.DataFrame(
            columns=[
                "index_letter",
                "index_rank",
                "problem_count",
                "problem_share",
                "rating_mean",
                "rating_median",
            ]
        )
    grouped["problem_share"] = (grouped["problem_count"] / len(frame)).round(6)
    grouped["rating_mean"] = grouped["rating_mean"].round(3)
    grouped["rating_median"] = grouped["rating_median"].round(3)
    return grouped.loc[
        :,
        [
            "index_letter",
            "index_rank",
            "problem_count",
            "problem_share",
            "rating_mean",
            "rating_median",
        ],
    ].sort_values(["index_rank", "index_letter"], kind="mergesort")


def build_rating_band_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize row counts and solve behavior by 200-point rating bands."""
    _require_columns(frame, ("rating", "solved_count", "tags"), "processed table")
    rating = pd.to_numeric(frame["rating"], errors="coerce")
    valid = frame.loc[rating.notna()].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "rating_band",
                "rating_min",
                "rating_max",
                "problem_count",
                "problem_share",
                "solved_count_median",
                "tag_count_mean",
            ]
        )
    valid["rating"] = pd.to_numeric(valid["rating"], errors="coerce")
    valid["solved_count"] = pd.to_numeric(
        valid["solved_count"],
        errors="coerce",
    )
    valid["tag_count"] = valid["tags"].map(lambda value: len(_normalize_tags(value)))
    lower = int(math.floor(float(valid["rating"].min()) / 200.0) * 200)
    upper = int(math.ceil(float(valid["rating"].max()) / 200.0) * 200 + 200)
    bins = list(range(lower, upper + 1, 200))
    valid["rating_band"] = pd.cut(
        valid["rating"],
        bins=bins,
        right=False,
        include_lowest=True,
    )
    grouped = (
        valid.groupby("rating_band", observed=True)
        .agg(
            rating_min=("rating", "min"),
            rating_max=("rating", "max"),
            problem_count=("rating", "size"),
            solved_count_median=("solved_count", "median"),
            tag_count_mean=("tag_count", "mean"),
        )
        .reset_index()
    )
    grouped["rating_band"] = grouped["rating_band"].map(
        lambda interval: f"{int(interval.left)}-{int(interval.right - 1)}"
    )
    grouped["problem_share"] = (grouped["problem_count"] / len(valid)).round(6)
    grouped["solved_count_median"] = grouped["solved_count_median"].round(3)
    grouped["tag_count_mean"] = grouped["tag_count_mean"].round(3)
    return grouped.loc[
        :,
        [
            "rating_band",
            "rating_min",
            "rating_max",
            "problem_count",
            "problem_share",
            "solved_count_median",
            "tag_count_mean",
        ],
    ]


def _split_size_summary(split: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return row and contest counts by split name."""
    _require_columns(split, ("contest_id", "split_name"), "split assignment")
    split_names = ("train", "valid", "test")
    row_counts = split["split_name"].value_counts()
    contest_counts = split.groupby("split_name")["contest_id"].nunique()
    return {
        "rows": {
            name: int(row_counts.get(name, 0))
            for name in split_names
        },
        "contests": {
            name: int(contest_counts.get(name, 0))
            for name in split_names
        },
    }


def build_dataset_summary(
    processed: pd.DataFrame,
    features: pd.DataFrame,
    contest_split: pd.DataFrame,
    time_split: pd.DataFrame,
    tag_frequency: pd.DataFrame,
    *,
    config_path: Path,
    processed_path: Path,
    feature_path: Path,
    contest_split_path: Path,
    time_split_path: Path,
) -> dict[str, object]:
    """Build a machine-readable summary for the EDA report."""
    _require_columns(
        processed,
        ("contest_id", "rating", "points", "tags", "solved_count"),
        "processed problem table",
    )
    _require_columns(features, ("contest_id", "rating"), "feature table")
    rating = pd.to_numeric(processed["rating"], errors="coerce")
    solved_count = pd.to_numeric(processed["solved_count"], errors="coerce")
    top_tags = tag_frequency.head(10).loc[:, ["tag", "count"]].to_dict(
        orient="records"
    )
    return {
        "inputs": {
            "config_path": config_path.as_posix(),
            "config_exists": config_path.exists(),
            "processed_path": processed_path.as_posix(),
            "feature_path": feature_path.as_posix(),
            "contest_split_path": contest_split_path.as_posix(),
            "time_split_path": time_split_path.as_posix(),
        },
        "row_counts": {
            "processed": int(len(processed)),
            "feature_table": int(len(features)),
            "contest_grouped_split": int(len(contest_split)),
            "forward_time_split": int(len(time_split)),
        },
        "column_counts": {
            "processed": int(len(processed.columns)),
            "feature_table": int(len(features.columns)),
        },
        "unique_contests": {
            "processed": int(processed["contest_id"].nunique(dropna=True)),
            "feature_table": int(features["contest_id"].nunique(dropna=True)),
        },
        "rating_range": {
            "min": _finite_int(rating.min()),
            "max": _finite_int(rating.max()),
        },
        "rating_distribution": {
            "mean": _finite_float(rating.mean()),
            "median": _finite_float(rating.median()),
        },
        "points_missing_count": int(processed["points"].isna().sum()),
        "solved_count_missing_count": int(solved_count.isna().sum()),
        "solved_count_quantiles": build_solved_count_quantiles(processed),
        "top_tags": top_tags,
        "split_sizes": {
            "contest_grouped": _split_size_summary(contest_split),
            "forward_time": _split_size_summary(time_split),
        },
    }


def prepare_histogram_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return clean numeric values for histogram plotting."""
    _require_columns(frame, (column,), "plot source table")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return values.sort_values(kind="mergesort").reset_index(drop=True)


def prepare_log1p_histogram_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return non-negative log1p-transformed values for plotting."""
    values = prepare_histogram_values(frame, column)
    values = values.loc[values >= 0]
    return np.log1p(values).sort_values(kind="mergesort").reset_index(drop=True)


def prepare_p99_clipped_histogram_values(
    frame: pd.DataFrame,
    column: str,
) -> tuple[pd.Series, float | None]:
    """Return values clipped at the 99th percentile and the clip threshold."""
    values = prepare_histogram_values(frame, column).astype(float)
    if values.empty:
        return values, None
    threshold = float(values.quantile(0.99))
    clipped = values.clip(upper=threshold)
    return clipped.sort_values(kind="mergesort").reset_index(drop=True), threshold


def prepare_top_tags_plot_data(
    tag_frequency: pd.DataFrame,
    top_n: int = TOP_N_TAGS,
) -> pd.DataFrame:
    """Return deterministic top-tag rows for horizontal bar plotting."""
    _require_columns(tag_frequency, ("tag", "count"), "tag frequency table")
    if top_n < 1:
        raise EDAError("top_n must be at least 1.")
    return (
        tag_frequency.sort_values(
            ["count", "tag"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(top_n)
        .sort_values(["count", "tag"], ascending=[True, False], kind="mergesort")
        .reset_index(drop=True)
    )


def prepare_rating_by_index_boxplot_data(
    frame: pd.DataFrame,
) -> tuple[list[str], list[np.ndarray]]:
    """Return labels and rating arrays grouped by leading index letter."""
    _require_columns(frame, ("rating",), "plot source table")
    with_index = _ensure_index_features(frame)
    groups: list[tuple[int, str, np.ndarray]] = []
    for (rank, letter), group in with_index.groupby(
        ["index_rank", "index_letter"],
        dropna=False,
        sort=True,
    ):
        values = pd.to_numeric(group["rating"], errors="coerce").dropna()
        if values.empty:
            continue
        groups.append((int(rank), str(letter), values.to_numpy(dtype=float)))
    groups.sort(key=lambda item: (item[0], item[1]))
    return [letter for _rank, letter, _values in groups], [
        values for _rank, _letter, values in groups
    ]


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Persist a matplotlib figure with stable output settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "cf-diff-eda"},
    )
    plt.close(fig)


def _empty_figure(title: str) -> plt.Figure:
    """Create a readable placeholder figure for empty inputs."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    ax.set_title(title)
    ax.set_axis_off()
    return fig


def plot_histogram(
    values: pd.Series,
    path: Path,
    *,
    title: str,
    xlabel: str,
    bins: int = 30,
) -> None:
    """Save a clean histogram figure."""
    if values.empty:
        _save_figure(_empty_figure(title), path)
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(values, bins=bins, color="#4C78A8", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of problems")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, path)


def plot_p99_clipped_histogram(
    values: pd.Series,
    threshold: float | None,
    path: Path,
    *,
    bins: int = 40,
) -> None:
    """Save a solved-count histogram clipped at the 99th percentile."""
    title = "Distribution of solved counts clipped at p99"
    if values.empty or threshold is None:
        _save_figure(_empty_figure(title), path)
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(values, bins=bins, color="#4C78A8", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(f"Solved count, clipped at p99 = {threshold:,.0f}")
    ax.set_ylabel("Number of problems")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, path)


def plot_index_distribution(index_distribution: pd.DataFrame, path: Path) -> None:
    """Save a bar chart of problem counts by problem index."""
    if index_distribution.empty:
        _save_figure(_empty_figure("Problems by index"), path)
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        index_distribution["index_letter"].astype(str),
        index_distribution["problem_count"],
        color="#59A14F",
    )
    ax.set_title("Problems by leading index")
    ax.set_xlabel("Problem index")
    ax.set_ylabel("Number of problems")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, path)


def plot_top_tags(tag_frequency: pd.DataFrame, path: Path) -> None:
    """Save a horizontal bar chart of the most common tags."""
    plot_data = prepare_top_tags_plot_data(tag_frequency, TOP_N_TAGS)
    if plot_data.empty:
        _save_figure(_empty_figure("Top Codeforces tags"), path)
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_data["tag"], plot_data["count"], color="#F28E2B")
    ax.set_title(f"Top {min(TOP_N_TAGS, len(plot_data))} Codeforces tags")
    ax.set_xlabel("Number of problems")
    ax.set_ylabel("Tag")
    ax.grid(axis="x", alpha=0.25)
    _save_figure(fig, path)


def plot_rating_by_index_boxplot(frame: pd.DataFrame, path: Path) -> None:
    """Save a boxplot of official ratings grouped by problem index."""
    labels, values = prepare_rating_by_index_boxplot_data(frame)
    if not values:
        _save_figure(_empty_figure("Rating by problem index"), path)
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    try:
        ax.boxplot(
            values,
            tick_labels=labels,
            showfliers=False,
            patch_artist=True,
        )
    except TypeError:
        ax.boxplot(
            values,
            labels=labels,
            showfliers=False,
            patch_artist=True,
        )
    for patch in ax.patches:
        patch.set_facecolor("#B07AA1")
        patch.set_alpha(0.65)
    ax.set_title("Rating distribution by leading index")
    ax.set_xlabel("Problem index")
    ax.set_ylabel("Official rating")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, path)


def plot_rating_vs_log_solved_count(frame: pd.DataFrame, path: Path) -> None:
    """Save a scatterplot comparing rating and log solve count."""
    _require_columns(frame, ("rating", "log_solved_count"), "feature table")
    plot_data = frame.loc[:, ["rating", "log_solved_count"]].copy()
    plot_data["rating"] = pd.to_numeric(plot_data["rating"], errors="coerce")
    plot_data["log_solved_count"] = pd.to_numeric(
        plot_data["log_solved_count"],
        errors="coerce",
    )
    plot_data = plot_data.dropna()
    if plot_data.empty:
        _save_figure(_empty_figure("Rating vs. log solve count"), path)
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(
        plot_data["log_solved_count"],
        plot_data["rating"],
        s=12,
        alpha=0.35,
        color="#E15759",
        edgecolors="none",
    )
    ax.set_title("Rating vs. log solved count")
    ax.set_xlabel("log1p(solved count)")
    ax.set_ylabel("Official rating")
    ax.grid(alpha=0.25)
    _save_figure(fig, path)


def run_eda(
    *,
    config_path: Path,
    processed_path: Path,
    feature_path: Path,
    contest_split_path: Path,
    time_split_path: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Path]:
    """Run the full local EDA pipeline and write all required artifacts."""
    logger = configure_logger(log_path)
    try:
        processed = pd.read_parquet(processed_path, engine="pyarrow")
        features = pd.read_parquet(feature_path, engine="pyarrow")
        contest_split = pd.read_parquet(contest_split_path, engine="pyarrow")
        time_split = pd.read_parquet(time_split_path, engine="pyarrow")
        _require_columns(
            processed,
            ("contest_id", "index", "rating", "points", "tags", "solved_count"),
            "processed problem table",
        )
        _require_columns(
            features,
            (
                "contest_id",
                "index",
                "rating",
                "index_letter",
                "log_solved_count",
                "solved_count",
            ),
            "feature table",
        )

        output_dir = output_dir.resolve()
        summary_dir = output_dir / SUMMARY_DIR_NAME
        figure_dir = output_dir / FIGURE_DIR_NAME
        summary_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        tag_frequency = build_tag_frequency(processed)
        index_distribution = build_index_distribution(features)
        rating_band_summary = build_rating_band_summary(processed)
        dataset_summary = build_dataset_summary(
            processed,
            features,
            contest_split,
            time_split,
            tag_frequency,
            config_path=config_path,
            processed_path=processed_path,
            feature_path=feature_path,
            contest_split_path=contest_split_path,
            time_split_path=time_split_path,
        )

        paths = {
            "dataset_summary": summary_dir / "dataset_summary.json",
            "tag_frequency": summary_dir / "tag_frequency.csv",
            "index_distribution": summary_dir / "index_distribution.csv",
            "rating_band_summary": summary_dir / "rating_band_summary.csv",
            "rating_histogram": figure_dir / "rating_histogram.png",
            "solved_count_histogram": figure_dir / "solved_count_histogram.png",
            "solved_count_hist_log": figure_dir / "solved_count_hist_log.png",
            "solved_count_hist_p99": figure_dir / "solved_count_hist_p99.png",
            "log_solved_count_histogram": (
                figure_dir / "log_solved_count_histogram.png"
            ),
            "problems_by_index": figure_dir / "problems_by_index.png",
            "top_tags": figure_dir / "top_tags.png",
            "rating_by_index_boxplot": figure_dir / "rating_by_index_boxplot.png",
            "rating_vs_log_solved_count": (
                figure_dir / "rating_vs_log_solved_count.png"
            ),
        }

        write_json(paths["dataset_summary"], dataset_summary)
        tag_frequency.to_csv(paths["tag_frequency"], index=False)
        index_distribution.to_csv(paths["index_distribution"], index=False)
        rating_band_summary.to_csv(paths["rating_band_summary"], index=False)

        plot_histogram(
            prepare_histogram_values(processed, "rating"),
            paths["rating_histogram"],
            title="Distribution of official Codeforces ratings",
            xlabel="Official rating",
            bins=30,
        )
        plot_histogram(
            prepare_histogram_values(processed, "solved_count"),
            paths["solved_count_histogram"],
            title="Distribution of solved counts",
            xlabel="Solved count",
            bins=40,
        )
        plot_histogram(
            prepare_log1p_histogram_values(processed, "solved_count"),
            paths["solved_count_hist_log"],
            title="Distribution of log1p solved counts",
            xlabel="log1p(solved count)",
            bins=40,
        )
        clipped_solved_count, p99_threshold = prepare_p99_clipped_histogram_values(
            processed,
            "solved_count",
        )
        plot_p99_clipped_histogram(
            clipped_solved_count,
            p99_threshold,
            paths["solved_count_hist_p99"],
            bins=40,
        )
        plot_histogram(
            prepare_histogram_values(features, "log_solved_count"),
            paths["log_solved_count_histogram"],
            title="Distribution of log solved counts",
            xlabel="log1p(solved count)",
            bins=30,
        )
        plot_index_distribution(index_distribution, paths["problems_by_index"])
        plot_top_tags(tag_frequency, paths["top_tags"])
        plot_rating_by_index_boxplot(features, paths["rating_by_index_boxplot"])
        plot_rating_vs_log_solved_count(
            features,
            paths["rating_vs_log_solved_count"],
        )

        logger.info(
            "Completed Codeforces EDA",
            extra={
                "event": "eda_completed",
                "details": {
                    "processed_rows": len(processed),
                    "feature_rows": len(features),
                    "output_dir": output_dir.as_posix(),
                },
            },
        )
        return paths
    except Exception:
        logger.exception(
            "Codeforces EDA failed",
            extra={"event": "eda_failed"},
        )
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the EDA command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate Codeforces EDA tables and figures."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=DEFAULT_PROCESSED_PATH,
    )
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
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
    """Run the EDA CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        paths = run_eda(
            config_path=args.config,
            processed_path=args.processed_path,
            feature_path=args.feature_path,
            contest_split_path=args.contest_split_path,
            time_split_path=args.time_split_path,
            output_dir=args.output_dir,
            log_path=args.log_path,
        )
    except (EDAError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote EDA summary: {paths['dataset_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
