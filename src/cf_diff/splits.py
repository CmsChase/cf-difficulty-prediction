"""Generate leakage-resistant contest-grouped and forward-time splits."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.utils import check_random_state

from cf_diff.features import (
    DEFAULT_CONFIG_PATH,
    ExperimentConfig,
    FeatureError,
    JsonLogFormatter,
    SplitRatios,
    experiment_config_fingerprint,
    load_experiment_config,
    write_json,
)

DEFAULT_LOG_PATH: Final[Path] = Path("outputs/logs/splits.log")
SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "valid", "test")


class SplitError(RuntimeError):
    """Raised when leakage-safe split generation is impossible."""


def configure_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """Create the dedicated structured split-generation logger."""
    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.splits")
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
    """Flush and close all handlers attached to the split logger."""
    for handler in tuple(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _partition_counts(total: int, ratios: SplitRatios) -> dict[str, int]:
    """Allocate positive deterministic partition counts summing to total."""
    values = {
        "train": ratios.train,
        "valid": ratios.valid,
        "test": ratios.test,
    }
    if total < len(values):
        raise SplitError(
            f"At least {len(values)} groups are required; found {total}."
        )
    raw = {name: total * values[name] for name in SPLIT_NAMES}
    floors = {name: math.floor(raw[name]) for name in SPLIT_NAMES}
    counts = dict(floors)
    left = total - sum(floors.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(raw[name] - floors[name]), SPLIT_NAMES.index(name)),
    )
    for name in order[:left]:
        counts[name] += 1

    for empty_name in (
        name for name in SPLIT_NAMES if counts[name] == 0
    ):
        donor = max(
            (
                name
                for name in SPLIT_NAMES
                if counts[name] > 1
            ),
            key=lambda name: (counts[name], values[name], -SPLIT_NAMES.index(name)),
        )
        counts[donor] -= 1
        counts[empty_name] += 1
    return counts


def _validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate identifiers and ensure one timestamp per contest."""
    required = ("contest_id", "index", "start_time_seconds")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SplitError(f"Model table lacks required columns: {missing}")
    result = frame.copy()
    result["contest_id"] = pd.to_numeric(
        result["contest_id"],
        errors="coerce",
    ).astype("Int64")
    result["start_time_seconds"] = pd.to_numeric(
        result["start_time_seconds"],
        errors="coerce",
    ).astype("Int64")
    result["index"] = result["index"].astype("string")
    if result[["contest_id", "index"]].isna().any().any():
        raise SplitError("Split identifiers contest_id and index cannot be null.")
    duplicated = result.duplicated(
        subset=["contest_id", "index"],
        keep=False,
    )
    if duplicated.any():
        raise SplitError(
            f"Model table has {int(duplicated.sum())} duplicate identifiers."
        )
    time_counts = (
        result.dropna(subset=["start_time_seconds"])
        .groupby("contest_id")["start_time_seconds"]
        .nunique()
    )
    if (time_counts > 1).any():
        bad = time_counts[time_counts > 1].index.tolist()
        raise SplitError(f"Contests have inconsistent start times: {bad[:5]}")
    return result


def _assignment_table(
    frame: pd.DataFrame,
    contest_to_split: Mapping[int, str],
    strategy: str,
) -> pd.DataFrame:
    """Create the required row-level split assignment schema."""
    assignment = frame.loc[:, ["contest_id", "index"]].copy()
    assignment["split_name"] = assignment["contest_id"].map(
        contest_to_split
    )
    if assignment["split_name"].isna().any():
        raise SplitError(f"{strategy} did not assign every contest.")
    fold_map = {"train": 0, "valid": 1, "test": 2}
    assignment["fold"] = assignment["split_name"].map(fold_map).astype("Int8")
    assignment["strategy"] = strategy
    return assignment.sort_values(
        ["fold", "contest_id", "index"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_contest_grouped_split(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Assign whole contests to randomized deterministic partitions."""
    validated = _validate_input(frame)
    contests = sorted(int(value) for value in validated["contest_id"].unique())
    counts = _partition_counts(len(contests), config.grouped_split)
    random_state = check_random_state(config.random_seed)
    shuffled = random_state.permutation(contests).tolist()
    train_end = counts["train"]
    valid_end = train_end + counts["valid"]
    contest_to_split = {
        **{contest: "train" for contest in shuffled[:train_end]},
        **{
            contest: "valid"
            for contest in shuffled[train_end:valid_end]
        },
        **{contest: "test" for contest in shuffled[valid_end:]},
    }
    return _assignment_table(
        validated,
        contest_to_split,
        "contest_grouped",
    )


def build_forward_time_split(
    frame: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Assign complete timestamp buckets in chronological order."""
    validated = _validate_input(frame)
    if validated["start_time_seconds"].isna().any():
        count = int(validated["start_time_seconds"].isna().sum())
        raise SplitError(
            f"Forward-time split requires start_time_seconds; {count} rows "
            "are missing it."
        )
    contest_times = (
        validated.loc[:, ["contest_id", "start_time_seconds"]]
        .drop_duplicates()
        .sort_values(
            ["start_time_seconds", "contest_id"],
            kind="mergesort",
        )
    )
    time_buckets = [
        (int(timestamp), sorted(int(value) for value in group["contest_id"]))
        for timestamp, group in contest_times.groupby(
            "start_time_seconds",
            sort=True,
        )
    ]
    counts = _partition_counts(
        len(time_buckets),
        config.forward_time_split,
    )
    train_end = counts["train"]
    valid_end = train_end + counts["valid"]
    bucket_splits = (
        [("train", bucket) for bucket in time_buckets[:train_end]]
        + [
            ("valid", bucket)
            for bucket in time_buckets[train_end:valid_end]
        ]
        + [("test", bucket) for bucket in time_buckets[valid_end:]]
    )
    contest_to_split = {
        contest: split_name
        for split_name, (_timestamp, contests) in bucket_splits
        for contest in contests
    }
    assignment = _assignment_table(
        validated,
        contest_to_split,
        "forward_time",
    )
    time_by_contest = contest_times.set_index("contest_id")[
        "start_time_seconds"
    ]
    assigned_times = assignment.assign(
        start_time_seconds=assignment["contest_id"].map(time_by_contest)
    )
    train_max = assigned_times.loc[
        assigned_times["split_name"].eq("train"),
        "start_time_seconds",
    ].max()
    valid_min = assigned_times.loc[
        assigned_times["split_name"].eq("valid"),
        "start_time_seconds",
    ].min()
    valid_max = assigned_times.loc[
        assigned_times["split_name"].eq("valid"),
        "start_time_seconds",
    ].max()
    test_min = assigned_times.loc[
        assigned_times["split_name"].eq("test"),
        "start_time_seconds",
    ].min()
    if not (train_max < valid_min and valid_max < test_min):
        raise SplitError("Forward-time partitions are not strictly ordered.")
    return assignment


def _split_counts(assignment: pd.DataFrame) -> dict[str, int]:
    """Return row counts for all expected split names."""
    counts = assignment["split_name"].value_counts()
    return {name: int(counts.get(name, 0)) for name in SPLIT_NAMES}


def _contest_counts(assignment: pd.DataFrame) -> dict[str, int]:
    """Return unique contest counts for all expected split names."""
    grouped = assignment.groupby("split_name")["contest_id"].nunique()
    return {name: int(grouped.get(name, 0)) for name in SPLIT_NAMES}


def _contest_overlap(assignment: pd.DataFrame) -> int:
    """Count contest ids appearing in more than one partition."""
    per_contest = assignment.groupby("contest_id")["split_name"].nunique()
    return int((per_contest > 1).sum())


def build_split_summary(
    model_table: pd.DataFrame,
    grouped: pd.DataFrame,
    forward: pd.DataFrame,
    config: ExperimentConfig,
) -> dict[str, object]:
    """Build auditable statistics for both split strategies."""
    time_lookup = model_table.set_index(
        ["contest_id", "index"]
    )["start_time_seconds"]
    forward_with_time = forward.copy()
    forward_with_time["start_time_seconds"] = [
        time_lookup.loc[(contest_id, index)]
        for contest_id, index in zip(
            forward_with_time["contest_id"],
            forward_with_time["index"],
            strict=True,
        )
    ]
    time_ranges: dict[str, dict[str, int | None]] = {}
    for name in SPLIT_NAMES:
        values = forward_with_time.loc[
            forward_with_time["split_name"].eq(name),
            "start_time_seconds",
        ].dropna()
        time_ranges[name] = {
            "min": int(values.min()) if not values.empty else None,
            "max": int(values.max()) if not values.empty else None,
        }
    overlap = _contest_overlap(grouped)
    return {
        "random_seed": config.random_seed,
        "effective_config": asdict(config),
        "config_fingerprint_sha256": experiment_config_fingerprint(config),
        "contest_grouped": {
            "row_counts": _split_counts(grouped),
            "contest_counts": _contest_counts(grouped),
            "contest_overlap_count": overlap,
            "zero_contest_overlap": overlap == 0,
        },
        "forward_time": {
            "row_counts": _split_counts(forward),
            "contest_counts": _contest_counts(forward),
            "time_ranges": time_ranges,
            "strictly_ordered": (
                time_ranges["train"]["max"] < time_ranges["valid"]["min"]
                and time_ranges["valid"]["max"] < time_ranges["test"]["min"]
            ),
        },
    }


def generate_splits(
    input_path: Path,
    output_dir: Path,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
) -> dict[str, Path]:
    """Generate both split strategies and write their audit artifacts."""
    logger = configure_logger(log_path)
    try:
        config = load_experiment_config(config_path)
        model_table = pd.read_parquet(input_path, engine="pyarrow")
        grouped = build_contest_grouped_split(model_table, config)
        forward = build_forward_time_split(model_table, config)
        summary = build_split_summary(
            model_table,
            grouped,
            forward,
            config,
        )
        config_fingerprint = experiment_config_fingerprint(config)
        grouped["config_fingerprint_sha256"] = config_fingerprint
        forward["config_fingerprint_sha256"] = config_fingerprint
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "contest_grouped_split": (
                output_dir / "contest_grouped_split.parquet"
            ),
            "forward_time_split": output_dir / "forward_time_split.parquet",
            "split_summary": output_dir / "split_summary.json",
        }
        grouped.to_parquet(
            paths["contest_grouped_split"],
            engine="pyarrow",
            index=False,
        )
        forward.to_parquet(
            paths["forward_time_split"],
            engine="pyarrow",
            index=False,
        )
        write_json(paths["split_summary"], summary)
        logger.info(
            "Completed Codeforces split generation",
            extra={
                "event": "splits_completed",
                "details": summary,
            },
        )
        return paths
    except Exception:
        logger.exception(
            "Codeforces split generation failed",
            extra={"event": "splits_failed"},
        )
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the split-generation command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate Codeforces evaluation splits."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the split generation CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        paths = generate_splits(
            args.input_path,
            args.output_dir,
            config_path=args.config,
            log_path=args.log_path,
        )
    except (
        FeatureError,
        SplitError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote split summary: {paths['split_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
