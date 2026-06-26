"""Clean raw Codeforces API snapshots into research-ready Parquet tables."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_diff import RANDOM_SEED

PROBLEMSET_ENTRY: Final[str] = "problemset_problems"
CONTEST_ENTRY: Final[str] = "contest_list"
JOIN_KEYS: Final[tuple[str, str]] = ("contest_id", "index")


class DataCleaningError(RuntimeError):
    """Raised when raw snapshots cannot be cleaned safely."""


def to_snake_case(name: str) -> str:
    """Convert one API field name to lowercase snake_case."""
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return value.strip("_").lower()


def normalize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose columns use unique snake_case names."""
    normalized = frame.copy()
    normalized.columns = [to_snake_case(str(column)) for column in frame.columns]
    if normalized.columns.duplicated().any():
        duplicates = sorted(
            set(normalized.columns[normalized.columns.duplicated()].tolist())
        )
        raise DataCleaningError(
            f"Column normalization produced duplicate names: {duplicates}"
        )
    return normalized


def read_json_payload(path: Path) -> dict[str, object]:
    """Read and validate a raw Codeforces API JSON response object."""
    try:
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except json.JSONDecodeError as error:
        raise DataCleaningError(f"Invalid JSON in {path}: {error}") from error

    if not isinstance(payload, dict):
        raise DataCleaningError(f"Raw snapshot {path} must contain a JSON object.")

    status = payload.get("status")
    if status != "OK":
        comment = payload.get("comment")
        detail = f": {comment}" if comment else ""
        raise DataCleaningError(
            f"Raw snapshot {path} has API status {status!r}{detail}"
        )
    if "result" not in payload:
        raise DataCleaningError(
            f"Raw snapshot {path} lacks a top-level 'result' field."
        )
    return payload


def resolve_input_paths(
    *,
    manifest_path: Path | None,
    problemset_snapshot: Path | None,
    contest_snapshot: Path | None,
) -> tuple[Path, Path]:
    """Resolve raw snapshot paths from a manifest or an explicit path pair."""
    explicit_supplied = (
        problemset_snapshot is not None or contest_snapshot is not None
    )
    if manifest_path is not None and explicit_supplied:
        raise DataCleaningError(
            "Provide either --manifest or the two explicit snapshot paths, not both."
        )
    if manifest_path is None:
        if problemset_snapshot is None or contest_snapshot is None:
            raise DataCleaningError(
                "Provide --manifest, or provide both --problemset-snapshot "
                "and --contest-snapshot."
            )
        return problemset_snapshot.resolve(), contest_snapshot.resolve()

    manifest = read_json_payload_without_api_contract(manifest_path)
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        raise DataCleaningError(
            f"Manifest {manifest_path} lacks an object-valued 'entries' field."
        )

    return (
        _resolve_manifest_entry(manifest_path, entries, PROBLEMSET_ENTRY),
        _resolve_manifest_entry(manifest_path, entries, CONTEST_ENTRY),
    )


def read_json_payload_without_api_contract(path: Path) -> dict[str, object]:
    """Read a general JSON object such as a snapshot manifest."""
    try:
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except json.JSONDecodeError as error:
        raise DataCleaningError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataCleaningError(f"JSON file {path} must contain an object.")
    return payload


def _resolve_manifest_entry(
    manifest_path: Path,
    entries: Mapping[str, object],
    entry_name: str,
) -> Path:
    """Resolve one snapshot path stored in a latest-snapshot manifest."""
    entry = entries.get(entry_name)
    if not isinstance(entry, dict):
        raise DataCleaningError(
            f"Manifest {manifest_path} lacks entry {entry_name!r}."
        )
    snapshot_path = entry.get("snapshot_path")
    if not isinstance(snapshot_path, str) or not snapshot_path:
        raise DataCleaningError(
            f"Manifest entry {entry_name!r} lacks a valid 'snapshot_path'."
        )

    path = Path(snapshot_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def extract_tables(
    problemset_payload: Mapping[str, object],
    contest_payload: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract normalized problems, statistics, and contests tables."""
    problemset_result = problemset_payload.get("result")
    if not isinstance(problemset_result, dict):
        raise DataCleaningError(
            "problemset.problems 'result' must be a JSON object."
        )

    problems_data = problemset_result.get("problems")
    statistics_data = problemset_result.get("problemStatistics")
    contests_data = contest_payload.get("result")
    if not isinstance(problems_data, list):
        raise DataCleaningError(
            "problemset.problems result must contain a 'problems' list."
        )
    if not isinstance(statistics_data, list):
        raise DataCleaningError(
            "problemset.problems result must contain a 'problemStatistics' list."
        )
    if not isinstance(contests_data, list):
        raise DataCleaningError("contest.list 'result' must be a list.")

    problems = normalize_column_names(
        pd.json_normalize(problems_data, sep="_")
    )
    problem_statistics = normalize_column_names(
        pd.json_normalize(statistics_data, sep="_")
    )
    contests = normalize_column_names(
        pd.json_normalize(contests_data, sep="_")
    )

    problems = _prepare_problems(problems)
    problem_statistics = _prepare_problem_statistics(problem_statistics)
    contests = _prepare_contests(contests)
    return problems, problem_statistics, contests


def _ensure_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Return a copy containing each requested optional column."""
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result


def _prepare_problems(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize required and optional problem fields."""
    result = _ensure_columns(
        frame,
        ("contest_id", "index", "type", "rating", "points", "tags"),
    )
    result["contest_id"] = pd.to_numeric(
        result["contest_id"], errors="coerce"
    ).astype("Int64")
    result["rating"] = pd.to_numeric(
        result["rating"], errors="coerce"
    ).astype("Int64")
    result["points"] = pd.to_numeric(
        result["points"], errors="coerce"
    ).astype("Float64")
    result["index"] = result["index"].astype("string")
    result["type"] = result["type"].astype("string")
    result["tags"] = result["tags"].map(_normalize_tags)
    return result


def _normalize_tags(value: object) -> list[str]:
    """Normalize an optional Codeforces tags value to a list of strings."""
    if isinstance(value, list):
        return [str(tag) for tag in value]
    if isinstance(value, tuple):
        return [str(tag) for tag in value]
    return []


def _prepare_problem_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize problem-statistics join keys and solved counts."""
    result = _ensure_columns(frame, (*JOIN_KEYS, "solved_count"))
    result["contest_id"] = pd.to_numeric(
        result["contest_id"], errors="coerce"
    ).astype("Int64")
    result["index"] = result["index"].astype("string")
    result["solved_count"] = pd.to_numeric(
        result["solved_count"], errors="coerce"
    ).astype("Int64")
    return result


def _prepare_contests(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize contest identifiers and required metadata fields."""
    result = _ensure_columns(
        frame,
        ("id", "type", "phase", "start_time_seconds", "duration_seconds"),
    )
    for column in ("id", "start_time_seconds", "duration_seconds"):
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        ).astype("Int64")
    result["type"] = result["type"].astype("string")
    result["phase"] = result["phase"].astype("string")
    return result


def merge_tables(
    problems: pd.DataFrame,
    problem_statistics: pd.DataFrame,
    contests: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Merge normalized API tables and return per-column source metadata."""
    statistics_rename = {
        column: f"{column}_statistics"
        for column in problem_statistics.columns
        if column not in JOIN_KEYS and column in problems.columns
    }
    statistics_for_merge = problem_statistics.rename(
        columns=statistics_rename
    )
    merged = problems.merge(
        statistics_for_merge,
        how="left",
        on=list(JOIN_KEYS),
        sort=False,
        validate="one_to_one",
    )

    contest_rename = {
        column: (
            "contest_metadata_id"
            if column == "id"
            else f"contest_{column}"
        )
        for column in contests.columns
    }
    contests_for_merge = contests.rename(columns=contest_rename)
    collisions = sorted(set(merged.columns) & set(contests_for_merge.columns))
    if collisions:
        raise DataCleaningError(
            f"Contest merge would overwrite columns: {collisions}"
        )

    merged = merged.merge(
        contests_for_merge,
        how="left",
        left_on="contest_id",
        right_on="contest_metadata_id",
        sort=False,
        validate="many_to_one",
    )

    source_map = {column: "problems" for column in problems.columns}
    source_map.update(
        {
            statistics_rename.get(column, column): "problem_statistics"
            for column in problem_statistics.columns
            if column not in JOIN_KEYS
        }
    )
    source_map.update(
        {
            renamed_column: "contests"
            for renamed_column in contest_rename.values()
        }
    )
    return merged, source_map


def build_model_ready(merged: pd.DataFrame) -> pd.DataFrame:
    """Filter merged rows and add the requested deterministic model features."""
    required_columns = (
        "type",
        "rating",
        "solved_count",
        "contest_id",
        "index",
        "contest_metadata_id",
        "points",
        "tags",
        "contest_start_time_seconds",
        "contest_duration_seconds",
        "contest_phase",
        "contest_type",
    )
    missing = [column for column in required_columns if column not in merged.columns]
    if missing:
        raise DataCleaningError(
            f"Merged table lacks required modeling columns: {missing}"
        )

    keep = (
        merged["type"].eq("PROGRAMMING")
        & merged["rating"].notna()
        & merged["solved_count"].notna()
        & merged["contest_id"].notna()
        & merged["index"].notna()
        & merged["contest_metadata_id"].notna()
    )
    model = merged.loc[keep].copy().reset_index(drop=True)
    model = model.drop(columns=["contest_metadata_id"])

    if (model["solved_count"] < 0).any():
        raise DataCleaningError("solved_count cannot be negative.")

    model["has_points"] = model["points"].notna().astype("Int8")
    model["tag_count"] = model["tags"].map(len).astype("Int64")
    model["log_solved_count"] = model["solved_count"].map(
        lambda value: math.log1p(int(value))
    )

    index_values = model["index"].astype("string")
    index_letter = index_values.str.extract(
        r"^([A-Za-z]+)", expand=False
    ).str.upper()
    index_suffix = index_values.str.extract(r"(\d+)$", expand=False)
    model["index_letter"] = index_letter.astype("string")
    model["index_suffix"] = pd.to_numeric(
        index_suffix, errors="coerce"
    ).astype("Int64")
    model["index_rank"] = index_letter.str[0].map(
        {
            letter: rank
            for rank, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ", start=1)
        }
    ).astype("Int64")
    return model


def write_data_dictionary(
    frame: pd.DataFrame,
    path: Path,
    source_map: Mapping[str, str],
) -> None:
    """Write one markdown dictionary row for every model-ready column."""
    derived_sources = {
        "has_points": "problems",
        "tag_count": "problems",
        "log_solved_count": "problem_statistics",
        "contest_start_time_seconds": "contests",
        "contest_duration_seconds": "contests",
        "contest_phase": "contests",
        "contest_type": "contests",
        "index_letter": "problems",
        "index_suffix": "problems",
        "index_rank": "problems",
    }
    descriptions = {
        "has_points": "1 when problem points are present, otherwise 0.",
        "tag_count": "Number of Codeforces tags attached to the problem.",
        "log_solved_count": "Natural logarithm of one plus solved_count.",
        "contest_start_time_seconds": (
            "Contest start time as Unix seconds from the contest API."
        ),
        "contest_duration_seconds": "Contest duration in seconds.",
        "contest_phase": "Contest lifecycle phase.",
        "contest_type": "Codeforces contest type.",
        "index_letter": "Leading alphabetic part of index, uppercased.",
        "index_suffix": "Trailing digits of index as a nullable integer.",
        "index_rank": "Alphabet rank of the first index letter, A=1 through Z=26.",
        "tags": "List of Codeforces problem tags.",
        "solved_count": "Number of accepted solvers reported by Codeforces.",
        "rating": "Codeforces problem difficulty rating.",
        "points": "Problem point value when supplied by Codeforces.",
    }
    derived_columns = set(derived_sources)

    lines = [
        "# Codeforces Model-Ready Data Dictionary",
        "",
        "| column_name | dtype | source_table | description | is_derived |",
        "|---|---|---|---|---|",
    ]
    for column in frame.columns:
        source = derived_sources.get(column, source_map.get(column, "unknown"))
        description = descriptions.get(
            column,
            f"Normalized API field `{column}` from the {source} table.",
        )
        values = (
            column,
            str(frame[column].dtype),
            source,
            description,
            "yes" if column in derived_columns else "no",
        )
        escaped = [str(value).replace("|", r"\|") for value in values]
        lines.append(f"| {' | '.join(escaped)} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def clean_snapshots(
    problemset_snapshot: Path,
    contest_snapshot: Path,
    output_root: Path,
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, Path]:
    """Clean two raw snapshots and write all interim and processed outputs."""
    random.seed(seed)
    problemset_payload = read_json_payload(problemset_snapshot)
    contest_payload = read_json_payload(contest_snapshot)
    problems, problem_statistics, contests = extract_tables(
        problemset_payload,
        contest_payload,
    )
    merged, source_map = merge_tables(
        problems,
        problem_statistics,
        contests,
    )
    model_ready = build_model_ready(merged)

    output_root = output_root.resolve()
    interim_dir = output_root / "interim"
    processed_dir = output_root / "processed"
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "problems": interim_dir / "problems.parquet",
        "problem_statistics": interim_dir / "problem_statistics.parquet",
        "contests": interim_dir / "contests.parquet",
        "problems_merged": processed_dir / "problems_merged.parquet",
        "problems_model_ready": (
            processed_dir / "problems_model_ready.parquet"
        ),
        "data_dictionary": processed_dir / "data_dictionary.md",
    }
    problems.to_parquet(output_paths["problems"], engine="pyarrow", index=False)
    problem_statistics.to_parquet(
        output_paths["problem_statistics"],
        engine="pyarrow",
        index=False,
    )
    contests.to_parquet(output_paths["contests"], engine="pyarrow", index=False)
    merged.to_parquet(
        output_paths["problems_merged"],
        engine="pyarrow",
        index=False,
    )
    model_ready.to_parquet(
        output_paths["problems_model_ready"],
        engine="pyarrow",
        index=False,
    )
    write_data_dictionary(
        model_ready,
        output_paths["data_dictionary"],
        source_map,
    )
    return output_paths


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Clean raw Codeforces snapshots into Parquet tables."
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--problemset-snapshot", type=Path)
    parser.add_argument("--contest-snapshot", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Project random seed (default: {RANDOM_SEED}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic Codeforces data-cleaning CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        problemset_path, contest_path = resolve_input_paths(
            manifest_path=args.manifest,
            problemset_snapshot=args.problemset_snapshot,
            contest_snapshot=args.contest_snapshot,
        )
        outputs = clean_snapshots(
            problemset_path,
            contest_path,
            args.output_root,
            seed=args.seed,
        )
    except (DataCleaningError, OSError, ValueError, pd.errors.MergeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        "Wrote cleaned Codeforces data: "
        f"{outputs['problems_model_ready']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
