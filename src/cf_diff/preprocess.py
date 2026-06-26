"""Preprocess raw Codeforces API snapshots into documented Parquet tables."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pandas as pd

PROBLEMSET_FILENAME: Final[str] = "problemset.problems.json"
CONTEST_FILENAME: Final[str] = "contest.list.json"
MANIFEST_FILENAME: Final[str] = "manifest.json"
JOIN_KEYS: Final[tuple[str, str]] = ("contest_id", "index")
KEY_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "contest_id",
    "index",
    "type",
    "rating",
    "solved_count",
    "start_time_seconds",
)


class PreprocessError(RuntimeError):
    """Raised when a raw snapshot cannot be preprocessed safely."""


class JsonLogFormatter(logging.Formatter):
    """Format preprocessing log records as JSON Lines."""

    def format(self, record: logging.LogRecord) -> str:
        """Return one machine-readable JSON log line."""
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


def configure_logger(log_path: Path) -> logging.Logger:
    """Create a dedicated structured preprocessing logger."""
    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.preprocess")
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
    """Flush, close, and detach a dedicated logger's handlers."""
    for handler in tuple(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def to_snake_case(name: str) -> str:
    """Convert a Codeforces API field name to lowercase snake_case."""
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return value.strip("_").lower()


def normalize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with unique snake_case column names."""
    result = frame.copy()
    result.columns = [to_snake_case(str(column)) for column in frame.columns]
    duplicated = result.columns[result.columns.duplicated()].tolist()
    if duplicated:
        raise PreprocessError(
            f"Column normalization produced duplicates: {sorted(set(duplicated))}"
        )
    return result


def read_json_object(path: Path) -> dict[str, object]:
    """Read a UTF-8 JSON file and require a top-level object."""
    try:
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except json.JSONDecodeError as error:
        raise PreprocessError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PreprocessError(f"JSON file {path} must contain an object.")
    return payload


def validate_api_payload(
    payload: Mapping[str, object],
    endpoint: str,
) -> None:
    """Validate the common top-level Codeforces API response contract."""
    status = payload.get("status")
    if status != "OK":
        comment = payload.get("comment")
        detail = f": {comment}" if comment else ""
        raise PreprocessError(
            f"{endpoint} payload has API status {status!r}{detail}"
        )
    if "result" not in payload:
        raise PreprocessError(f"{endpoint} payload lacks top-level 'result'.")


def load_raw_snapshot(
    raw_dir: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Load and validate the manifest and two raw API responses."""
    raw_dir = raw_dir.resolve()
    manifest_path = raw_dir / MANIFEST_FILENAME
    problemset_path = raw_dir / PROBLEMSET_FILENAME
    contest_path = raw_dir / CONTEST_FILENAME
    for path in (manifest_path, problemset_path, contest_path):
        if not path.is_file():
            raise PreprocessError(f"Required raw snapshot file is missing: {path}")

    manifest = read_json_object(manifest_path)
    resources = manifest.get("resources")
    if not isinstance(resources, dict):
        raise PreprocessError(
            f"Manifest {manifest_path} lacks object-valued 'resources'."
        )
    for endpoint in ("problemset.problems", "contest.list"):
        if endpoint not in resources:
            raise PreprocessError(
                f"Manifest {manifest_path} lacks resource {endpoint!r}."
            )

    problemset_payload = read_json_object(problemset_path)
    contest_payload = read_json_object(contest_path)
    validate_api_payload(problemset_payload, "problemset.problems")
    validate_api_payload(contest_payload, "contest.list")
    return problemset_payload, contest_payload, manifest


def _ensure_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Return a copy containing all requested optional columns."""
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            if column == "tags":
                result[column] = [[] for _ in range(len(result))]
            else:
                result[column] = pd.NA
    return result


def _coerce_numeric(
    frame: pd.DataFrame,
    column: str,
    dtype: str,
    *,
    table_name: str,
    invalid_counts: dict[str, int],
    logger: logging.Logger,
) -> None:
    """Coerce one numeric column while reporting values lost to coercion."""
    original = frame[column]
    converted = pd.to_numeric(original, errors="coerce")
    invalid_count = int((original.notna() & converted.isna()).sum())
    invalid_counts[f"{table_name}.{column}"] = invalid_count
    if invalid_count:
        logger.warning(
            "Invalid numeric values were coerced to null",
            extra={
                "event": "numeric_coercion",
                "details": {
                    "table": table_name,
                    "column": column,
                    "invalid_count": invalid_count,
                },
            },
        )
    frame[column] = converted.astype(dtype)


def _normalize_tags(value: object) -> list[str]:
    """Normalize an optional tags value to a list of strings."""
    if isinstance(value, list):
        return [str(tag) for tag in value]
    if isinstance(value, tuple):
        return [str(tag) for tag in value]
    return []


def _order_columns(
    frame: pd.DataFrame,
    preferred: Sequence[str],
) -> pd.DataFrame:
    """Return columns in a stable semantic-first, alphabetical-rest order."""
    preferred_present = [column for column in preferred if column in frame.columns]
    remaining = sorted(
        column for column in frame.columns if column not in preferred_present
    )
    return frame.loc[:, [*preferred_present, *remaining]]


def _sort_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Sort by available columns with stable ordering and a fresh index."""
    available = [column for column in columns if column in frame.columns]
    if not available:
        return frame.reset_index(drop=True)
    return frame.sort_values(
        available,
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def normalize_base_tables(
    problemset_payload: Mapping[str, object],
    contest_payload: Mapping[str, object],
    *,
    logger: logging.Logger,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, int],
]:
    """Normalize raw API objects into three deterministic base tables."""
    problemset_result = problemset_payload.get("result")
    if not isinstance(problemset_result, dict):
        raise PreprocessError("problemset.problems result must be an object.")
    problem_records = problemset_result.get("problems")
    statistic_records = problemset_result.get("problemStatistics")
    contest_records = contest_payload.get("result")
    if not isinstance(problem_records, list):
        raise PreprocessError(
            "problemset.problems result must contain a 'problems' list."
        )
    if not isinstance(statistic_records, list):
        raise PreprocessError(
            "problemset.problems result must contain a "
            "'problemStatistics' list."
        )
    if not isinstance(contest_records, list):
        raise PreprocessError("contest.list result must be a list.")

    problems = normalize_column_names(
        pd.json_normalize(problem_records, sep="_")
    )
    statistics = normalize_column_names(
        pd.json_normalize(statistic_records, sep="_")
    )
    contests = normalize_column_names(
        pd.json_normalize(contest_records, sep="_")
    )
    problems = _ensure_columns(
        problems,
        (
            "contest_id",
            "problemset_name",
            "index",
            "name",
            "type",
            "points",
            "rating",
            "tags",
        ),
    )
    statistics = _ensure_columns(
        statistics,
        ("contest_id", "index", "solved_count"),
    )
    contests = _ensure_columns(
        contests,
        (
            "id",
            "name",
            "type",
            "phase",
            "start_time_seconds",
            "duration_seconds",
        ),
    )

    invalid_counts: dict[str, int] = {}
    for column, dtype in (
        ("contest_id", "Int64"),
        ("points", "Float64"),
        ("rating", "Int64"),
    ):
        _coerce_numeric(
            problems,
            column,
            dtype,
            table_name="problems",
            invalid_counts=invalid_counts,
            logger=logger,
        )
    for column, dtype in (
        ("contest_id", "Int64"),
        ("solved_count", "Int64"),
    ):
        _coerce_numeric(
            statistics,
            column,
            dtype,
            table_name="problem_statistics",
            invalid_counts=invalid_counts,
            logger=logger,
        )
    for column in (
        "id",
        "start_time_seconds",
        "duration_seconds",
        "relative_time_seconds",
        "freeze_duration_seconds",
    ):
        if column in contests.columns:
            _coerce_numeric(
                contests,
                column,
                "Int64",
                table_name="contests",
                invalid_counts=invalid_counts,
                logger=logger,
            )

    for column in ("problemset_name", "index", "name", "type"):
        problems[column] = problems[column].astype("string")
    for column in ("index",):
        statistics[column] = statistics[column].astype("string")
    for column in ("name", "type", "phase"):
        contests[column] = contests[column].astype("string")
    problems["tags"] = problems["tags"].map(_normalize_tags)

    problems = _order_columns(
        problems,
        (
            "contest_id",
            "problemset_name",
            "index",
            "name",
            "type",
            "points",
            "rating",
            "tags",
        ),
    )
    statistics = _order_columns(
        statistics,
        ("contest_id", "index", "solved_count"),
    )
    contests = _order_columns(
        contests,
        (
            "id",
            "name",
            "type",
            "phase",
            "start_time_seconds",
            "duration_seconds",
        ),
    )
    problems = _sort_frame(problems, ("contest_id", "index"))
    statistics = _sort_frame(statistics, ("contest_id", "index"))
    contests = _sort_frame(contests, ("start_time_seconds", "id"))
    return problems, statistics, contests, invalid_counts


def _raise_for_duplicate_keys(
    frame: pd.DataFrame,
    keys: Sequence[str],
    table_name: str,
) -> None:
    """Fail if non-null join keys are duplicated."""
    valid = frame.dropna(subset=list(keys))
    duplicate_count = int(valid.duplicated(subset=list(keys), keep=False).sum())
    if duplicate_count:
        raise PreprocessError(
            f"{table_name} has {duplicate_count} rows with duplicate join keys "
            f"{tuple(keys)}."
        )


def merge_problem_tables(
    problems: pd.DataFrame,
    statistics: pd.DataFrame,
    contests: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    """Left join statistics and contest metadata onto problem records."""
    _raise_for_duplicate_keys(statistics, JOIN_KEYS, "problem_statistics")
    valid_statistics = statistics.dropna(subset=list(JOIN_KEYS)).copy()

    statistic_renames = {
        column: f"{column}_statistics"
        for column in valid_statistics.columns
        if column not in JOIN_KEYS and column in problems.columns
    }
    valid_statistics = valid_statistics.rename(columns=statistic_renames)
    merged = problems.merge(
        valid_statistics,
        how="left",
        on=list(JOIN_KEYS),
        sort=False,
        validate="many_to_one",
    )

    _raise_for_duplicate_keys(contests, ("id",), "contests")
    valid_contests = contests.dropna(subset=["id"]).copy()
    contest_renames: dict[str, str] = {"id": "_contest_join_id"}
    for column in valid_contests.columns:
        if column == "id":
            continue
        contest_renames[column] = (
            f"contest_{column}" if column in merged.columns else column
        )
    valid_contests = valid_contests.rename(columns=contest_renames)
    merged = merged.merge(
        valid_contests,
        how="left",
        left_on="contest_id",
        right_on="_contest_join_id",
        sort=False,
        validate="many_to_one",
    ).drop(columns=["_contest_join_id"])

    source_map: dict[str, tuple[str, str]] = {
        column: ("problemset.problems", "Problem")
        for column in problems.columns
    }
    source_map.update(
        {
            statistic_renames.get(column, column): (
                "problemset.problems",
                "ProblemStatistics",
            )
            for column in statistics.columns
            if column not in JOIN_KEYS
        }
    )
    source_map.update(
        {
            renamed: ("contest.list", "Contest")
            for original, renamed in contest_renames.items()
            if original != "id"
        }
    )
    merged = _order_columns(
        merged,
        (
            "contest_id",
            "problemset_name",
            "index",
            "name",
            "type",
            "points",
            "rating",
            "tags",
            "solved_count",
            "contest_name",
            "contest_type",
            "phase",
            "start_time_seconds",
            "duration_seconds",
        ),
    )
    return _sort_frame(merged, ("contest_id", "index")), source_map


def build_rated_programming_table(
    merged: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter to rated programming problems and return exclusion counts."""
    required = ("type", "rating", "contest_id")
    missing = [column for column in required if column not in merged.columns]
    if missing:
        raise PreprocessError(
            f"Merged problem table lacks required columns: {missing}"
        )

    programming = merged["type"].eq("PROGRAMMING")
    rated = merged["rating"].notna()
    has_contest = merged["contest_id"].notna()
    keep = programming & rated & has_contest
    filter_counts = {
        "input_rows": len(merged),
        "non_programming_rows": int((~programming.fillna(False)).sum()),
        "missing_rating_rows": int(merged["rating"].isna().sum()),
        "missing_contest_id_rows": int(merged["contest_id"].isna().sum()),
        "output_rows": int(keep.sum()),
    }
    result = merged.loc[keep].copy()
    result = _sort_frame(
        result,
        ("start_time_seconds", "contest_id", "index"),
    )
    return result, filter_counts


def _description_for(column: str) -> str:
    """Return a concise documentation description for one output column."""
    descriptions = {
        "contest_id": "Codeforces contest identifier for the problem.",
        "problemset_name": "Problemset name when supplied by the API.",
        "index": "Problem index within its contest or problemset.",
        "name": "Problem title.",
        "type": "Problem type, such as PROGRAMMING.",
        "points": "Problem point value when supplied.",
        "rating": "Codeforces problem difficulty rating.",
        "tags": "List of Codeforces topic tags.",
        "solved_count": "Number of accepted solvers reported by Codeforces.",
        "id": "Codeforces contest identifier.",
        "contest_name": "Contest name joined from contest.list.",
        "contest_type": "Contest type joined from contest.list.",
        "phase": "Current contest lifecycle phase.",
        "start_time_seconds": "Contest start time as Unix seconds.",
        "duration_seconds": "Contest duration in seconds.",
    }
    return descriptions.get(column, f"Normalized Codeforces API field {column}.")


def build_data_dictionary(
    tables: Mapping[str, pd.DataFrame],
    merged_source_map: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    """Build machine-readable documentation for every Parquet column."""
    base_sources = {
        "problems": ("problemset.problems", "Problem"),
        "problem_statistics": (
            "problemset.problems",
            "ProblemStatistics",
        ),
        "contests": ("contest.list", "Contest"),
    }
    rows: list[dict[str, object]] = []
    for table_name, frame in tables.items():
        for column in frame.columns:
            if table_name in base_sources:
                source_endpoint, source_object = base_sources[table_name]
            else:
                source_endpoint, source_object = merged_source_map.get(
                    column,
                    ("derived", "JoinedProblem"),
                )
            rows.append(
                {
                    "column_name": column,
                    "table_name": table_name,
                    "pandas_dtype": str(frame[column].dtype),
                    "source_endpoint": source_endpoint,
                    "source_object": source_object,
                    "description": _description_for(column),
                    "nullable": bool(frame[column].isna().any()),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "column_name",
            "table_name",
            "pandas_dtype",
            "source_endpoint",
            "source_object",
            "description",
            "nullable",
        ],
    )


def _optional_integer_stat(series: pd.Series, operation: str) -> int | None:
    """Return a nullable integer min or max for a pandas series."""
    non_null = series.dropna()
    if non_null.empty:
        return None
    value = non_null.min() if operation == "min" else non_null.max()
    return int(value)


def build_summary(
    *,
    raw_dir: Path,
    problems: pd.DataFrame,
    statistics: pd.DataFrame,
    contests: pd.DataFrame,
    merged: pd.DataFrame,
    rated_programming: pd.DataFrame,
    filter_counts: Mapping[str, int],
    invalid_counts: Mapping[str, int],
) -> dict[str, object]:
    """Build a JSON-serializable preprocessing summary."""
    missing_counts = {
        column: (
            int(rated_programming[column].isna().sum())
            if column in rated_programming.columns
            else len(rated_programming)
        )
        for column in KEY_SUMMARY_COLUMNS
    }
    rating_series = rated_programming["rating"]
    start_series = (
        rated_programming["start_time_seconds"]
        if "start_time_seconds" in rated_programming.columns
        else pd.Series(dtype="Int64")
    )
    return {
        "input_snapshot_path": raw_dir.resolve().as_posix(),
        "row_counts": {
            "problems": len(problems),
            "problem_statistics": len(statistics),
            "contests": len(contests),
            "problems_merged": len(merged),
            "rated_programming_problems": len(rated_programming),
        },
        "number_of_rated_programming_problems": len(rated_programming),
        "missing_counts": missing_counts,
        "min_rating": _optional_integer_stat(rating_series, "min"),
        "max_rating": _optional_integer_stat(rating_series, "max"),
        "min_start_time_seconds": _optional_integer_stat(start_series, "min"),
        "max_start_time_seconds": _optional_integer_stat(start_series, "max"),
        "filter_counts": dict(filter_counts),
        "invalid_numeric_counts": dict(invalid_counts),
    }


def write_json(path: Path, payload: object) -> None:
    """Write deterministic human-readable UTF-8 JSON."""
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def preprocess_snapshot(
    raw_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
    *,
    log_path: Path = Path("outputs/logs/preprocess.log"),
) -> dict[str, Path]:
    """Preprocess one fetched raw snapshot into all required outputs."""
    logger = configure_logger(log_path)
    try:
        problemset_payload, contest_payload, _manifest = load_raw_snapshot(
            raw_dir
        )
        problems, statistics, contests, invalid_counts = normalize_base_tables(
            problemset_payload,
            contest_payload,
            logger=logger,
        )
        merged, source_map = merge_problem_tables(
            problems,
            statistics,
            contests,
        )
        rated_programming, filter_counts = build_rated_programming_table(merged)

        logger.info(
            "Normalized Codeforces API tables",
            extra={
                "event": "tables_normalized",
                "details": {
                    "problems": len(problems),
                    "problem_statistics": len(statistics),
                    "contests": len(contests),
                    "problems_merged": len(merged),
                    "invalid_numeric_counts": invalid_counts,
                },
            },
        )
        logger.info(
            "Filtered rated programming problems",
            extra={
                "event": "modeling_filter_applied",
                "details": filter_counts,
            },
        )

        interim_dir = interim_dir.resolve()
        processed_dir = processed_dir.resolve()
        interim_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "problems": interim_dir / "problems.parquet",
            "problem_statistics": (
                interim_dir / "problem_statistics.parquet"
            ),
            "contests": interim_dir / "contests.parquet",
            "problems_merged": interim_dir / "problems_merged.parquet",
            "data_dictionary": interim_dir / "data_dictionary.csv",
            "rated_programming_problems": (
                processed_dir / "rated_programming_problems.parquet"
            ),
            "preprocess_summary": (
                processed_dir / "preprocess_summary.json"
            ),
        }
        for table_name, frame in (
            ("problems", problems),
            ("problem_statistics", statistics),
            ("contests", contests),
            ("problems_merged", merged),
            ("rated_programming_problems", rated_programming),
        ):
            frame.to_parquet(paths[table_name], engine="pyarrow", index=False)

        dictionary = build_data_dictionary(
            {
                "problems": problems,
                "problem_statistics": statistics,
                "contests": contests,
                "problems_merged": merged,
                "rated_programming_problems": rated_programming,
            },
            source_map,
        )
        dictionary.to_csv(
            paths["data_dictionary"],
            index=False,
            lineterminator="\n",
        )
        summary = build_summary(
            raw_dir=raw_dir,
            problems=problems,
            statistics=statistics,
            contests=contests,
            merged=merged,
            rated_programming=rated_programming,
            filter_counts=filter_counts,
            invalid_counts=invalid_counts,
        )
        write_json(paths["preprocess_summary"], summary)
        logger.info(
            "Completed Codeforces preprocessing",
            extra={
                "event": "preprocess_completed",
                "details": {
                    "rated_programming_problems": len(rated_programming),
                    "processed_dir": str(processed_dir),
                },
            },
        )
        return paths
    except Exception:
        logger.exception(
            "Codeforces preprocessing failed",
            extra={"event": "preprocess_failed"},
        )
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the preprocessing command-line parser."""
    parser = argparse.ArgumentParser(
        description="Preprocess a raw Codeforces API snapshot."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--interim-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("outputs/logs/preprocess.log"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Codeforces preprocessing CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        paths = preprocess_snapshot(
            args.raw_dir,
            args.interim_dir,
            args.processed_dir,
            log_path=args.log_path,
        )
    except (
        OSError,
        PreprocessError,
        ValueError,
        pd.errors.MergeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        "Wrote rated programming problems: "
        f"{paths['rated_programming_problems']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
