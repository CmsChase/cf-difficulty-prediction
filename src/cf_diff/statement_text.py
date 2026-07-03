"""Extract normalized statement text from cached Codeforces problem pages.

This v6 module reads local cached HTML only. It does not fetch pages, call
Codeforces, train models, or use solved-count behavior.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


DEFAULT_FEATURE_PATH: Final[Path] = Path("data/processed/features/model_table.parquet")
DEFAULT_CACHE_DIR: Final[Path] = Path("data/raw/codeforces/problem_pages")
DEFAULT_OUTPUT_DIR: Final[Path] = Path("data/processed/statement_text")
DEFAULT_LOG_PATH: Final[Path] = Path("outputs/logs/statement_text.log")

TEXT_COLUMNS: Final[tuple[str, ...]] = (
    "title_text",
    "statement_text",
    "input_text",
    "output_text",
    "note_text",
    "examples_text",
    "combined_text",
)
STATUS_COLUMNS: Final[tuple[str, ...]] = (
    "text_extract_status",
    "text_extract_error",
    "html_cache_found",
    "statement_text_available",
)
CONSERVATIVE_NOTES: Final[tuple[str, ...]] = (
    "Statement text extraction is approximate HTML parsing.",
    "Text fields are intended for v6 semantic text modeling.",
    "Cached HTML is local and not committed to the repository.",
    "This module does not train models.",
    "This module does not use solved-count behavior.",
)


class StatementTextError(RuntimeError):
    """Raised when statement text extraction cannot be configured safely."""


class JsonLogFormatter(logging.Formatter):
    """Format statement-text logs as JSON Lines."""

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
class ParsedStatementText:
    """Parsed normalized text fields from one cached problem page."""

    status: str
    error: str
    title_text: str
    statement_text: str
    input_text: str
    output_text: str
    note_text: str
    examples_text: str
    combined_text: str


class CodeforcesStatementTextParser(HTMLParser):
    """Conservative parser for Codeforces problem-statement HTML."""

    def __init__(self) -> None:
        """Initialize parser state."""

        super().__init__(convert_charrefs=True)
        self.in_statement = False
        self.statement_depth = 0
        self.skip_depth = 0
        self.context_stack: list[tuple[str, int]] = []
        self.title_parts: list[str] = []
        self.statement_parts: list[str] = []
        self.input_parts: list[str] = []
        self.output_parts: list[str] = []
        self.note_parts: list[str] = []
        self.examples_parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Track statement boundaries and approximate Codeforces sections."""

        classes = _class_set(attrs)
        if tag in {"script", "style"}:
            self.skip_depth += 1
            return
        if not self.in_statement and tag == "div" and "problem-statement" in classes:
            self.in_statement = True
            self.statement_depth = 1
            return
        if self.in_statement:
            self.statement_depth += 1
            context = _context_from_classes(classes, self.context_stack)
            if context is not None:
                self.context_stack.append((context, self.statement_depth))

    def handle_endtag(self, tag: str) -> None:
        """Close skip, context, and statement scopes."""

        if tag in {"script", "style"} and self.skip_depth > 0:
            self.skip_depth -= 1
            return
        if not self.in_statement:
            return
        while self.context_stack and self.context_stack[-1][1] == self.statement_depth:
            self.context_stack.pop()
        self.statement_depth -= 1
        if self.statement_depth <= 0:
            self.in_statement = False
            self.statement_depth = 0
            self.context_stack.clear()

    def handle_data(self, data: str) -> None:
        """Collect visible text into approximate sections."""

        if not self.in_statement or self.skip_depth > 0 or not data.strip():
            return
        context = self.context_stack[-1][0] if self.context_stack else "statement"
        if context == "title":
            self.title_parts.append(data)
        elif context == "input":
            self.input_parts.append(data)
        elif context == "output":
            self.output_parts.append(data)
        elif context == "note":
            self.note_parts.append(data)
        elif context == "examples":
            self.examples_parts.append(data)
        elif context in {"header", "time_limit", "memory_limit"}:
            return
        else:
            self.statement_parts.append(data)


def configure_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """Create the dedicated structured statement-text logger."""

    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.statement_text")
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
    """Flush and close all handlers attached to the statement-text logger."""

    for handler in tuple(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _class_set(attrs: Iterable[tuple[str, str | None]]) -> set[str]:
    """Return lower-case CSS classes from parser attrs."""

    result: set[str] = set()
    for key, value in attrs:
        if key.lower() == "class" and value:
            result.update(part.strip().lower() for part in value.split())
    return result


def _active_context(context_stack: Sequence[tuple[str, int]]) -> str | None:
    """Return the active parser context, if any."""

    return context_stack[-1][0] if context_stack else None


def _context_from_classes(
    classes: set[str],
    context_stack: Sequence[tuple[str, int]],
) -> str | None:
    """Map Codeforces CSS classes to text extraction contexts."""

    active = _active_context(context_stack)
    if "sample-tests" in classes or "sample-test" in classes:
        return "examples"
    if "input-specification" in classes:
        return "input"
    if "output-specification" in classes:
        return "output"
    if "note" in classes:
        return "note"
    if "time-limit" in classes:
        return "time_limit"
    if "memory-limit" in classes:
        return "memory_limit"
    if "header" in classes:
        return "header"
    if "title" in classes and active == "header":
        return "title"
    if active == "examples" and "title" in classes:
        return "examples"
    return None


def normalize_identifier(value: object) -> str:
    """Normalize contest ids and problem indices to stable strings."""

    if value is None or pd.isna(value):
        raise StatementTextError("Problem identifier contains a missing value.")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def safe_filename_part(value: object) -> str:
    """Create a filesystem-safe cache filename component."""

    normalized = normalize_identifier(value)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized)
    return safe.strip("._") or "missing"


def cache_path_for_problem(cache_dir: Path, contest_id: object, index: object) -> Path:
    """Return the expected deterministic cache path for one problem page."""

    contest = safe_filename_part(contest_id)
    problem_index = safe_filename_part(index)
    return cache_dir / f"{contest}_{problem_index}.html"


def find_cached_html_path(cache_dir: Path, contest_id: object, index: object) -> Path | None:
    """Find a cached HTML path, tolerating index casing and safe-name variants."""

    candidates = [
        cache_path_for_problem(cache_dir, contest_id, index),
        cache_path_for_problem(cache_dir, contest_id, str(index).upper()),
        cache_path_for_problem(cache_dir, contest_id, str(index).lower()),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not cache_dir.exists():
        return None
    contest = safe_filename_part(contest_id).lower()
    target_stem = f"{contest}_{safe_filename_part(index).lower()}"
    for path in sorted(cache_dir.glob(f"{safe_filename_part(contest_id)}_*.html")):
        if path.stem.lower() == target_stem:
            return path
    return None


def remove_script_style(html_text: str) -> str:
    """Remove script and style blocks before visible text extraction."""

    return re.sub(
        r"(?is)<(script|style)\b[^>]*>.*?</\1>",
        " ",
        html_text,
    )


def normalize_text(value: str) -> str:
    """Decode entities, normalize math symbols lightly, and collapse whitespace."""

    decoded = html.unescape(value)
    replacements = {
        "\xa0": " ",
        "≤": "<=",
        "≥": ">=",
        "−": "-",
        "×": "x",
        "÷": "/",
        "∑": "sum",
    }
    for source, target in replacements.items():
        decoded = decoded.replace(source, target)
    decoded = re.sub(r"\s+", " ", decoded)
    return decoded.strip()


def strip_html_tags(html_text: str) -> str:
    """Return normalized visible text after removing tags and scripts/styles."""

    without_scripts = remove_script_style(html_text)
    with_spaces = re.sub(r"(?is)<\s*(br|p|div|pre|li|tr|h[1-6])\b[^>]*>", " ", without_scripts)
    without_tags = re.sub(r"(?is)<[^>]+>", " ", with_spaces)
    return normalize_text(without_tags)


def _normalize_parts(parts: Sequence[str]) -> str:
    """Normalize concatenated parser text parts."""

    return normalize_text(" ".join(parts))


def _drop_section_title(text: str, title: str) -> str:
    """Remove duplicated Codeforces section labels such as ``Input``."""

    pattern = rf"(?i)^{re.escape(title)}\b[:.]?\s*"
    return re.sub(pattern, "", text).strip()


def parse_statement_html(html_text: str) -> ParsedStatementText:
    """Parse one cached Codeforces HTML document into normalized text fields."""

    parser = CodeforcesStatementTextParser()
    try:
        parser.feed(remove_script_style(html_text))
        parser.close()
    except Exception as error:  # HTMLParser should be forgiving; keep rows on failure.
        return ParsedStatementText(
            status="failed",
            error=f"HTML parsing failed: {error}",
            title_text="",
            statement_text="",
            input_text="",
            output_text="",
            note_text="",
            examples_text="",
            combined_text="",
        )
    title_text = _normalize_parts(parser.title_parts)
    statement_text = _normalize_parts(parser.statement_parts)
    input_text = _drop_section_title(_normalize_parts(parser.input_parts), "Input")
    output_text = _drop_section_title(_normalize_parts(parser.output_parts), "Output")
    note_text = _drop_section_title(_normalize_parts(parser.note_parts), "Note")
    examples_text = _normalize_parts(parser.examples_parts)
    examples_text = re.sub(r"(?i)\b(example|sample|input|output)\b[:.]?", " ", examples_text)
    examples_text = normalize_text(examples_text)
    combined_text = normalize_text(
        " ".join(
            part
            for part in (
                title_text,
                statement_text,
                input_text,
                output_text,
                note_text,
                examples_text,
            )
            if part
        )
    )
    if not combined_text:
        return ParsedStatementText(
            status="failed",
            error="problem-statement block was not found or had no text",
            title_text="",
            statement_text="",
            input_text="",
            output_text="",
            note_text="",
            examples_text="",
            combined_text="",
        )
    return ParsedStatementText(
        status="parsed",
        error="",
        title_text=title_text,
        statement_text=statement_text,
        input_text=input_text,
        output_text=output_text,
        note_text=note_text,
        examples_text=examples_text,
        combined_text=combined_text,
    )


def detect_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Find a column using common exact names."""

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def load_problem_rows(feature_path: Path) -> pd.DataFrame:
    """Load the model table and standardize identifier columns."""

    if not feature_path.exists():
        raise StatementTextError(f"Feature table does not exist: {feature_path}")
    frame = pd.read_parquet(feature_path, engine="pyarrow")
    contest_column = detect_column(frame, ("contest_id", "contestId", "contestid"))
    index_column = detect_column(frame, ("index", "problem_index", "problemIndex"))
    if contest_column is None or index_column is None:
        raise StatementTextError("Feature table must include contest_id and index columns.")
    output = pd.DataFrame(
        {
            "contest_id": frame[contest_column],
            "index": frame[index_column],
            "name": frame[detect_column(frame, ("name", "problem_name", "problemName"))]
            if detect_column(frame, ("name", "problem_name", "problemName")) is not None
            else "",
            "rating": frame[detect_column(frame, ("rating", "problem_rating"))]
            if detect_column(frame, ("rating", "problem_rating")) is not None
            else np.nan,
        }
    )
    output["_contest_key"] = output["contest_id"].map(normalize_identifier)
    output["_index_key"] = output["index"].map(normalize_identifier)
    return output.sort_values(["_contest_key", "_index_key"], kind="mergesort").reset_index(
        drop=True
    )


def empty_text_record(
    *,
    row: pd.Series,
    status: str,
    error: str,
    html_cache_found: bool,
) -> dict[str, object]:
    """Build one output record for missing or failed extraction."""

    return {
        "contest_id": row["contest_id"],
        "index": row["index"],
        "name": row.get("name", ""),
        "rating": row.get("rating", np.nan),
        "text_extract_status": status,
        "text_extract_error": error,
        "html_cache_found": bool(html_cache_found),
        "statement_text_available": False,
        **{column: "" for column in TEXT_COLUMNS},
    }


def extract_text_for_row(row: pd.Series, cache_dir: Path) -> dict[str, object]:
    """Extract normalized text for one model-table row from cached HTML."""

    path = find_cached_html_path(cache_dir, row["contest_id"], row["index"])
    if path is None:
        return empty_text_record(
            row=row,
            status="missing_cache",
            error="cached HTML file not found",
            html_cache_found=False,
        )
    try:
        html_text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return empty_text_record(
            row=row,
            status="failed",
            error=f"failed to read cached HTML: {error}",
            html_cache_found=True,
        )
    parsed = parse_statement_html(html_text)
    record = {
        "contest_id": row["contest_id"],
        "index": row["index"],
        "name": row.get("name", ""),
        "rating": row.get("rating", np.nan),
        "text_extract_status": parsed.status,
        "text_extract_error": parsed.error,
        "html_cache_found": True,
        "statement_text_available": bool(parsed.statement_text),
        "title_text": parsed.title_text,
        "statement_text": parsed.statement_text,
        "input_text": parsed.input_text,
        "output_text": parsed.output_text,
        "note_text": parsed.note_text,
        "examples_text": parsed.examples_text,
        "combined_text": parsed.combined_text,
    }
    return record


def build_summary(records: pd.DataFrame, input_row_count: int) -> dict[str, object]:
    """Build the machine-readable extraction summary."""

    lengths = records["combined_text"].fillna("").map(len) if not records.empty else pd.Series(dtype=int)
    cached_found = int(records["html_cache_found"].eq(True).sum()) if not records.empty else 0
    statement_available = (
        int(records["statement_text_available"].eq(True).sum()) if not records.empty else 0
    )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_model_table_rows": int(input_row_count),
        "cached_html_found_count": cached_found,
        "cached_html_missing_count": int(input_row_count - cached_found),
        "extracted_success_count": int(records["text_extract_status"].eq("parsed").sum())
        if not records.empty
        else 0,
        "extracted_failure_count": int(records["text_extract_status"].ne("parsed").sum())
        if not records.empty
        else 0,
        "statement_text_available_count": statement_available,
        "statement_text_available_rate": round(
            statement_available / input_row_count if input_row_count else 0.0,
            6,
        ),
        "average_combined_text_length": round(float(lengths.mean()), 6)
        if len(lengths)
        else 0.0,
        "median_combined_text_length": round(float(lengths.median()), 6)
        if len(lengths)
        else 0.0,
        "max_combined_text_length": int(lengths.max()) if len(lengths) else 0,
        "conservative_notes": list(CONSERVATIVE_NOTES),
    }


def write_json(path: Path, payload: object) -> None:
    """Write pretty UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_statement_text_extraction(
    *,
    feature_path: Path,
    cache_dir: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Path]:
    """Extract statement text from cached HTML and write all artifacts."""

    logger = configure_logger(log_path)
    try:
        problem_rows = load_problem_rows(feature_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        records = pd.DataFrame(
            [extract_text_for_row(row, cache_dir) for _, row in problem_rows.iterrows()]
        )
        output_columns = [
            "contest_id",
            "index",
            "name",
            "rating",
            *STATUS_COLUMNS,
            *TEXT_COLUMNS,
        ]
        if records.empty:
            records = pd.DataFrame(columns=output_columns)
        else:
            records = records.loc[:, output_columns]
        paths = {
            "statement_text_parquet": output_dir / "statement_text.parquet",
            "statement_text_csv": output_dir / "statement_text.csv",
            "statement_text_summary": output_dir / "statement_text_summary.json",
        }
        records.to_parquet(paths["statement_text_parquet"], engine="pyarrow", index=False)
        records.to_csv(paths["statement_text_csv"], index=False)
        summary = build_summary(records, len(problem_rows))
        write_json(paths["statement_text_summary"], summary)
        logger.info(
            "Completed cached statement text extraction",
            extra={
                "event": "statement_text_completed",
                "details": {
                    "input_rows": len(problem_rows),
                    "cached_html_found_count": summary["cached_html_found_count"],
                    "extracted_success_count": summary["extracted_success_count"],
                    "output_dir": output_dir.as_posix(),
                },
            },
        )
        return paths
    except Exception:
        logger.exception(
            "Statement text extraction failed",
            extra={"event": "statement_text_failed"},
        )
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Extract normalized statement text from cached Codeforces HTML."
    )
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the statement-text extraction CLI."""

    args = _build_argument_parser().parse_args(argv)
    try:
        paths = run_statement_text_extraction(
            feature_path=args.feature_path,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            log_path=args.log_path,
        )
    except (StatementTextError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote statement text: {paths['statement_text_parquet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
