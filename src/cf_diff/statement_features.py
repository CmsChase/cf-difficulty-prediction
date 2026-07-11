"""Extract lightweight Codeforces problem-statement structure features."""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Final, Iterable, Sequence

import numpy as np
import pandas as pd

from cf_diff.features import write_json

DEFAULT_FEATURE_PATH: Final[Path] = Path(
    "data/processed/features/model_table.parquet"
)
DEFAULT_CACHE_DIR: Final[Path] = Path("data/raw/codeforces/problem_pages")
DEFAULT_OUTPUT_DIR: Final[Path] = Path("data/processed/statement_features")
DEFAULT_LOG_PATH: Final[Path] = Path("outputs/logs/statement_features.log")
BASE_PROBLEM_URL: Final[str] = "https://codeforces.com/problemset/problem"
USER_AGENT: Final[str] = (
    "cf-difficulty-prediction research script "
    "(polite statement feature extraction; contact: local-research)"
)
DEFAULT_SLEEP_SECONDS: Final[float] = 2.5
DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
DEFAULT_RETRIES: Final[int] = 3

IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = ("contest_id", "index", "name", "url")
STATUS_COLUMNS: Final[tuple[str, ...]] = (
    "statement_fetch_status",
    "statement_parse_status",
    "statement_error",
)
STATEMENT_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "statement_available",
    "statement_char_len",
    "statement_word_count",
    "statement_line_count",
    "statement_paragraph_count",
    "input_section_char_len",
    "output_section_char_len",
    "note_section_char_len",
    "has_input_section",
    "has_output_section",
    "has_note_section",
    "sample_count",
    "sample_input_output_block_count",
    "number_count",
    "integer_count",
    "float_count",
    "inequality_symbol_count",
    "math_symbol_count",
    "constraint_keyword_count",
    "big_o_like_count",
    "uppercase_token_count",
    "single_letter_variable_count",
    "latex_like_token_count",
    "code_like_token_count",
    "kw_graph",
    "kw_tree",
    "kw_array",
    "kw_string",
    "kw_dp",
    "kw_geometry",
    "kw_greedy",
    "kw_probability",
    "kw_interactive",
    "kw_permutation",
    "kw_binary",
    "kw_shortest_path",
    "kw_query",
    "time_limit_ms",
    "memory_limit_mb",
    "has_time_limit",
    "has_memory_limit",
)
KEYWORD_PATTERNS: Final[dict[str, str]] = {
    "kw_graph": r"\bgraph(s)?\b|\bvertices\b|\bedges\b",
    "kw_tree": r"\btree(s)?\b|\brooted\b",
    "kw_array": r"\barray(s)?\b|\bsequence(s)?\b|\blist(s)?\b",
    "kw_string": r"\bstring(s)?\b|\bsubstring(s)?\b|\bcharacter(s)?\b",
    "kw_dp": r"\bdp\b|\bdynamic programming\b",
    "kw_geometry": r"\bgeometry\b|\bpoint(s)?\b|\bpolygon(s)?\b|\bcircle(s)?\b",
    "kw_greedy": r"\bgreedy\b",
    "kw_probability": r"\bprobability\b|\bexpected value\b|\brandom\b",
    "kw_interactive": r"\binteractive\b|\bflush\b|\bquery the judge\b",
    "kw_permutation": r"\bpermutation(s)?\b",
    "kw_binary": r"\bbinary\b|\bbit(s)?\b|\bbitwise\b",
    "kw_shortest_path": r"\bshortest path(s)?\b|\bdijkstra\b|\bfloyd\b",
    "kw_query": r"\bquery\b|\bqueries\b",
}


class StatementFeatureError(RuntimeError):
    """Raised when statement feature extraction cannot proceed safely."""


class JsonLogFormatter(logging.Formatter):
    """Format statement-feature logs as JSON Lines."""

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
class FetchResult:
    """Represent one cached or fetched HTML page."""

    status: str
    cache_path: Path
    html_text: str | None
    http_status: int | None = None
    error: str = ""
    content_type: str | None = None
    final_url: str | None = None


@dataclass(frozen=True)
class ParsedStatement:
    """Represent parsed statement text and approximate section text."""

    status: str
    statement_text: str
    input_text: str
    output_text: str
    note_text: str
    time_limit_text: str
    memory_limit_text: str
    paragraph_count: int
    sample_count: int
    sample_input_output_block_count: int
    error: str = ""


class CodeforcesStatementParser(HTMLParser):
    """Conservative parser for public Codeforces problem-statement HTML."""

    def __init__(self) -> None:
        """Initialize parser state."""
        super().__init__(convert_charrefs=True)
        self.in_statement = False
        self.statement_depth = 0
        self.skip_depth = 0
        self.statement_parts: list[str] = []
        self.section_parts: dict[str, list[str]] = {
            "input": [],
            "output": [],
            "note": [],
            "time_limit": [],
            "memory_limit": [],
        }
        self.section_stack: list[tuple[str, int]] = []
        self.paragraph_count = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Track relevant statement and section tags."""
        classes = _class_set(attrs)
        if tag in {"script", "style"}:
            self.skip_depth += 1
            return
        if not self.in_statement and tag == "div" and "problem-statement" in classes:
            self.in_statement = True
            self.statement_depth = 1
        elif self.in_statement:
            self.statement_depth += 1

        if not self.in_statement:
            return
        if tag == "p":
            self.paragraph_count += 1
        section = _section_name_from_classes(classes)
        if section is not None:
            self.section_stack.append((section, self.statement_depth))

    def handle_endtag(self, tag: str) -> None:
        """Close statement, section, and skip scopes."""
        if tag in {"script", "style"} and self.skip_depth > 0:
            self.skip_depth -= 1
            return
        if not self.in_statement:
            return
        while self.section_stack and self.section_stack[-1][1] == self.statement_depth:
            self.section_stack.pop()
        self.statement_depth -= 1
        if self.statement_depth <= 0:
            self.in_statement = False
            self.statement_depth = 0
            self.section_stack.clear()

    def handle_data(self, data: str) -> None:
        """Collect text inside the problem statement and active section."""
        if not self.in_statement or self.skip_depth > 0:
            return
        if not data.strip():
            return
        self.statement_parts.append(data)
        if self.section_stack:
            section = self.section_stack[-1][0]
            self.section_parts[section].append(data)


def configure_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """Create the dedicated structured statement-feature logger."""
    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.statement_features")
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


def _class_set(attrs: Iterable[tuple[str, str | None]]) -> set[str]:
    """Return lower-case CSS classes from parser attrs."""
    result: set[str] = set()
    for key, value in attrs:
        if key.lower() == "class" and value:
            result.update(part.strip().lower() for part in value.split())
    return result


def _section_name_from_classes(classes: set[str]) -> str | None:
    """Map Codeforces section classes to internal names."""
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
    return None


def _normalize_identifier(value: object) -> str:
    """Normalize contest ids and indices into stable strings."""
    if value is None or pd.isna(value):
        raise StatementFeatureError("Problem identifier contains a missing value.")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _safe_filename_part(value: str) -> str:
    """Create a filesystem-safe cache filename component."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "missing"


def detect_contest_id_column(frame: pd.DataFrame) -> str:
    """Find a contest-id column using common project/API spellings."""
    for column in ("contest_id", "contestId", "contestid"):
        if column in frame.columns:
            return column
    raise StatementFeatureError(
        "Feature table lacks a contest id column such as contest_id or contestId."
    )


def build_problem_url(contest_id: object, index: object) -> str:
    """Build the public Codeforces problem-page URL."""
    contest = _normalize_identifier(contest_id)
    problem_index = urllib.parse.quote(_normalize_identifier(index), safe="")
    return f"{BASE_PROBLEM_URL}/{contest}/{problem_index}"


def cache_path_for_problem(cache_dir: Path, contest_id: object, index: object) -> Path:
    """Return the deterministic cache path for one problem page."""
    contest = _safe_filename_part(_normalize_identifier(contest_id))
    problem_index = _safe_filename_part(_normalize_identifier(index))
    return cache_dir / f"{contest}_{problem_index}.html"


def _decode_bytes(raw: bytes, content_type: str | None = None) -> str:
    """Decode HTTP bytes, preferring charset from content type when present."""
    encoding = "utf-8"
    if content_type:
        match = re.search(r"charset=([\w.-]+)", content_type, flags=re.I)
        if match:
            encoding = match.group(1)
    return raw.decode(encoding, errors="replace")


def fetch_or_load_problem_page(
    *,
    url: str,
    cache_path: Path,
    sleep_seconds: float,
    timeout: int,
    retries: int = DEFAULT_RETRIES,
    last_fetch_time: float | None = None,
) -> tuple[FetchResult, float | None]:
    """Load cached HTML or fetch it politely with retries."""
    if cache_path.exists():
        return (
            FetchResult(
                status="cached",
                cache_path=cache_path,
                html_text=cache_path.read_text(encoding="utf-8", errors="replace"),
            ),
            last_fetch_time,
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if last_fetch_time is not None:
        elapsed = time.monotonic() - last_fetch_time
        if elapsed < sleep_seconds:
            time.sleep(sleep_seconds - elapsed)

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = ""
    http_status: int | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                http_status = getattr(response, "status", None)
                text = _decode_bytes(raw, response.headers.get("Content-Type"))
            cache_path.write_text(text, encoding="utf-8", newline="\n")
            return (
                FetchResult(
                    status="fetched",
                    cache_path=cache_path,
                    html_text=text,
                    http_status=http_status,
                ),
                time.monotonic(),
            )
        except urllib.error.HTTPError as error:
            http_status = error.code
            last_error = f"HTTP {error.code}: {error.reason}"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = str(error)
        if attempt < retries:
            time.sleep(min(sleep_seconds * attempt, 10.0))
    return (
        FetchResult(
            status="failed",
            cache_path=cache_path,
            html_text=None,
            http_status=http_status,
            error=last_error or "unknown fetch error",
        ),
        time.monotonic(),
    )


def strip_html_tags(html_text: str) -> str:
    """Strip scripts, styles, and tags, then decode HTML entities."""
    without_scripts = re.sub(
        r"(?is)<(script|style)\b[^>]*>.*?</\1>",
        " ",
        html_text,
    )
    with_breaks = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</tr>", "\n", without_scripts)
    without_tags = re.sub(r"(?s)<[^>]+>", " ", with_breaks)
    return normalize_text(html.unescape(without_tags))


def normalize_text(value: str) -> str:
    """Normalize whitespace while preserving rough line boundaries."""
    decoded = html.unescape(value)
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in decoded.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _statement_html_fragment(html_text: str) -> str:
    """Return a rough problem-statement fragment for regex-based counters."""
    match = re.search(
        r'(?is)<div[^>]*class=["\'][^"\']*\bproblem-statement\b[^"\']*["\'][^>]*>',
        html_text,
    )
    return html_text[match.start() :] if match else html_text


def parse_problem_statement(html_text: str) -> ParsedStatement:
    """Parse public Codeforces problem statement HTML approximately."""
    parser = CodeforcesStatementParser()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception as error:  # HTMLParser should be forgiving, but be safe.
        return ParsedStatement(
            status="failed",
            statement_text="",
            input_text="",
            output_text="",
            note_text="",
            time_limit_text="",
            memory_limit_text="",
            paragraph_count=0,
            sample_count=0,
            sample_input_output_block_count=0,
            error=str(error),
        )

    statement_text = normalize_text("\n".join(parser.statement_parts))
    input_text = normalize_text("\n".join(parser.section_parts["input"]))
    output_text = normalize_text("\n".join(parser.section_parts["output"]))
    note_text = normalize_text("\n".join(parser.section_parts["note"]))
    time_limit_text = normalize_text("\n".join(parser.section_parts["time_limit"]))
    memory_limit_text = normalize_text("\n".join(parser.section_parts["memory_limit"]))
    fragment = _statement_html_fragment(html_text)
    sample_count = count_sample_tests(fragment)
    sample_blocks = count_sample_input_output_blocks(fragment)

    if not statement_text:
        return ParsedStatement(
            status="missing_statement",
            statement_text="",
            input_text=input_text,
            output_text=output_text,
            note_text=note_text,
            time_limit_text=time_limit_text,
            memory_limit_text=memory_limit_text,
            paragraph_count=0,
            sample_count=sample_count,
            sample_input_output_block_count=sample_blocks,
            error="problem-statement block was not found or had no text",
        )
    paragraph_count = parser.paragraph_count
    if paragraph_count == 0:
        paragraph_count = max(1, len([line for line in statement_text.splitlines() if line]))
    return ParsedStatement(
        status="parsed",
        statement_text=statement_text,
        input_text=input_text,
        output_text=output_text,
        note_text=note_text,
        time_limit_text=time_limit_text,
        memory_limit_text=memory_limit_text,
        paragraph_count=paragraph_count,
        sample_count=sample_count,
        sample_input_output_block_count=sample_blocks,
    )


def parse_time_limit_ms(text: str) -> int | None:
    """Parse a Codeforces time-limit string into milliseconds."""
    normalized = normalize_text(text).lower()
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(milliseconds?|msecs?|ms|seconds?|secs?|s)\b",
        normalized,
    )
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit.startswith(("millisecond", "msec", "ms")):
        return int(round(value))
    return int(round(value * 1000.0))


def parse_memory_limit_mb(text: str) -> int | None:
    """Parse a Codeforces memory-limit string into megabytes."""
    normalized = normalize_text(text).lower()
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(megabytes?|mb|mib|gigabytes?|gb|gib|kilobytes?|kb|kib)\b",
        normalized,
    )
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit.startswith(("gigabyte", "gb", "gib")):
        return int(round(value * 1024.0))
    if unit.startswith(("kilobyte", "kb", "kib")):
        return int(math.ceil(value / 1024.0))
    return int(round(value))


def count_sample_tests(html_text: str) -> int:
    """Count sample-test blocks in a Codeforces-like statement fragment."""
    count = len(
        re.findall(
            r'(?is)<div[^>]*class=["\'][^"\']*\bsample-test\b[^"\']*["\']',
            html_text,
        )
    )
    if count:
        return count
    input_blocks = len(
        re.findall(r'(?is)<div[^>]*class=["\'][^"\']*\binput\b[^"\']*["\']', html_text)
    )
    output_blocks = len(
        re.findall(r'(?is)<div[^>]*class=["\'][^"\']*\boutput\b[^"\']*["\']', html_text)
    )
    return min(input_blocks, output_blocks)


def count_sample_input_output_blocks(html_text: str) -> int:
    """Count sample input/output blocks in a Codeforces-like fragment."""
    count = 0
    for match in re.finditer(
        r'(?is)<div[^>]*class=["\']([^"\']+)["\']',
        html_text,
    ):
        classes = {part.strip().lower() for part in match.group(1).split()}
        if classes & {"input", "output"}:
            count += 1
    return count


def _count_regex(pattern: str, text: str, flags: int = 0) -> int:
    """Count regex matches."""
    return len(re.findall(pattern, text, flags=flags))


def extract_keyword_features(text: str) -> dict[str, int]:
    """Extract binary keyword indicator features."""
    return {
        column: int(bool(re.search(pattern, text, flags=re.I)))
        for column, pattern in KEYWORD_PATTERNS.items()
    }


def _empty_feature_values() -> dict[str, object]:
    """Return zero/null values for failed or missing statement parses."""
    values: dict[str, object] = {column: 0 for column in STATEMENT_FEATURE_COLUMNS}
    values["time_limit_ms"] = None
    values["memory_limit_mb"] = None
    return values


def build_statement_feature_values(parsed: ParsedStatement) -> dict[str, object]:
    """Build text-light feature values from parsed statement text."""
    if parsed.status != "parsed":
        return _empty_feature_values()

    text = parsed.statement_text
    lower_text = text.lower()
    words = re.findall(r"\b[\w'-]+\b", text, flags=re.U)
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    floats = re.findall(r"[-+]?\d+\.\d+", text)
    integers = [
        number
        for number in numbers
        if "." not in number
    ]
    time_limit_ms = parse_time_limit_ms(parsed.time_limit_text)
    memory_limit_mb = parse_memory_limit_mb(parsed.memory_limit_text)
    features: dict[str, object] = {
        "statement_available": int(bool(text.strip())),
        "statement_char_len": len(text),
        "statement_word_count": len(words),
        "statement_line_count": len(text.splitlines()) if text else 0,
        "statement_paragraph_count": parsed.paragraph_count,
        "input_section_char_len": len(parsed.input_text),
        "output_section_char_len": len(parsed.output_text),
        "note_section_char_len": len(parsed.note_text),
        "has_input_section": int(bool(parsed.input_text)),
        "has_output_section": int(bool(parsed.output_text)),
        "has_note_section": int(bool(parsed.note_text)),
        "sample_count": parsed.sample_count,
        "sample_input_output_block_count": parsed.sample_input_output_block_count,
        "number_count": len(numbers),
        "integer_count": len(integers),
        "float_count": len(floats),
        "inequality_symbol_count": _count_regex(r"<=|>=|<|>|≤|≥", text),
        "math_symbol_count": _count_regex(r"[+\-*/=%^]|<=|>=|≤|≥", text),
        "constraint_keyword_count": _count_regex(
            r"\b(constraints?|limits?|integer(s)?|positive|negative|at most|at least|sum)\b",
            lower_text,
            flags=re.I,
        ),
        "big_o_like_count": _count_regex(r"\bO\s*\([^)]*\)", text),
        "uppercase_token_count": _count_regex(r"\b[A-Z]{2,}\b", text),
        "single_letter_variable_count": _count_regex(r"\b[A-Za-z]\b", text),
        "latex_like_token_count": _count_regex(
            r"\\[A-Za-z]+|\\\(|\\\)|\\\[|\\\]|\$[^$]+\$",
            text,
        ),
        "code_like_token_count": _count_regex(
            r"\b[A-Za-z_]\w*\s*\(|\b\w+\[[^\]]*\]|==|!=|<=|>=|::|->",
            text,
        ),
        **extract_keyword_features(text),
        "time_limit_ms": time_limit_ms,
        "memory_limit_mb": memory_limit_mb,
        "has_time_limit": int(time_limit_ms is not None),
        "has_memory_limit": int(memory_limit_mb is not None),
    }
    return features


def build_feature_row(
    *,
    contest_id: object,
    index: object,
    name: object,
    url: str,
    fetch_result: FetchResult,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build one output feature row and one fetch-report row."""
    base_error = fetch_result.error
    if fetch_result.html_text is None:
        parsed = ParsedStatement(
            status="not_parsed",
            statement_text="",
            input_text="",
            output_text="",
            note_text="",
            time_limit_text="",
            memory_limit_text="",
            paragraph_count=0,
            sample_count=0,
            sample_input_output_block_count=0,
            error=base_error,
        )
    else:
        parsed = parse_problem_statement(fetch_result.html_text)
    statement_error = base_error or parsed.error
    row = {
        "contest_id": _normalize_identifier(contest_id),
        "index": _normalize_identifier(index),
        "name": "" if name is None or pd.isna(name) else str(name),
        "url": url,
        "statement_fetch_status": fetch_result.status,
        "statement_parse_status": parsed.status,
        "statement_error": statement_error,
        **build_statement_feature_values(parsed),
    }
    report_row = {
        "contest_id": row["contest_id"],
        "index": row["index"],
        "name": row["name"],
        "url": url,
        "cache_path": fetch_result.cache_path.as_posix(),
        "fetch_status": fetch_result.status,
        "http_status": fetch_result.http_status,
        "parse_status": parsed.status,
        "error": statement_error,
    }
    return row, report_row


def _select_problem_rows(frame: pd.DataFrame, max_pages: int | None) -> pd.DataFrame:
    """Select deterministic problem rows to process."""
    contest_column = detect_contest_id_column(frame)
    if "index" not in frame.columns:
        raise StatementFeatureError("Feature table lacks required index column.")
    columns = [contest_column, "index"]
    if "name" in frame.columns:
        columns.append("name")
    selected = frame.loc[:, columns].drop_duplicates([contest_column, "index"]).copy()
    selected = selected.rename(columns={contest_column: "contest_id"})
    if "name" not in selected.columns:
        selected["name"] = ""
    selected["_contest_sort"] = pd.to_numeric(selected["contest_id"], errors="coerce")
    selected = selected.sort_values(
        ["_contest_sort", "contest_id", "index"],
        kind="mergesort",
        na_position="last",
    ).drop(columns=["_contest_sort"])
    if max_pages is not None:
        selected = selected.head(max_pages)
    return selected.reset_index(drop=True)


def _summary_payload(
    *,
    input_row_count: int,
    attempted_page_count: int,
    feature_rows: pd.DataFrame,
    fetch_report: pd.DataFrame,
) -> dict[str, object]:
    """Build summary JSON payload."""
    statement_lengths = pd.to_numeric(
        feature_rows.get("statement_char_len", pd.Series(dtype=float)),
        errors="coerce",
    )
    available = pd.to_numeric(
        feature_rows.get("statement_available", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0)
    fetched_count = int(fetch_report["fetch_status"].eq("fetched").sum())
    cached_count = int(fetch_report["fetch_status"].eq("cached").sum())
    failed_count = int(fetch_report["fetch_status"].eq("failed").sum())
    parsed_success = int(feature_rows["statement_parse_status"].eq("parsed").sum())
    parsed_failure = int(len(feature_rows) - parsed_success)
    available_count = int(available.sum())
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_row_count": int(input_row_count),
        "attempted_page_count": int(attempted_page_count),
        "cached_page_count": cached_count,
        "fetched_page_count": fetched_count,
        "failed_fetch_count": failed_count,
        "parsed_success_count": parsed_success,
        "parsed_failure_count": parsed_failure,
        "statement_available_count": available_count,
        "statement_available_rate": (
            round(available_count / len(feature_rows), 6)
            if len(feature_rows)
            else 0.0
        ),
        "average_statement_length": _finite_float(statement_lengths.mean()),
        "median_statement_length": _finite_float(statement_lengths.median()),
        "max_statement_length": (
            int(statement_lengths.max())
            if len(statement_lengths.dropna())
            else 0
        ),
        "feature_count": len(STATEMENT_FEATURE_COLUMNS),
        "conservative_notes": [
            "Problem-page HTML parsing is approximate.",
            "Statement text-light features are intended for cold-start analysis.",
            "These features do not use post-publication solved statistics.",
            "Raw cached HTML is not committed to the repository.",
        ],
    }


def _finite_float(value: object) -> float | None:
    """Convert finite numeric values to JSON-safe floats."""
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, 6)


def run_statement_feature_extraction(
    *,
    feature_path: Path,
    cache_dir: Path,
    output_dir: Path,
    sleep_seconds: float,
    timeout: int,
    max_pages: int | None,
    log_path: Path,
) -> dict[str, Path]:
    """Extract statement text-light features and write artifacts."""
    logger = configure_logger(log_path)
    try:
        feature_table = pd.read_parquet(feature_path, engine="pyarrow")
        problem_rows = _select_problem_rows(feature_table, max_pages)
        output_dir = output_dir.resolve()
        cache_dir = cache_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        feature_records: list[dict[str, object]] = []
        report_records: list[dict[str, object]] = []
        last_fetch_time: float | None = None
        for problem in problem_rows.to_dict(orient="records"):
            contest_id = problem["contest_id"]
            index = problem["index"]
            name = problem.get("name", "")
            url = build_problem_url(contest_id, index)
            cache_path = cache_path_for_problem(cache_dir, contest_id, index)
            fetch_result, last_fetch_time = fetch_or_load_problem_page(
                url=url,
                cache_path=cache_path,
                sleep_seconds=sleep_seconds,
                timeout=timeout,
                last_fetch_time=last_fetch_time,
            )
            feature_row, report_row = build_feature_row(
                contest_id=contest_id,
                index=index,
                name=name,
                url=url,
                fetch_result=fetch_result,
            )
            feature_records.append(feature_row)
            report_records.append(report_row)

        features = pd.DataFrame(feature_records)
        fetch_report = pd.DataFrame(report_records)
        if features.empty:
            features = pd.DataFrame(
                columns=[*IDENTIFIER_COLUMNS, *STATUS_COLUMNS, *STATEMENT_FEATURE_COLUMNS]
            )
        if fetch_report.empty:
            fetch_report = pd.DataFrame(
                columns=[
                    "contest_id",
                    "index",
                    "name",
                    "url",
                    "cache_path",
                    "fetch_status",
                    "http_status",
                    "parse_status",
                    "error",
                ]
            )
        features = features.loc[
            :,
            [*IDENTIFIER_COLUMNS, *STATUS_COLUMNS, *STATEMENT_FEATURE_COLUMNS],
        ].sort_values(["contest_id", "index"], kind="mergesort")
        fetch_report = fetch_report.sort_values(
            ["contest_id", "index"],
            kind="mergesort",
        )

        paths = {
            "statement_features_parquet": output_dir / "statement_features.parquet",
            "statement_features_csv": output_dir / "statement_features.csv",
            "statement_feature_columns": output_dir / "statement_feature_columns.json",
            "statement_feature_summary": output_dir / "statement_feature_summary.json",
            "statement_fetch_report": output_dir / "statement_fetch_report.csv",
        }
        features.to_parquet(paths["statement_features_parquet"], engine="pyarrow", index=False)
        features.to_csv(paths["statement_features_csv"], index=False)
        fetch_report.to_csv(paths["statement_fetch_report"], index=False)
        write_json(
            paths["statement_feature_columns"],
            {
                "identifier_columns": list(IDENTIFIER_COLUMNS),
                "status_columns": list(STATUS_COLUMNS),
                "feature_columns": list(STATEMENT_FEATURE_COLUMNS),
                "notes": [
                    "Features are approximate statement-structure indicators.",
                    "Solved-count fields are not used by this module.",
                ],
            },
        )
        summary = _summary_payload(
            input_row_count=len(feature_table),
            attempted_page_count=len(problem_rows),
            feature_rows=features,
            fetch_report=fetch_report,
        )
        write_json(paths["statement_feature_summary"], summary)
        logger.info(
            "Completed Codeforces statement feature extraction",
            extra={
                "event": "statement_features_completed",
                "details": {
                    "input_rows": len(feature_table),
                    "attempted_pages": len(problem_rows),
                    "cached_pages": summary["cached_page_count"],
                    "fetched_pages": summary["fetched_page_count"],
                    "failed_fetches": summary["failed_fetch_count"],
                    "parsed_successes": summary["parsed_success_count"],
                    "parsed_failures": summary["parsed_failure_count"],
                    "output_dir": output_dir.as_posix(),
                },
            },
        )
        return paths
    except Exception:
        logger.exception(
            "Codeforces statement feature extraction failed",
            extra={"event": "statement_features_failed"},
        )
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the statement-feature command-line parser."""
    parser = argparse.ArgumentParser(
        description="Extract lightweight Codeforces problem statement features."
    )
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the statement-feature extraction CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        paths = run_statement_feature_extraction(
            feature_path=args.feature_path,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            sleep_seconds=args.sleep_seconds,
            timeout=args.timeout,
            max_pages=args.max_pages,
            log_path=args.log_path,
        )
    except (StatementFeatureError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote statement features: {paths['statement_features_parquet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
