"""Capture label-isolated T0 features for one prospective contest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Final, Sequence

import pandas as pd

from cf_diff.prospective_model import (
    ProspectiveModelError,
    _sha256_lf_text_file,
    load_frozen_protocol,
)
from cf_diff.statement_features import (
    STATEMENT_FEATURE_COLUMNS,
    USER_AGENT,
    FetchResult,
    _decode_bytes,
    build_problem_url,
    build_statement_feature_values,
    parse_problem_statement,
)


DEFAULT_PROTOCOL_PATH: Final[Path] = Path("configs/prospective_protocol_v2.json")
DEFAULT_SLEEP_SECONDS: Final[float] = 2.5
DEFAULT_TIMEOUT_SECONDS: Final[int] = 20
DEFAULT_RETRIES: Final[int] = 2
_DERIVED_INDEX_COLUMNS: Final[frozenset[str]] = frozenset(
    {"index_rank", "index_number"}
)
_DECODE_POLICY: Final[str] = "utf-8_errors_replace"
_PROTOCOL_DECODE_POLICY: Final[str] = (
    "Decode raw statement bytes as UTF-8 with replacement for invalid byte "
    "sequences; response Content-Type and final redirect URL are audit-only fields."
)


class ProspectiveInputError(RuntimeError):
    """Raised when a T0 capture cannot be completed without leakage or ambiguity."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: str | datetime, description: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise ProspectiveInputError(
                f"{description} must be an ISO-8601 timestamp."
            ) from error
    else:
        raise ProspectiveInputError(
            f"{description} must be an ISO-8601 timestamp."
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProspectiveInputError(f"{description} must include a UTC offset.")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_contest_id(value: object) -> str:
    if isinstance(value, bool):
        raise ProspectiveInputError("contest_id must be a positive integer.")
    normalized = str(value).strip()
    if not re.fullmatch(r"[0-9]+", normalized):
        raise ProspectiveInputError("contest_id must be a positive integer.")
    number = int(normalized)
    if number < 1:
        raise ProspectiveInputError("contest_id must be a positive integer.")
    return str(number)


def _normalize_indices(values: Sequence[object]) -> list[str]:
    if not values:
        raise ProspectiveInputError("At least one problem index is required.")
    normalized: list[str] = []
    for value in values:
        index = str(value).strip().upper()
        if not index or not re.fullmatch(r"[A-Z0-9]+", index):
            raise ProspectiveInputError(
                f"Invalid Codeforces problem index: {value!r}"
            )
        normalized.append(index)
    duplicates = sorted(
        {index for index in normalized if normalized.count(index) > 1}
    )
    if duplicates:
        raise ProspectiveInputError(
            f"Problem indices must be unique after normalization: {duplicates}"
        )
    return normalized


def _protocol_feature_columns(protocol: dict[str, object]) -> list[str]:
    bundle = protocol.get("model_bundle")
    if not isinstance(bundle, dict):
        raise ProspectiveInputError("Protocol model_bundle must be an object.")
    primary = bundle.get("primary_feature_columns")
    if not isinstance(primary, list) or any(
        not isinstance(column, str) for column in primary
    ):
        raise ProspectiveInputError(
            "Protocol primary_feature_columns must be a string list."
        )
    statement_columns = [
        column for column in primary if column not in _DERIVED_INDEX_COLUMNS
    ]
    if tuple(statement_columns) != STATEMENT_FEATURE_COLUMNS:
        raise ProspectiveInputError(
            "Protocol statement feature allowlist must exactly match the frozen "
            "statement extractor, including order."
        )
    if primary != [
        "index_rank",
        "index_number",
        *STATEMENT_FEATURE_COLUMNS,
    ]:
        raise ProspectiveInputError(
            "Protocol primary features must begin with the two internally derived "
            "index fields followed by the frozen statement feature allowlist."
        )
    capture = protocol.get("input_capture")
    if (
        not isinstance(capture, dict)
        or capture.get("decode_policy") != _PROTOCOL_DECODE_POLICY
    ):
        raise ProspectiveInputError(
            "Protocol input_capture.decode_policy does not match the frozen UTF-8 policy."
        )
    return statement_columns


def _cohort_window(protocol: dict[str, object]) -> tuple[datetime, datetime, int]:
    cohort = protocol.get("cohort")
    timepoint = protocol.get("prediction_timepoint")
    if not isinstance(cohort, dict) or not isinstance(timepoint, dict):
        raise ProspectiveInputError(
            "Protocol cohort and prediction_timepoint must be objects."
        )
    start = _parse_utc(
        cohort.get("eligibility_start_utc"),  # type: ignore[arg-type]
        "protocol.cohort.eligibility_start_utc",
    )
    end = _parse_utc(
        cohort.get("eligibility_end_utc"),  # type: ignore[arg-type]
        "protocol.cohort.eligibility_end_utc",
    )
    deadline_minutes = timepoint.get("lock_deadline_minutes_after_contest_start")
    if (
        isinstance(deadline_minutes, bool)
        or not isinstance(deadline_minutes, int)
        or deadline_minutes < 1
    ):
        raise ProspectiveInputError(
            "Protocol lock deadline must be a positive integer."
        )
    return start, end, deadline_minutes


def _require_capture_time(
    now: datetime,
    contest_start: datetime,
    deadline: datetime,
    phase: str,
) -> None:
    current = _parse_utc(now, phase)
    if current < contest_start:
        raise ProspectiveInputError(
            f"{phase} is before contest_start_utc; T0 capture cannot begin."
        )
    if current > deadline:
        raise ProspectiveInputError(
            f"{phase} is after the frozen T0 lock deadline."
        )


def _fetch_fresh_problem_page(
    *,
    url: str,
    raw_path: Path,
    timeout: int,
    retries: int,
    deadline: datetime,
    clock: Callable[[], datetime],
) -> FetchResult:
    """Fetch one English problem page into a new, immutable raw file."""
    if raw_path.exists():
        raise ProspectiveInputError(
            f"Raw capture path already exists and cannot be reused: {raw_path}"
        )
    last_error = ""
    for attempt in range(1, retries + 1):
        remaining = (deadline - _parse_utc(clock(), "fetch clock")).total_seconds()
        if remaining <= 0:
            last_error = "T0 lock deadline reached before fetch completed."
            break
        request_timeout = max(0.1, min(float(timeout), remaining))
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                raw = response.read()
                status = getattr(response, "status", None)
                content_type = response.headers.get("Content-Type")
                final_url = response.geturl()
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with raw_path.open("xb") as handle:
                handle.write(raw)
            return FetchResult(
                status="fetched",
                cache_path=raw_path,
                html_text=_decode_bytes(raw, "text/html; charset=utf-8"),
                http_status=status,
                content_type=content_type,
                final_url=final_url,
            )
        except urllib.error.HTTPError as error:
            last_error = f"HTTP {error.code}: {error.reason}"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = f"{type(error).__name__}: {error}"
        if attempt < retries:
            remaining = (
                deadline - _parse_utc(clock(), "retry clock")
            ).total_seconds()
            if remaining <= 0:
                break
            time.sleep(min(1.0, remaining))
    return FetchResult(
        status="failed",
        cache_path=raw_path,
        html_text=None,
        error=last_error or "unknown fetch error",
    )


def _write_sidecar_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    created = False
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            created = True
            handle.write(serialized)
    except FileExistsError as error:
        raise ProspectiveInputError(
            f"Capture sidecar already exists and will not be overwritten: {path}"
        ) from error
    except Exception:
        if created and path.exists():
            path.unlink()
        raise


def _not_attempted_record(index: str, contest_id: str, reason: str) -> dict[str, object]:
    return {
        "index": index,
        "url": build_problem_url(contest_id, index),
        "fetch_started_at_utc": None,
        "fetch_completed_at_utc": None,
        "http_status": None,
        "fetch_status": "not_attempted",
        "parse_status": "not_parsed",
        "raw_html_sha256": None,
        "decoded_html_sha256": None,
        "raw_path": None,
        "response_content_type": None,
        "final_url": None,
        "error": reason,
    }


def capture_prospective_input(
    *,
    protocol_path: Path,
    contest_id: object,
    indices: Sequence[object],
    contest_start_utc: str | datetime,
    output_path: Path,
    sidecar_path: Path,
    raw_dir: Path,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    clock: Callable[[], datetime] | None = None,
    fetcher: Callable[..., FetchResult] | None = None,
) -> dict[str, Path]:
    """Capture one complete contest input without reading any label-bearing table."""
    if output_path.suffix.lower() != ".csv":
        raise ProspectiveInputError("Prospective T0 input must be written as CSV.")
    if timeout < 1 or retries < 1 or sleep_seconds < 0:
        raise ProspectiveInputError(
            "timeout and retries must be positive; sleep_seconds cannot be negative."
        )
    existing = [
        path
        for path in (output_path, sidecar_path, raw_dir)
        if path.exists()
    ]
    if existing:
        raise ProspectiveInputError(
            "Capture targets already exist and will not be overwritten: "
            f"{existing}"
        )

    clock_fn = clock or _utc_now
    fetch_fn = fetcher or _fetch_fresh_problem_page
    normalized_contest = _normalize_contest_id(contest_id)
    normalized_indices = _normalize_indices(indices)
    try:
        protocol = load_frozen_protocol(protocol_path)
    except ProspectiveModelError as error:
        raise ProspectiveInputError(str(error)) from error
    statement_columns = _protocol_feature_columns(protocol)
    cohort_start, cohort_end, deadline_minutes = _cohort_window(protocol)
    contest_start = _parse_utc(contest_start_utc, "contest_start_utc")
    if not cohort_start <= contest_start <= cohort_end:
        raise ProspectiveInputError(
            "contest_start_utc is outside the frozen protocol cohort window."
        )
    deadline = contest_start + timedelta(minutes=deadline_minutes)
    capture_started = _parse_utc(clock_fn(), "capture start")
    _require_capture_time(
        capture_started,
        contest_start,
        deadline,
        "capture start",
    )

    raw_dir.mkdir(parents=True, exist_ok=False)
    protocol_sha256 = _sha256_file(protocol_path)
    problem_records: list[dict[str, object]] = []
    feature_records: list[dict[str, object]] = []
    failure_reason = ""

    for position, index in enumerate(normalized_indices):
        if position and sleep_seconds:
            time.sleep(sleep_seconds)
        try:
            _require_capture_time(
                _parse_utc(clock_fn(), "pre-fetch clock"),
                contest_start,
                deadline,
                "pre-fetch clock",
            )
        except ProspectiveInputError as error:
            failure_reason = str(error)
            problem_records.extend(
                _not_attempted_record(
                    remaining,
                    normalized_contest,
                    failure_reason,
                )
                for remaining in normalized_indices[position:]
            )
            break

        url = build_problem_url(normalized_contest, index)
        raw_path = raw_dir / f"{normalized_contest}_{index}.html"
        fetch_started = _parse_utc(clock_fn(), "fetch start")
        try:
            fetch_result = fetch_fn(
                url=url,
                raw_path=raw_path,
                timeout=timeout,
                retries=retries,
                deadline=deadline,
                clock=clock_fn,
            )
        except Exception as error:
            fetch_result = FetchResult(
                status="failed",
                cache_path=raw_path,
                html_text=None,
                error=f"{type(error).__name__}: {error}",
            )
        if not isinstance(fetch_result, FetchResult):
            fetch_result = FetchResult(
                status="failed",
                cache_path=raw_path,
                html_text=None,
                error="Fetcher returned an invalid result object.",
            )
        fetch_completed = _parse_utc(clock_fn(), "fetch completion")
        parse_status = "not_parsed"
        error_text = fetch_result.error
        raw_hash = _sha256_file(raw_path) if raw_path.is_file() else None
        decoded_html_hash = (
            _sha256_bytes(fetch_result.html_text.encode("utf-8"))
            if fetch_result.html_text is not None
            else None
        )
        feature_values: dict[str, object] | None = None
        if fetch_completed > deadline:
            error_text = "Fetch completed after the frozen T0 lock deadline."
        elif fetch_result.status != "fetched" or fetch_result.html_text is None:
            error_text = error_text or "Problem page was not freshly fetched."
        elif fetch_result.http_status != 200:
            error_text = (
                "Problem page fetch did not return HTTP 200: "
                f"{fetch_result.http_status!r}."
            )
        elif (
            not raw_path.is_file()
            or raw_hash is None
            or fetch_result.cache_path.resolve() != raw_path.resolve()
        ):
            error_text = "Fresh raw statement capture is missing or misplaced."
        elif fetch_result.html_text != _decode_bytes(
            raw_path.read_bytes(),
            "text/html; charset=utf-8",
        ):
            error_text = "Decoded statement does not match the frozen UTF-8 policy."
        else:
            parsed = parse_problem_statement(fetch_result.html_text)
            parse_status = parsed.status
            if parsed.status != "parsed":
                error_text = parsed.error or "Problem statement could not be parsed."
            else:
                feature_values = build_statement_feature_values(parsed)
                if feature_values.get("statement_available") != 1:
                    error_text = "Fetched page did not contain an available statement."

        problem_records.append(
            {
                "index": index,
                "url": url,
                "fetch_started_at_utc": _format_utc(fetch_started),
                "fetch_completed_at_utc": _format_utc(fetch_completed),
                "http_status": fetch_result.http_status,
                "fetch_status": fetch_result.status,
                "parse_status": parse_status,
                "raw_html_sha256": raw_hash,
                "decoded_html_sha256": decoded_html_hash,
                "raw_path": raw_path.as_posix() if raw_path.is_file() else None,
                "response_content_type": fetch_result.content_type,
                "final_url": fetch_result.final_url or url,
                "error": error_text,
            }
        )
        if error_text or feature_values is None:
            failure_reason = error_text or "T0 capture failed."
            problem_records.extend(
                _not_attempted_record(
                    remaining,
                    normalized_contest,
                    "Not attempted after an earlier capture failure.",
                )
                for remaining in normalized_indices[position + 1 :]
            )
            break
        feature_records.append(
            {
                "contest_id": normalized_contest,
                "index": index,
                **{column: feature_values[column] for column in statement_columns},
            }
        )

    base_sidecar: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "contest_id": normalized_contest,
        "contest_start_utc": _format_utc(contest_start),
        "contest_start_source": "explicit_cli_argument_unverified_at_t0",
        "lock_deadline_utc": _format_utc(deadline),
        "capture_started_at_utc": _format_utc(capture_started),
        "requested_indices": normalized_indices,
        "raw_capture_dir": raw_dir.as_posix(),
        "request_policy": {
            "source": "direct_public_problem_statement_pages",
            "metadata_api_used": False,
            "accept_language": "en-US,en;q=0.9",
            "decode_policy": _DECODE_POLICY,
        },
        "extractor_sha256": {
            "prospective_input": _sha256_lf_text_file(Path(__file__)),
            "statement_features": _sha256_lf_text_file(
                Path(__file__).with_name("statement_features.py")
            ),
        },
        "problems": problem_records,
    }
    if failure_reason:
        failed_at = _parse_utc(clock_fn(), "capture failure")
        base_sidecar.update(
            {
                "status": "failed",
                "capture_completed_at_utc": _format_utc(failed_at),
                "output": None,
                "error": failure_reason,
            }
        )
        _write_sidecar_exclusive(sidecar_path, base_sidecar)
        raise ProspectiveInputError(
            f"T0 capture failed; no model input was written: {failure_reason}"
        )

    columns = ["contest_id", "index", *statement_columns]
    frame = pd.DataFrame(feature_records, columns=columns)
    if (
        len(frame) != len(normalized_indices)
        or list(frame.columns) != columns
        or frame["contest_id"].nunique() != 1
        or frame.duplicated(["contest_id", "index"]).any()
        or frame["index"].tolist() != normalized_indices
    ):
        raise ProspectiveInputError(
            "Internal capture completeness or exact-schema check failed."
        )
    csv_text = frame.to_csv(index=False, lineterminator="\n")
    csv_bytes = csv_text.encode("utf-8")
    try:
        _require_capture_time(
            _parse_utc(clock_fn(), "pre-publication clock"),
            contest_start,
            deadline,
            "pre-publication clock",
        )
    except ProspectiveInputError as error:
        failed_at = _parse_utc(clock_fn(), "capture failure")
        base_sidecar.update(
            {
                "status": "failed",
                "capture_completed_at_utc": _format_utc(failed_at),
                "output": None,
                "error": str(error),
            }
        )
        _write_sidecar_exclusive(sidecar_path, base_sidecar)
        raise ProspectiveInputError(
            f"T0 capture failed; no model input was written: {error}"
        ) from error
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_created = False
    try:
        with output_path.open("xb") as handle:
            output_created = True
            handle.write(csv_bytes)
    except FileExistsError as error:
        raise ProspectiveInputError(
            f"Prospective input already exists and will not be overwritten: {output_path}"
        ) from error
    except Exception:
        if output_created and output_path.exists():
            output_path.unlink()
        raise

    capture_completed = _parse_utc(clock_fn(), "capture completion")
    if capture_completed > deadline:
        output_path.unlink()
        base_sidecar.update(
            {
                "status": "failed",
                "capture_completed_at_utc": _format_utc(capture_completed),
                "output": None,
                "error": "Input publication completed after the frozen T0 lock deadline.",
            }
        )
        _write_sidecar_exclusive(sidecar_path, base_sidecar)
        raise ProspectiveInputError(
            "T0 capture crossed the lock deadline; no model input was retained."
        )

    base_sidecar.update(
        {
            "status": "complete",
            "capture_completed_at_utc": _format_utc(capture_completed),
            "output": {
                "path": output_path.as_posix(),
                "columns": columns,
                "row_count": len(frame),
                "sha256": _sha256_bytes(csv_bytes),
            },
            "error": "",
        }
    )
    try:
        _write_sidecar_exclusive(sidecar_path, base_sidecar)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return {
        "input": output_path,
        "sidecar": sidecar_path,
        "raw_dir": raw_dir,
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one label-isolated prospective contest input directly from "
            "public problem statement pages."
        )
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--contest-id", required=True)
    parser.add_argument("--indices", nargs="+", required=True)
    parser.add_argument("--contest-start-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    try:
        paths = capture_prospective_input(
            protocol_path=args.protocol,
            contest_id=args.contest_id,
            indices=args.indices,
            contest_start_utc=args.contest_start_utc,
            output_path=args.output,
            sidecar_path=args.sidecar,
            raw_dir=args.raw_dir,
            timeout=args.timeout,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
        )
    except (ProspectiveInputError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote prospective T0 input: {paths['input']}")
    print(f"Wrote capture sidecar: {paths['sidecar']}")
    print(f"Wrote raw statement captures: {paths['raw_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
