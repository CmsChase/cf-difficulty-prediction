"""Acquire exact raw Codeforces API responses with provenance metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

import requests

API_BASE_URL: Final[str] = "https://codeforces.com/api"
API_MIN_INTERVAL_SECONDS: Final[float] = 2.0
DEFAULT_SLEEP_SECONDS: Final[float] = 2.1
DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
DEFAULT_LOG_PATH: Final[Path] = Path("outputs/logs/fetch.log")
MAX_ATTEMPTS: Final[int] = 3
INITIAL_BACKOFF_SECONDS: Final[float] = 1.0
TRANSIENT_HTTP_STATUS_CODES: Final[frozenset[int]] = frozenset(
    {429, 500, 502, 503, 504}
)

_rate_limit_lock = threading.Lock()
_last_request_time: float | None = None


class CodeforcesAPIError(RuntimeError):
    """Raised when a Codeforces request or API payload is invalid."""


@dataclass(frozen=True)
class ResourceSpec:
    """Describe one Codeforces API resource to acquire."""

    endpoint: str
    query_params: dict[str, object]
    output_filename: str


@dataclass(frozen=True)
class FetchResult:
    """Hold one validated response and its exact raw representation."""

    spec: ResourceSpec
    request_url: str
    fetched_at_utc: str
    http_status: int
    api_status: str
    raw_bytes: bytes
    payload: dict[str, object]
    record_counts: dict[str, int]


class JsonLogFormatter(logging.Formatter):
    """Format acquisition log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a machine-readable JSON log line."""
        payload: dict[str, object] = {
            "timestamp_utc": utc_now().isoformat(timespec="milliseconds"),
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


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def create_snapshot_id(timestamp: datetime | None = None) -> str:
    """Create a compact UTC snapshot identifier."""
    value = timestamp if timestamp is not None else utc_now()
    if value.tzinfo is None:
        raise ValueError("Snapshot timestamps must be timezone-aware.")
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def compute_sha256(data: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of exact response bytes."""
    return hashlib.sha256(data).hexdigest()


def configure_fetch_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """Create a dedicated structured file logger for fetch activity."""
    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("cf_diff.fetch_api")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(resolved_path, encoding="utf-8")
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    """Flush, close, and detach all handlers from a dedicated logger."""
    for handler in tuple(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _wait_for_request_slot() -> None:
    """Throttle reusable helper calls to at most one request per two seconds."""
    global _last_request_time

    with _rate_limit_lock:
        now = time.monotonic()
        if _last_request_time is not None:
            remaining = API_MIN_INTERVAL_SECONDS - (now - _last_request_time)
            if remaining > 0:
                time.sleep(remaining)
        _last_request_time = time.monotonic()


def _is_transient_request_error(error: requests.RequestException) -> bool:
    """Return whether a request error is suitable for retrying."""
    if error.response is None:
        return True
    return error.response.status_code in TRANSIENT_HTTP_STATUS_CODES


def _validate_api_payload(payload: object, method_name: str) -> dict[str, object]:
    """Validate the generic top-level Codeforces API response contract."""
    if not isinstance(payload, dict):
        raise CodeforcesAPIError(
            f"Codeforces API method {method_name!r} returned a non-object "
            "JSON payload."
        )

    status = payload.get("status")
    if not isinstance(status, str):
        raise CodeforcesAPIError(
            f"Codeforces API method {method_name!r} returned JSON without "
            "a string top-level 'status' field."
        )
    if status != "OK":
        comment = payload.get("comment")
        detail = f": {comment}" if comment else ""
        raise CodeforcesAPIError(
            f"Codeforces API method {method_name!r} returned status "
            f"{status!r}{detail}"
        )
    if "result" not in payload:
        raise CodeforcesAPIError(
            f"Codeforces API method {method_name!r} returned status 'OK' "
            "without a top-level 'result' field."
        )
    return cast(dict[str, object], payload)


def extract_record_counts(
    endpoint: str,
    payload: Mapping[str, object],
) -> dict[str, int]:
    """Validate a resource-specific result and return its record counts."""
    if endpoint == "problemset.problems":
        result = payload.get("result")
        if not isinstance(result, dict):
            raise CodeforcesAPIError(
                "problemset.problems response 'result' must be an object."
            )
        problems = result.get("problems")
        problem_statistics = result.get("problemStatistics")
        if not isinstance(problems, list):
            raise CodeforcesAPIError(
                "problemset.problems result must contain a 'problems' list."
            )
        if not isinstance(problem_statistics, list):
            raise CodeforcesAPIError(
                "problemset.problems result must contain a "
                "'problemStatistics' list."
            )
        return {
            "problems": len(problems),
            "problem_statistics": len(problem_statistics),
        }

    if endpoint == "contest.list":
        contests = payload.get("result")
        if not isinstance(contests, list):
            raise CodeforcesAPIError(
                "contest.list response 'result' must be a list."
            )
        return {"contests": len(contests)}

    raise ValueError(f"Unsupported Codeforces endpoint: {endpoint!r}")


def extract_summary(
    problemset_response: dict[str, object],
    contest_response: dict[str, object],
) -> dict[str, int]:
    """Extract combined counts from the two supported API responses."""
    return {
        **extract_record_counts("problemset.problems", problemset_response),
        **extract_record_counts("contest.list", contest_response),
    }


def fetch_codeforces_api(
    method_name: str,
    params: dict[str, object] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Fetch and validate one API method for reuse by downstream modules."""
    if not method_name or "/" in method_name:
        raise ValueError("method_name must be a non-empty API method name.")
    if timeout <= 0:
        raise ValueError("timeout must be positive.")

    url = f"{API_BASE_URL}/{method_name}"
    request_params = dict(params) if params is not None else {}

    with requests.Session() as session:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            _wait_for_request_slot()
            try:
                response = session.get(
                    url,
                    params=request_params,
                    timeout=timeout,
                )
                response.raise_for_status()
            except requests.RequestException as error:
                if (
                    attempt == MAX_ATTEMPTS
                    or not _is_transient_request_error(error)
                ):
                    raise CodeforcesAPIError(
                        f"Codeforces API request for {method_name!r} failed "
                        f"on attempt {attempt}/{MAX_ATTEMPTS}: {error}"
                    ) from error
                time.sleep(INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)))
                continue

            try:
                payload = response.json()
            except ValueError as error:
                raise CodeforcesAPIError(
                    f"Codeforces API method {method_name!r} returned invalid JSON."
                ) from error
            return _validate_api_payload(payload, method_name)

    raise AssertionError("The retry loop exited unexpectedly.")


def fetch_problemset_problems(lang: str = "en") -> dict[str, object]:
    """Fetch the public problemset and aggregate problem statistics."""
    return fetch_codeforces_api(
        "problemset.problems",
        params={"lang": lang},
    )


def fetch_contest_list(
    gym: bool = False,
    lang: str = "en",
) -> dict[str, object]:
    """Fetch public contests, excluding gym contests by default."""
    return fetch_codeforces_api(
        "contest.list",
        params={"gym": str(gym).lower(), "lang": lang},
    )


def _prepared_request_url(
    endpoint: str,
    query_params: Mapping[str, object],
) -> str:
    """Build the full request URL when a mocked response has no URL."""
    request = requests.Request(
        "GET",
        f"{API_BASE_URL}/{endpoint}",
        params=dict(query_params),
    )
    prepared = request.prepare()
    if prepared.url is None:
        raise AssertionError("requests failed to prepare a request URL.")
    return prepared.url


def fetch_raw_resource(
    session: requests.Session,
    spec: ResourceSpec,
    *,
    timeout: int,
) -> FetchResult:
    """Fetch, parse, and validate one resource while preserving exact bytes."""
    if timeout <= 0:
        raise ValueError("timeout must be positive.")

    endpoint_url = f"{API_BASE_URL}/{spec.endpoint}"
    try:
        response = session.get(
            endpoint_url,
            params=spec.query_params,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise CodeforcesAPIError(
            f"HTTP request for {spec.endpoint!r} failed: {error}"
        ) from error

    raw_bytes = response.content
    try:
        decoded = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CodeforcesAPIError(
            f"Codeforces API method {spec.endpoint!r} returned malformed JSON."
        ) from error

    payload = _validate_api_payload(decoded, spec.endpoint)
    record_counts = extract_record_counts(spec.endpoint, payload)
    request_url = response.url or _prepared_request_url(
        spec.endpoint,
        spec.query_params,
    )
    return FetchResult(
        spec=spec,
        request_url=request_url,
        fetched_at_utc=utc_now().isoformat(timespec="seconds"),
        http_status=response.status_code,
        api_status=cast(str, payload["status"]),
        raw_bytes=raw_bytes,
        payload=payload,
        record_counts=record_counts,
    )


def build_resource_manifest(
    result: FetchResult,
    output_path: Path,
) -> dict[str, object]:
    """Build provenance metadata for one exact raw response."""
    return {
        "endpoint": result.spec.endpoint,
        "request_url": result.request_url,
        "query_params": result.spec.query_params,
        "fetched_at_utc": result.fetched_at_utc,
        "http_status": result.http_status,
        "api_status": result.api_status,
        "sha256": compute_sha256(result.raw_bytes),
        "bytes": len(result.raw_bytes),
        "output_path": output_path.as_posix(),
        "top_level_keys": sorted(result.payload.keys()),
        "record_counts": result.record_counts,
    }


def _write_bytes_exclusive(path: Path, data: bytes) -> None:
    """Write immutable raw bytes, refusing to replace an existing file."""
    with path.open("xb") as output_file:
        output_file.write(data)


def _write_json_exclusive(path: Path, payload: object) -> None:
    """Write a new UTF-8 JSON file without replacing existing content."""
    data = (
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _write_bytes_exclusive(path, data)


def _publish_latest(snapshot_dir: Path, latest_dir: Path) -> None:
    """Replace the latest directory with a complete copy of the snapshot."""
    snapshot_dir = snapshot_dir.resolve()
    latest_dir = latest_dir.resolve()
    if latest_dir == snapshot_dir:
        raise ValueError("latest_dir must differ from the snapshot directory.")

    latest_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = latest_dir.with_name(
        f".{latest_dir.name}.tmp-{uuid.uuid4().hex}"
    )
    try:
        shutil.copytree(snapshot_dir, staging_dir)
        if latest_dir.exists():
            if latest_dir.is_dir():
                shutil.rmtree(latest_dir)
            else:
                latest_dir.unlink()
        staging_dir.replace(latest_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def acquire_snapshot(
    output_root: Path,
    *,
    lang: str = "en",
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    latest_dir: Path = Path("data/raw/codeforces/latest"),
    log_path: Path = DEFAULT_LOG_PATH,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Path:
    """Fetch both resources and publish an immutable raw snapshot."""
    if sleep_seconds < API_MIN_INTERVAL_SECONDS:
        raise ValueError(
            f"sleep_seconds must be at least {API_MIN_INTERVAL_SECONDS}."
        )
    if timeout <= 0:
        raise ValueError("timeout must be positive.")

    run_timestamp = utc_now()
    snapshot_id = create_snapshot_id(run_timestamp)
    created_at_utc = run_timestamp.isoformat(timespec="seconds")
    output_root = output_root.resolve()
    snapshot_dir = output_root / snapshot_id
    if snapshot_dir.exists():
        raise FileExistsError(
            f"Snapshot directory already exists: {snapshot_dir}"
        )

    logger = configure_fetch_logger(log_path)
    specs = (
        ResourceSpec(
            endpoint="problemset.problems",
            query_params={"lang": lang},
            output_filename="problemset.problems.json",
        ),
        ResourceSpec(
            endpoint="contest.list",
            query_params={"gym": "false", "lang": lang},
            output_filename="contest.list.json",
        ),
    )

    logger.info(
        "Starting Codeforces snapshot acquisition",
        extra={
            "event": "snapshot_started",
            "details": {
                "snapshot_id": snapshot_id,
                "output_root": str(output_root),
            },
        },
    )

    results: list[FetchResult] = []
    try:
        with requests.Session() as session:
            for index, spec in enumerate(specs):
                if index:
                    logger.info(
                        "Sleeping between Codeforces requests",
                        extra={
                            "event": "rate_limit_sleep",
                            "details": {"seconds": sleep_seconds},
                        },
                    )
                    sleep_fn(sleep_seconds)
                result = fetch_raw_resource(
                    session,
                    spec,
                    timeout=timeout,
                )
                results.append(result)
                logger.info(
                    "Fetched Codeforces API resource",
                    extra={
                        "event": "resource_fetched",
                        "details": {
                            "endpoint": spec.endpoint,
                            "http_status": result.http_status,
                            "api_status": result.api_status,
                            "bytes": len(result.raw_bytes),
                            "sha256": compute_sha256(result.raw_bytes),
                            "record_counts": result.record_counts,
                        },
                    },
                )

        snapshot_dir.mkdir(parents=True, exist_ok=False)
        resources: dict[str, dict[str, object]] = {}
        for result in results:
            raw_path = snapshot_dir / result.spec.output_filename
            _write_bytes_exclusive(raw_path, result.raw_bytes)
            resources[result.spec.endpoint] = build_resource_manifest(
                result,
                raw_path.relative_to(output_root),
            )

        manifest = {
            "snapshot_id": snapshot_id,
            "created_at_utc": created_at_utc,
            "resources": resources,
        }
        _write_json_exclusive(snapshot_dir / "manifest.json", manifest)
        _publish_latest(snapshot_dir, latest_dir)
    except Exception:
        logger.exception(
            "Codeforces snapshot acquisition failed",
            extra={
                "event": "snapshot_failed",
                "details": {"snapshot_id": snapshot_id},
            },
        )
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        _close_logger(logger)
        raise

    logger.info(
        "Completed Codeforces snapshot acquisition",
        extra={
            "event": "snapshot_completed",
            "details": {
                "snapshot_id": snapshot_id,
                "snapshot_dir": str(snapshot_dir),
                "latest_dir": str(latest_dir.resolve()),
            },
        },
    )
    _close_logger(logger)
    return snapshot_dir


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for raw acquisition."""
    parser = argparse.ArgumentParser(
        description="Fetch exact raw Codeforces API snapshots."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--latest-dir",
        type=Path,
        default=Path("data/raw/codeforces/latest"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Codeforces raw acquisition CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        snapshot_dir = acquire_snapshot(
            args.output_root,
            lang=args.lang,
            sleep_seconds=args.sleep_seconds,
            timeout=args.timeout,
            latest_dir=args.latest_dir,
        )
    except (CodeforcesAPIError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Created Codeforces raw snapshot: {snapshot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
