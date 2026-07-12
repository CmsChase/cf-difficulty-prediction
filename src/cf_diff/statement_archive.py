"""Archive and verify the current normalized statement cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Final


CONTENT_SEMANTICS: Final[str] = (
    "current normalized cache content, not original HTTP response bytes"
)
MANIFEST_FIELDS: Final[tuple[str, ...]] = (
    "relative_path",
    "sha256",
    "size_bytes",
    "mtime_utc",
    "kind",
    "content_semantics",
)
KINDS: Final[tuple[str, ...]] = ("html", "pdf", "empty", "other")


class ArchiveError(RuntimeError):
    """Raised when an archive cannot be created or verified safely."""


def _format_mtime_utc(mtime_ns: int) -> str:
    """Format a filesystem timestamp as an exact UTC ISO-8601 value."""
    seconds, nanoseconds = divmod(mtime_ns, 1_000_000_000)
    timestamp = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{timestamp:%Y-%m-%dT%H:%M:%S}.{nanoseconds:09d}Z"


def classify_content(relative_path: str, data: bytes) -> str:
    """Classify cached bytes conservatively as HTML, PDF, empty, or other."""
    if not data:
        return "empty"

    prefix = data[:4096].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    suffix = PurePosixPath(relative_path).suffix.lower()
    if prefix.startswith(b"%pdf-") or suffix == ".pdf":
        return "pdf"
    if suffix in {".html", ".htm"} or prefix.startswith(
        (b"<!doctype html", b"<html", b"<head", b"<body")
    ):
        return "html"
    return "other"


def scan_cache(cache_dir: Path) -> list[dict[str, object]]:
    """Return sorted manifest rows for every regular file in a cache tree."""
    root = cache_dir.resolve()
    if not root.is_dir():
        raise ArchiveError("Cache directory does not exist or is not a directory.")

    paths: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ArchiveError(
                f"Symbolic links are not supported in the cache: "
                f"{path.relative_to(root).as_posix()}"
            )
        if path.is_file():
            paths.append((path.relative_to(root).as_posix(), path))

    rows: list[dict[str, object]] = []
    for relative_path, path in sorted(paths, key=lambda item: item[0]):
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or len(data) != after.st_size
        ):
            raise ArchiveError(f"Cache file changed while being read: {relative_path}")
        rows.append(
            {
                "relative_path": relative_path,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
                "mtime_utc": _format_mtime_utc(after.st_mtime_ns),
                "kind": classify_content(relative_path, data),
                "content_semantics": CONTENT_SEMANTICS,
            }
        )
    return rows


def _manifest_bytes(rows: Sequence[dict[str, object]]) -> bytes:
    """Serialize manifest rows as deterministic UTF-8 CSV."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=MANIFEST_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _summary_bytes(rows: Sequence[dict[str, object]], manifest: bytes) -> bytes:
    """Serialize a deterministic, path-free archive summary."""
    counts = Counter(str(row["kind"]) for row in rows)
    summary = {
        "schema_version": 1,
        "content_semantics": CONTENT_SEMANTICS,
        "entry_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "kind_counts": {kind: counts[kind] for kind in KINDS},
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    return (
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _is_within(path: Path, directory: Path) -> bool:
    """Return whether a resolved path is inside a resolved directory."""
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _validate_output_paths(
    cache_dir: Path,
    manifest_path: Path,
    summary_path: Path,
) -> tuple[Path, Path]:
    """Resolve output paths and keep generated files outside the cache tree."""
    root = cache_dir.resolve()
    manifest = manifest_path.resolve()
    summary = summary_path.resolve()
    if manifest == summary:
        raise ArchiveError("Manifest and summary paths must be different.")
    if _is_within(manifest, root) or _is_within(summary, root):
        raise ArchiveError("Manifest and summary must be outside the cache directory.")
    return manifest, summary


def _atomic_write(path: Path, data: bytes, *, overwrite: bool) -> None:
    """Publish complete bytes atomically, refusing replacement by default."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path)
            except FileExistsError as error:
                raise ArchiveError(f"Refusing to overwrite {path.name}.") from error
            temporary_path.unlink()
    finally:
        temporary_path.unlink(missing_ok=True)


def create_archive(
    cache_dir: Path,
    manifest_path: Path,
    summary_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Create a deterministic manifest and summary for the current cache."""
    manifest, summary = _validate_output_paths(
        cache_dir,
        manifest_path,
        summary_path,
    )
    if not overwrite and (manifest.exists() or summary.exists()):
        raise ArchiveError("Refusing to overwrite existing archive artifacts.")

    rows = scan_cache(cache_dir)
    manifest_data = _manifest_bytes(rows)
    summary_data = _summary_bytes(rows, manifest_data)
    _atomic_write(manifest, manifest_data, overwrite=overwrite)
    _atomic_write(summary, summary_data, overwrite=overwrite)
    return json.loads(summary_data)


def read_manifest(manifest_path: Path) -> list[dict[str, object]]:
    """Read and validate one archive manifest."""
    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
                raise ArchiveError("Manifest header does not match the archive schema.")
            raw_rows = list(reader)
    except FileNotFoundError as error:
        raise ArchiveError("Manifest file does not exist.") from error

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in raw_rows:
        relative_path = raw["relative_path"]
        portable_path = PurePosixPath(relative_path)
        if (
            not relative_path
            or portable_path.is_absolute()
            or ".." in portable_path.parts
            or "\\" in relative_path
            or portable_path.as_posix() != relative_path
        ):
            raise ArchiveError("Manifest contains an unsafe relative path.")
        if relative_path in seen:
            raise ArchiveError(f"Manifest contains a duplicate path: {relative_path}")
        seen.add(relative_path)
        try:
            size_bytes = int(raw["size_bytes"])
        except ValueError as error:
            raise ArchiveError(
                f"Manifest contains an invalid size: {relative_path}"
            ) from error
        if size_bytes < 0:
            raise ArchiveError(f"Manifest contains an invalid size: {relative_path}")
        rows.append(
            {
                **raw,
                "size_bytes": size_bytes,
            }
        )
    return rows


def verify_archive(cache_dir: Path, manifest_path: Path) -> dict[str, object]:
    """Compare the current cache with a previously created manifest."""
    root = cache_dir.resolve()
    manifest = manifest_path.resolve()
    if _is_within(manifest, root):
        raise ArchiveError("Manifest must be outside the cache directory.")

    expected_rows = read_manifest(manifest)
    current_rows = scan_cache(root)
    expected = {str(row["relative_path"]): row for row in expected_rows}
    current = {str(row["relative_path"]): row for row in current_rows}

    expected_paths = set(expected)
    current_paths = set(current)
    shared_paths = sorted(expected_paths & current_paths)
    report: dict[str, object] = {
        "missing": sorted(expected_paths - current_paths),
        "extra": sorted(current_paths - expected_paths),
        "hash_mismatches": [
            path
            for path in shared_paths
            if expected[path]["sha256"] != current[path]["sha256"]
        ],
        "size_mismatches": [
            path
            for path in shared_paths
            if expected[path]["size_bytes"] != current[path]["size_bytes"]
        ],
        "kind_mismatches": [
            path
            for path in shared_paths
            if expected[path]["kind"] != current[path]["kind"]
        ],
    }
    report["ok"] = not any(report[key] for key in report)
    return report


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the statement-cache archive command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Archive or verify current normalized statement-cache content; "
            "this does not preserve original HTTP response bytes."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="Create manifest and summary files.")
    create.add_argument("--cache-dir", type=Path, required=True)
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--summary", type=Path, required=True)
    create.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing manifest and summary files.",
    )

    verify = commands.add_parser("verify", help="Verify a cache against a manifest.")
    verify.add_argument("--cache-dir", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the statement-cache archive CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        if args.command == "create":
            summary = create_archive(
                args.cache_dir,
                args.manifest,
                args.summary,
                overwrite=args.overwrite,
            )
            print(f"Archived {summary['entry_count']} current cache files.")
            return 0

        report = verify_archive(args.cache_dir, args.manifest)
    except (ArchiveError, OSError, csv.Error, UnicodeError) as error:
        print(f"error: {error}")
        return 1

    if report["ok"]:
        print("Archive verification passed.")
        return 0
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
