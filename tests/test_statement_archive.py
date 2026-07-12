"""Tests for deterministic statement-cache archiving."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import statement_archive


def _read_manifest(path: Path) -> list[dict[str, str]]:
    """Read a generated CSV manifest."""
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _create_paths(tmp_path: Path, name: str = "archive") -> tuple[Path, Path]:
    """Return manifest and summary paths outside the cache directory."""
    output_dir = tmp_path / name
    return output_dir / "manifest.csv", output_dir / "summary.json"


def test_create_is_sorted_deterministic_and_uses_relative_paths(tmp_path: Path) -> None:
    """Equivalent scans have stable bytes and sorted portable paths."""
    cache = tmp_path / "cache"
    (cache / "z").mkdir(parents=True)
    (cache / "a").mkdir()
    (cache / "z" / "second.html").write_bytes(b"<html>second</html>")
    (cache / "a" / "first.html").write_bytes(b"<!doctype html>first")
    fixed_ns = 1_700_000_000_123_456_700
    for path in cache.rglob("*.html"):
        os.utime(path, ns=(fixed_ns, fixed_ns))

    manifest_one, summary_one = _create_paths(tmp_path, "archive-one")
    manifest_two, summary_two = _create_paths(tmp_path, "archive-two")
    statement_archive.create_archive(cache, manifest_one, summary_one)
    statement_archive.create_archive(cache, manifest_two, summary_two)

    assert manifest_one.read_bytes() == manifest_two.read_bytes()
    assert summary_one.read_bytes() == summary_two.read_bytes()
    rows = _read_manifest(manifest_one)
    assert [row["relative_path"] for row in rows] == [
        "a/first.html",
        "z/second.html",
    ]
    assert all(row["kind"] == "html" for row in rows)
    assert all(row["mtime_utc"].endswith("Z") for row in rows)
    generated_text = manifest_one.read_text(encoding="utf-8") + summary_one.read_text(
        encoding="utf-8"
    )
    assert str(tmp_path) not in generated_text
    assert statement_archive.CONTENT_SEMANTICS in generated_text


def test_pdf_empty_and_other_files_are_summarized(tmp_path: Path) -> None:
    """Content classification and deterministic summary counts cover all kinds."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "document.bin").write_bytes(b"%PDF-1.7\n")
    (cache / "empty.pdf").write_bytes(b"")
    (cache / "note.txt").write_bytes(b"plain text")
    manifest, summary = _create_paths(tmp_path)

    result = statement_archive.create_archive(cache, manifest, summary)
    rows = {row["relative_path"]: row for row in _read_manifest(manifest)}

    assert rows["document.bin"]["kind"] == "pdf"
    assert rows["empty.pdf"]["kind"] == "empty"
    assert rows["note.txt"]["kind"] == "other"
    assert result["kind_counts"] == {
        "empty": 1,
        "html": 0,
        "other": 1,
        "pdf": 1,
    }
    assert json.loads(summary.read_text(encoding="utf-8")) == result


def test_verify_reports_hash_size_and_kind_changes(tmp_path: Path) -> None:
    """Tampering is reported across every changed content attribute."""
    cache = tmp_path / "cache"
    cache.mkdir()
    target = cache / "page.bin"
    target.write_bytes(b"other")
    manifest, summary = _create_paths(tmp_path)
    statement_archive.create_archive(cache, manifest, summary)

    target.write_bytes(b"%PDF-1.7")
    report = statement_archive.verify_archive(cache, manifest)

    assert report["ok"] is False
    assert report["hash_mismatches"] == ["page.bin"]
    assert report["size_mismatches"] == ["page.bin"]
    assert report["kind_mismatches"] == ["page.bin"]


def test_verify_reports_missing_and_extra_files(tmp_path: Path) -> None:
    """Tree membership changes are reported with sorted relative paths."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "kept.html").write_bytes(b"<html>kept</html>")
    missing = cache / "missing.html"
    missing.write_bytes(b"<html>missing</html>")
    manifest, summary = _create_paths(tmp_path)
    statement_archive.create_archive(cache, manifest, summary)

    missing.unlink()
    (cache / "z-extra.html").write_bytes(b"<html>extra</html>")
    (cache / "a-extra.html").write_bytes(b"<html>extra</html>")
    report = statement_archive.verify_archive(cache, manifest)

    assert report["ok"] is False
    assert report["missing"] == ["missing.html"]
    assert report["extra"] == ["a-extra.html", "z-extra.html"]


def test_create_refuses_overwrite_and_cli_verifies(tmp_path: Path) -> None:
    """Create is append-safe by default and verify is available through the CLI."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "page.html").write_bytes(b"<html>page</html>")
    manifest, summary = _create_paths(tmp_path)

    assert statement_archive.main(
        [
            "create",
            "--cache-dir",
            str(cache),
            "--manifest",
            str(manifest),
            "--summary",
            str(summary),
        ]
    ) == 0
    original_manifest = manifest.read_bytes()
    original_summary = summary.read_bytes()

    with pytest.raises(statement_archive.ArchiveError, match="overwrite"):
        statement_archive.create_archive(cache, manifest, summary)
    assert manifest.read_bytes() == original_manifest
    assert summary.read_bytes() == original_summary
    assert statement_archive.main(
        [
            "verify",
            "--cache-dir",
            str(cache),
            "--manifest",
            str(manifest),
        ]
    ) == 0
