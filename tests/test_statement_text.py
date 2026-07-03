"""Tests for cached Codeforces statement text extraction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import statement_text


def _synthetic_html() -> str:
    return """
    <html>
      <head>
        <style>.hidden { display: none; }</style>
        <script>window.secret = "do not keep";</script>
      </head>
      <body>
        <div class="problem-statement">
          <div class="header">
            <div class="title">C. Graph Paths &amp; Trees</div>
            <div class="time-limit">time limit per test 2 seconds</div>
            <div class="memory-limit">memory limit per test 256 megabytes</div>
          </div>
          <div>
            <p>You are given a graph with n vertices and m edges.</p>
            <p>Find the shortest path where ai &lt;= 10.</p>
          </div>
          <div class="input-specification">
            <div class="section-title">Input</div>
            <p>The first line contains n and m.</p>
          </div>
          <div class="output-specification">
            <div class="section-title">Output</div>
            <p>Print the answer.</p>
          </div>
          <div class="sample-tests">
            <div class="sample-test">
              <div class="input"><div class="title">Input</div><pre>3 2</pre></div>
              <div class="output"><div class="title">Output</div><pre>1</pre></div>
            </div>
          </div>
          <div class="note">
            <div class="section-title">Note</div>
            <p>The sample uses one tree edge.</p>
          </div>
        </div>
      </body>
    </html>
    """


def test_cache_filename_construction() -> None:
    path = statement_text.cache_path_for_problem(Path("cache"), 1000, "A")
    assert path == Path("cache") / "1000_A.html"
    special = statement_text.cache_path_for_problem(Path("cache"), 1791, "C+")
    assert special.name == "1791_C.html"


def test_html_tag_stripping() -> None:
    text = statement_text.strip_html_tags("<p>Hello <b>world</b>&nbsp;!</p>")
    assert text == "Hello world !"


def test_script_style_removal() -> None:
    text = statement_text.strip_html_tags(
        "<style>.x{}</style><script>alert(1)</script><p>Keep me</p>"
    )
    assert text == "Keep me"
    assert "alert" not in text


def test_section_extraction_from_synthetic_codeforces_html() -> None:
    parsed = statement_text.parse_statement_html(_synthetic_html())
    assert parsed.status == "parsed"
    assert parsed.title_text == "C. Graph Paths & Trees"
    assert "given a graph" in parsed.statement_text
    assert "first line contains n and m" in parsed.input_text
    assert "Print the answer" in parsed.output_text
    assert "tree edge" in parsed.note_text
    assert "3 2" in parsed.examples_text


def test_text_normalization() -> None:
    normalized = statement_text.normalize_text("a&nbsp;&le;&nbsp;b\n  and x ≥ y")
    assert normalized == "a <= b and x >= y"


def test_extracting_full_row_from_synthetic_cached_html(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    statement_text.cache_path_for_problem(cache_dir, 1000, "C").write_text(
        _synthetic_html(),
        encoding="utf-8",
    )
    row = pd.Series({"contest_id": 1000, "index": "C", "name": "Graph Paths", "rating": 1600})
    record = statement_text.extract_text_for_row(row, cache_dir)
    assert record["text_extract_status"] == "parsed"
    assert record["html_cache_found"] is True
    assert record["statement_text_available"] is True
    assert "window.secret" not in record["combined_text"]
    assert "<div" not in record["combined_text"]


def test_missing_cached_html_handling(tmp_path: Path) -> None:
    row = pd.Series({"contest_id": 1000, "index": "A", "name": "Missing", "rating": 800})
    record = statement_text.extract_text_for_row(row, tmp_path)
    assert record["text_extract_status"] == "missing_cache"
    assert record["html_cache_found"] is False
    assert record["combined_text"] == ""


def test_summary_generation() -> None:
    records = pd.DataFrame(
        [
            {
                "html_cache_found": True,
                "text_extract_status": "parsed",
                "statement_text_available": True,
                "combined_text": "abc",
            },
            {
                "html_cache_found": False,
                "text_extract_status": "missing_cache",
                "statement_text_available": False,
                "combined_text": "",
            },
        ]
    )
    summary = statement_text.build_summary(records, input_row_count=2)
    assert summary["cached_html_found_count"] == 1
    assert summary["cached_html_missing_count"] == 1
    assert summary["extracted_success_count"] == 1
    assert summary["statement_text_available_rate"] == 0.5
    assert "This module does not train models." in summary["conservative_notes"]


def test_cli_smoke_with_tiny_synthetic_model_table_and_cache(tmp_path: Path) -> None:
    feature_path = tmp_path / "model_table.parquet"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "statement_text"
    log_path = tmp_path / "logs" / "statement_text.log"
    cache_dir.mkdir()
    pd.DataFrame(
        [
            {"contest_id": 1000, "index": "C", "name": "Graph Paths", "rating": 1600},
            {"contest_id": 1000, "index": "D", "name": "Missing Page", "rating": 1800},
        ]
    ).to_parquet(feature_path, engine="pyarrow", index=False)
    statement_text.cache_path_for_problem(cache_dir, 1000, "C").write_text(
        _synthetic_html(),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cf_diff.statement_text",
            "--feature-path",
            str(feature_path),
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
            "--log-path",
            str(log_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    parquet_path = output_dir / "statement_text.parquet"
    csv_path = output_dir / "statement_text.csv"
    summary_path = output_dir / "statement_text_summary.json"
    assert parquet_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()
    assert log_path.exists()
    output = pd.read_parquet(parquet_path, engine="pyarrow")
    assert len(output) == 2
    assert output.loc[0, "text_extract_status"] == "parsed"
    assert output.loc[1, "text_extract_status"] == "missing_cache"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["input_model_table_rows"] == 2
    assert summary["cached_html_found_count"] == 1
