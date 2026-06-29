"""Tests for Codeforces statement text-light feature extraction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import statement_features


SYNTHETIC_HTML = """
<html>
  <head><style>.x { color: red; }</style><script>ignored()</script></head>
  <body>
    <div class="problem-statement">
      <div class="header">
        <div class="time-limit">time limit per test: 1.5 seconds</div>
        <div class="memory-limit">memory limit per test: 256 megabytes</div>
      </div>
      <div>
        <p>You are given a tree with n vertices and m queries.</p>
        <p>Find the shortest path. Constraints: 1 &lt;= n &lt;= 200000.</p>
        <p>The solution should run in O(n log n).</p>
      </div>
      <div class="input-specification">
        <div class="section-title">Input</div>
        <p>The first line contains integer n.</p>
      </div>
      <div class="output-specification">
        <div class="section-title">Output</div>
        <p>Print the answer.</p>
      </div>
      <div class="sample-test">
        <div class="input"><pre>3\n1 2\n2 3</pre></div>
        <div class="output"><pre>2</pre></div>
      </div>
      <div class="note">
        <div class="section-title">Note</div>
        <p>This is a graph example.</p>
      </div>
    </div>
  </body>
</html>
"""


def test_url_construction() -> None:
    """Problem URLs use the public Codeforces problemset route."""
    assert (
        statement_features.build_problem_url(123, "A")
        == "https://codeforces.com/problemset/problem/123/A"
    )
    assert (
        statement_features.build_problem_url(123, "A1")
        == "https://codeforces.com/problemset/problem/123/A1"
    )


def test_html_tag_stripping() -> None:
    """HTML stripping removes tags/scripts and decodes entities."""
    text = statement_features.strip_html_tags(
        "<p>A &lt; B</p><script>bad()</script><style>x</style>"
    )

    assert "A < B" in text
    assert "bad()" not in text
    assert "<p>" not in text


def test_problem_statement_extraction() -> None:
    """Synthetic Codeforces-like HTML yields statement and section text."""
    parsed = statement_features.parse_problem_statement(SYNTHETIC_HTML)

    assert parsed.status == "parsed"
    assert "tree with n vertices" in parsed.statement_text
    assert "first line contains integer" in parsed.input_text
    assert "Print the answer" in parsed.output_text
    assert "graph example" in parsed.note_text


def test_time_limit_parsing() -> None:
    """Seconds and milliseconds are normalized to milliseconds."""
    assert statement_features.parse_time_limit_ms("1.5 seconds") == 1500
    assert statement_features.parse_time_limit_ms("750 milliseconds") == 750
    assert statement_features.parse_time_limit_ms("bad") is None


def test_memory_limit_parsing() -> None:
    """Memory limits are normalized to megabytes."""
    assert statement_features.parse_memory_limit_mb("256 megabytes") == 256
    assert statement_features.parse_memory_limit_mb("1 gigabyte") == 1024
    assert statement_features.parse_memory_limit_mb("bad") is None


def test_sample_count_extraction() -> None:
    """Sample blocks are counted from Codeforces-like sample markup."""
    assert statement_features.count_sample_tests(SYNTHETIC_HTML) == 1
    assert statement_features.count_sample_input_output_blocks(SYNTHETIC_HTML) == 2


def test_keyword_feature_extraction() -> None:
    """Keyword indicators use simple case-insensitive matching."""
    features = statement_features.extract_keyword_features(
        "A graph tree query asks for the shortest path."
    )

    assert features["kw_graph"] == 1
    assert features["kw_tree"] == 1
    assert features["kw_shortest_path"] == 1
    assert features["kw_query"] == 1
    assert features["kw_geometry"] == 0


def test_feature_row_creation_from_synthetic_html(tmp_path: Path) -> None:
    """Feature row creation returns identifiers, statuses, and text features."""
    cache_path = tmp_path / "123_A.html"
    fetch_result = statement_features.FetchResult(
        status="cached",
        cache_path=cache_path,
        html_text=SYNTHETIC_HTML,
    )

    row, report = statement_features.build_feature_row(
        contest_id=123,
        index="A",
        name="Synthetic Problem",
        url=statement_features.build_problem_url(123, "A"),
        fetch_result=fetch_result,
    )

    assert row["contest_id"] == "123"
    assert row["index"] == "A"
    assert row["statement_fetch_status"] == "cached"
    assert row["statement_parse_status"] == "parsed"
    assert row["statement_available"] == 1
    assert row["kw_graph"] == 1
    assert row["kw_tree"] == 1
    assert row["kw_shortest_path"] == 1
    assert row["time_limit_ms"] == 1500
    assert row["memory_limit_mb"] == 256
    assert report["parse_status"] == "parsed"


def test_cli_smoke_with_cached_dataset(tmp_path: Path) -> None:
    """Smoke test writes all required outputs using pre-cached HTML."""
    feature_path = tmp_path / "model_table.parquet"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "statement_features"
    log_path = tmp_path / "logs" / "statement_features.log"
    frame = pd.DataFrame(
        {
            "contest_id": [123],
            "index": ["A"],
            "name": ["Synthetic Problem"],
            "rating": [1200],
        }
    )
    frame.to_parquet(feature_path, engine="pyarrow", index=False)
    cache_dir.mkdir(parents=True)
    statement_features.cache_path_for_problem(cache_dir, 123, "A").write_text(
        SYNTHETIC_HTML,
        encoding="utf-8",
    )

    paths = statement_features.run_statement_feature_extraction(
        feature_path=feature_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
        sleep_seconds=0.0,
        timeout=1,
        max_pages=None,
        log_path=log_path,
    )

    assert all(path.is_file() for path in paths.values())
    assert log_path.is_file()
    output = pd.read_csv(paths["statement_features_csv"])
    assert len(output) == 1
    assert output.loc[0, "statement_fetch_status"] == "cached"
    assert output.loc[0, "statement_available"] == 1
    summary = json.loads(
        paths["statement_feature_summary"].read_text(encoding="utf-8")
    )
    assert summary["attempted_page_count"] == 1
    assert summary["cached_page_count"] == 1
    assert summary["fetched_page_count"] == 0
