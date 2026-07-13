from __future__ import annotations

import csv
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INDEX = DOCS / "index.html"
SCRIPT = DOCS / "evidence.js"
TEST_OUTPUT = ROOT / "outputs" / "historical_statement_backtest" / "test"


class _SiteParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.references: list[tuple[str, str, str]] = []
        self.evidence_views: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.ids.append(element_id)

        if tag in {"a", "link"} and attributes.get("href"):
            self.references.append((tag, "href", attributes["href"]))
        if tag in {"img", "script"} and attributes.get("src"):
            self.references.append((tag, "src", attributes["src"]))

        evidence_view = attributes.get("data-evidence")
        if evidence_view:
            self.evidence_views.append(evidence_view)


def _parse_site() -> tuple[_SiteParser, str]:
    html = INDEX.read_text(encoding="utf-8")
    parser = _SiteParser()
    parser.feed(html)
    return parser, html


def test_every_local_reference_stays_in_docs_and_exists() -> None:
    parser, _ = _parse_site()
    docs_root = DOCS.resolve()

    for tag, attribute, reference in parser.references:
        parsed = urlsplit(reference)
        if parsed.scheme or parsed.netloc:
            assert parsed.scheme == "https", (
                f"External {tag} {attribute} must use HTTPS: {reference}"
            )
            continue

        if parsed.path:
            target = (DOCS / unquote(parsed.path)).resolve()
            assert target.is_relative_to(docs_root), (
                f"Local {tag} {attribute} escapes the published docs root: {reference}"
            )
            assert target.is_file(), (
                f"Local {tag} {attribute} does not exist in docs: {reference}"
            )

        if parsed.fragment and not parsed.path:
            assert parsed.fragment in parser.ids, (
                f"Page fragment has no matching id: {reference}"
            )


def test_ids_are_unique_and_evidence_tabs_match_script_views() -> None:
    parser, _ = _parse_site()
    assert len(parser.ids) == len(set(parser.ids))

    expected_views = {
        "question",
        "split",
        "selection",
        "test",
        "uncertainty",
        "artifacts",
    }
    assert set(parser.evidence_views) == expected_views

    script = SCRIPT.read_text(encoding="utf-8")
    for view in expected_views:
        assert f"    {view}: {{" in script

    script_dom_ids = set(re.findall(r'getElementById\("([^"]+)"\)', script))
    assert script_dom_ids <= set(parser.ids)

    repository_paths = re.findall(r"\$\{repository\}/([^`]+)`", script)
    assert repository_paths
    for repository_path in repository_paths:
        assert (ROOT / repository_path).is_file(), (
            f"Evidence view links to a missing repository artifact: {repository_path}"
        )


def test_site_headline_values_match_committed_test_artifacts() -> None:
    _, html = _parse_site()
    metrics = json.loads((TEST_OUTPUT / "test_metrics.json").read_text(encoding="utf-8"))
    bootstrap = json.loads(
        (TEST_OUTPUT / "paired_bootstrap.json").read_text(encoding="utf-8")
    )

    comparator = metrics["settings"]["comparator"]
    primary = metrics["settings"]["primary"]
    difference = metrics["primary_mae_minus_comparator_mae"]
    lower = bootstrap["confidence_interval"]["lower"]
    upper = bootstrap["confidence_interval"]["upper"]

    assert f"MAE {comparator['mae']:.4f}" in html
    assert f"MAE {primary['mae']:.4f}" in html
    assert f"−{abs(difference):.4f}" in html
    assert f"[−{abs(lower):.4f}, −{abs(upper):.4f}]" in html
    assert f"{metrics['test_rows']:,}" in html
    assert f"{bootstrap['cluster_count']:,}" in html
    assert f"{bootstrap['resamples']:,}" in html
    assert f"seed {bootstrap['random_seed']}" in html

    with (TEST_OUTPUT / "coverage.csv").open(encoding="utf-8", newline="") as handle:
        coverage_rows = {row["split"]: row for row in csv.DictReader(handle)}
    test_coverage = float(coverage_rows["test"]["coverage"])
    assert f"{test_coverage:.2%}" in html

    script = SCRIPT.read_text(encoding="utf-8")
    overall_buckets = int(coverage_rows["overall"]["start_time_buckets"])
    assert f"{overall_buckets:,} unique contest start-time buckets" in script


def test_site_does_not_present_the_removed_heuristic_predictor() -> None:
    _, html = _parse_site()
    script = SCRIPT.read_text(encoding="utf-8")

    old_control_ids = {
        "indexRank",
        "tagFamily",
        "statementLength",
        "solvedAvailable",
        "difficultyRange",
    }
    for control_id in old_control_ids:
        assert f'id="{control_id}"' not in html

    assert "indexBase" not in script
    assert "tagOffset" not in script
    assert "Illustrative difficulty demo" not in html


def test_site_states_target_and_claim_boundary_explicitly() -> None:
    _, html = _parse_site()
    script = SCRIPT.read_text(encoding="utf-8")

    assert "Rating is the target, not an input." in html
    assert "Rating is the target, not an input." in script
    assert "It is retrospective—not a future blind test." in html
    assert "does not establish prospective performance" in html
