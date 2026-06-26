"""Tests for paper audit artifact generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cf_diff import paper_audit


def test_count_words_counts_english_and_chinese() -> None:
    """Word counting handles English tokens and CJK characters."""
    text = "# Title\n\nThis is a short test. 这是测试。"

    assert paper_audit.count_words(text) == 10


def test_missing_expected_files() -> None:
    """Missing-file helper reports expected names absent from found files."""
    missing = paper_audit.missing_expected(
        ["01_abstract_en.md", "paper_en.md"],
        ["01_abstract_en.md", "02_introduction_en.md"],
    )

    assert missing == ["02_introduction_en.md"]


def test_build_audit_summary_reports_missing_files_and_word_counts(
    tmp_path: Path,
) -> None:
    """Audit summary scans paper files and counts current draft words."""
    paper_dir = tmp_path / "paper"
    (paper_dir / "sections").mkdir(parents=True)
    (paper_dir / "tables").mkdir()
    (paper_dir / "figures").mkdir()
    (paper_dir / "sections" / "01_abstract_en.md").write_text(
        "# Abstract\n\nA short abstract.",
        encoding="utf-8",
    )
    (paper_dir / "tables" / "dataset_summary.md").write_text(
        "| a |\n|---|\n| b |\n",
        encoding="utf-8",
    )
    (paper_dir / "figures" / "test_mae_by_model.png").write_bytes(b"png")
    (paper_dir / "paper_en.md").write_text(
        "# Abstract\n\nA short paper draft.",
        encoding="utf-8",
    )
    (paper_dir / "paper_cn.md").write_text(
        "# 摘要\n\n简短草稿。",
        encoding="utf-8",
    )

    summary = paper_audit.build_audit_summary(paper_dir)

    assert summary["paper_en_word_count"] == 5
    assert summary["paper_cn_word_count"] >= 4
    assert "02_introduction_en.md" in summary["missing_expected_sections"]
    assert "main_model_results.md" in summary["missing_expected_tables"]
    assert "within_200_by_model.png" in summary["missing_expected_figures"]


def test_evidence_map_generation_from_synthetic_outputs(tmp_path: Path) -> None:
    """Evidence map uses result files as claim sources."""
    outputs_dir = tmp_path / "outputs"
    (outputs_dir / "eda" / "summary").mkdir(parents=True)
    (outputs_dir / "analysis" / "summary").mkdir(parents=True)
    (outputs_dir / "analysis" / "tables").mkdir(parents=True)
    (outputs_dir / "ablations" / "summary").mkdir(parents=True)
    (outputs_dir / "ablations" / "tables").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "splits").mkdir(parents=True)

    (outputs_dir / "eda" / "summary" / "dataset_summary.json").write_text(
        json.dumps(
            {
                "row_counts": {"processed": 10979},
                "rating_range": {"min": 800, "max": 3500},
                "solved_count_quantiles": {
                    "p50": 4167,
                    "p99": 73912,
                    "max": 700377,
                },
            }
        ),
        encoding="utf-8",
    )
    (outputs_dir / "analysis" / "summary" / "analysis_summary.json").write_text(
        json.dumps(
            {
                "best_model_by_strategy": {
                    "contest_grouped": {"model_name": "hgb"},
                    "forward_time": {"model_name": "rf"},
                }
            }
        ),
        encoding="utf-8",
    )
    (outputs_dir / "ablations" / "summary" / "ablation_summary.json").write_text(
        json.dumps(
            {
                "feature_group_importance_notes": [
                    {"most_important_removed_group": "solved"}
                ]
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "evidence_map.md"
    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        paper_audit.write_evidence_map(outputs_dir, output_path)
    finally:
        os.chdir(old_cwd)

    text = output_path.read_text(encoding="utf-8")
    assert "dataset contains 10,979 rated programming problems" in text
    assert "rating range is 800–3500" in text
    assert "solved_count is highly skewed" in text
    assert "outputs/eda/summary/dataset_summary.json" in text


def test_run_paper_audit_writes_required_outputs(tmp_path: Path) -> None:
    """Tiny smoke test writes the audit artifact set."""
    paper_dir = tmp_path / "paper"
    outputs_dir = tmp_path / "outputs"
    output_dir = outputs_dir / "paper_audit"
    (paper_dir / "sections").mkdir(parents=True)
    (paper_dir / "tables").mkdir()
    (paper_dir / "figures").mkdir()
    (paper_dir / "paper_en.md").write_text("# Abstract\n\nShort draft.", encoding="utf-8")
    (paper_dir / "paper_cn.md").write_text("# 摘要\n\n短草稿。", encoding="utf-8")
    (paper_dir / "sections" / "01_abstract_en.md").write_text("# Abstract", encoding="utf-8")
    (paper_dir / "tables" / "dataset_summary.md").write_text("| a |\n|---|\n", encoding="utf-8")
    (paper_dir / "figures" / "test_mae_by_model.png").write_bytes(b"png")

    (outputs_dir / "eda" / "summary").mkdir(parents=True)
    (outputs_dir / "analysis" / "summary").mkdir(parents=True)
    (outputs_dir / "ablations" / "summary").mkdir(parents=True)
    (outputs_dir / "eda" / "summary" / "dataset_summary.json").write_text("{}", encoding="utf-8")
    (outputs_dir / "analysis" / "summary" / "analysis_summary.json").write_text("{}", encoding="utf-8")
    (outputs_dir / "ablations" / "summary" / "ablation_summary.json").write_text("{}", encoding="utf-8")

    paths = paper_audit.run_paper_audit(
        paper_dir=paper_dir,
        outputs_dir=outputs_dir,
        output_dir=output_dir,
        log_path=outputs_dir / "logs" / "paper_audit.log",
    )

    assert all(path.is_file() for path in paths.values())
    assert "short skeleton" in paths["revision_plan"].read_text(encoding="utf-8")
    assert "Related Work" in paths["final_paper_outline"].read_text(encoding="utf-8")
