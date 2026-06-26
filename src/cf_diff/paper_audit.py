"""Audit generated paper artifacts and map claims to result evidence."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

EXPECTED_SECTION_FILES: Final[tuple[str, ...]] = (
    "01_abstract_en.md",
    "01_abstract_cn.md",
    "02_introduction_en.md",
    "02_introduction_cn.md",
    "03_data_en.md",
    "03_data_cn.md",
    "04_methods_en.md",
    "04_methods_cn.md",
    "05_results_en.md",
    "05_results_cn.md",
    "06_ablation_en.md",
    "06_ablation_cn.md",
    "07_error_analysis_en.md",
    "07_error_analysis_cn.md",
    "08_limitations_en.md",
    "08_limitations_cn.md",
    "09_conclusion_en.md",
    "09_conclusion_cn.md",
)

EXPECTED_TABLE_FILES: Final[tuple[str, ...]] = (
    "dataset_summary.md",
    "main_model_results.md",
    "baseline_improvements.md",
    "ablation_drop_comparison.md",
    "error_by_tag_top.md",
    "error_by_index_rank.md",
)

EXPECTED_FIGURE_FILES: Final[tuple[str, ...]] = (
    "solved_count_hist_log.png",
    "test_mae_by_model.png",
    "within_200_by_model.png",
    "predicted_vs_actual_contest_grouped.png",
    "predicted_vs_actual_forward_time.png",
    "feature_drop_mae_change.png",
    "ablation_mae_by_feature_set_contest_grouped.png",
    "ablation_mae_by_feature_set_forward_time.png",
    "error_by_tag_top15_contest_grouped.png",
    "error_by_tag_top15_forward_time.png",
    "error_by_index_rank.png",
)


class PaperAuditError(RuntimeError):
    """Raised when the paper audit cannot proceed safely."""


class JsonLogFormatter(logging.Formatter):
    """Format paper-audit logs as JSON Lines."""

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


def configure_logger(log_path: Path) -> logging.Logger:
    """Create the dedicated structured paper-audit logger."""
    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.paper_audit")
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


def _write_text(path: Path, text: str) -> None:
    """Write normalized UTF-8 text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def _write_json(path: Path, payload: object) -> None:
    """Write deterministic UTF-8 JSON."""
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def read_json(path: Path) -> dict[str, object]:
    """Read a JSON object, returning an empty object when absent."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def count_words(text: str) -> int:
    """Count English-like tokens and CJK characters in Markdown text."""
    without_code = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    without_links = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", without_code)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", without_links)
    latin_tokens = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", without_links)
    return len(cjk_chars) + len(latin_tokens)


def list_relative_files(directory: Path, pattern: str) -> list[str]:
    """List files under a directory as sorted POSIX-style relative paths."""
    if not directory.exists():
        return []
    return sorted(path.relative_to(directory).as_posix() for path in directory.glob(pattern))


def missing_expected(found: Sequence[str], expected: Sequence[str]) -> list[str]:
    """Return expected filenames not present in a found-file list."""
    found_set = set(found)
    return [name for name in expected if name not in found_set]


def build_audit_summary(paper_dir: Path) -> dict[str, object]:
    """Build a structural summary of the current paper package."""
    section_files = list_relative_files(paper_dir / "sections", "*.md")
    table_files = list_relative_files(paper_dir / "tables", "*.md")
    figure_files = list_relative_files(paper_dir / "figures", "*.png")
    paper_en = paper_dir / "paper_en.md"
    paper_cn = paper_dir / "paper_cn.md"
    en_text = paper_en.read_text(encoding="utf-8") if paper_en.exists() else ""
    cn_text = paper_cn.read_text(encoding="utf-8") if paper_cn.exists() else ""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paper_section_files_found": section_files,
        "paper_table_files_found": table_files,
        "paper_figure_files_found": figure_files,
        "paper_en_word_count": count_words(en_text),
        "paper_cn_word_count": count_words(cn_text),
        "missing_expected_sections": missing_expected(
            section_files,
            EXPECTED_SECTION_FILES,
        ),
        "missing_expected_tables": missing_expected(
            table_files,
            EXPECTED_TABLE_FILES,
        ),
        "missing_expected_figures": missing_expected(
            figure_files,
            EXPECTED_FIGURE_FILES,
        ),
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Format a small Markdown table without external dependencies."""
    def _cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def build_evidence_rows(outputs_dir: Path) -> list[dict[str, str]]:
    """Build evidence-map rows from existing result files."""
    dataset_path = outputs_dir / "eda/summary/dataset_summary.json"
    analysis_path = outputs_dir / "analysis/summary/analysis_summary.json"
    ablation_path = outputs_dir / "ablations/summary/ablation_summary.json"
    ranking_path = outputs_dir / "analysis/tables/model_ranking_test.csv"
    improvements_path = outputs_dir / "analysis/tables/baseline_improvements.csv"
    drop_path = outputs_dir / "ablations/tables/ablation_drop_comparison.csv"
    split_path = Path("data/processed/splits/split_summary.json")

    dataset = read_json(dataset_path)
    analysis = read_json(analysis_path)
    ablation = read_json(ablation_path)
    split = read_json(split_path)

    row_count = dataset.get("row_counts", {}).get("processed", "unknown")
    rating = dataset.get("rating_range", {})
    quantiles = dataset.get("solved_count_quantiles", {})
    contest_overlap = split.get("contest_grouped", {}).get(
        "contest_overlap_count",
        "unknown",
    )
    strictly_ordered = split.get("forward_time", {}).get(
        "strictly_ordered",
        "unknown",
    )
    best = analysis.get("best_model_by_strategy", {})
    best_contest = best.get("contest_grouped", {}).get("model_name", "unknown")
    best_forward = best.get("forward_time", {}).get("model_name", "unknown")
    best_drop = ablation.get("feature_group_importance_notes", [{}])[0].get(
        "most_important_removed_group",
        "solved",
    )

    return [
        {
            "claim": f"dataset contains {row_count:,} rated programming problems"
            if isinstance(row_count, int)
            else "dataset row count is reported",
            "source": dataset_path.as_posix(),
            "evidence": f"row_counts.processed = {row_count}",
        },
        {
            "claim": (
                f"rating range is {rating.get('min', 'unknown')}–"
                f"{rating.get('max', 'unknown')}"
            ),
            "source": dataset_path.as_posix(),
            "evidence": f"rating_range = {rating}",
        },
        {
            "claim": "solved_count is highly skewed",
            "source": dataset_path.as_posix(),
            "evidence": (
                f"p50 = {quantiles.get('p50')}, p99 = {quantiles.get('p99')}, "
                f"max = {quantiles.get('max')}"
            ),
        },
        {
            "claim": "contest-grouped split has zero contest overlap",
            "source": split_path.as_posix(),
            "evidence": f"contest_overlap_count = {contest_overlap}",
        },
        {
            "claim": "forward-time split is strictly ordered",
            "source": split_path.as_posix(),
            "evidence": f"strictly_ordered = {strictly_ordered}",
        },
        {
            "claim": "solved-count-only is the strongest simple baseline",
            "source": ranking_path.as_posix(),
            "evidence": (
                "Check model_ranking_test.csv: solved_count_only_baseline ranks "
                "ahead of index_only_baseline and tag_only_baseline."
            ),
        },
        {
            "claim": "full models improve over solved-count-only",
            "source": improvements_path.as_posix(),
            "evidence": (
                f"Best full models are {best_contest} and {best_forward}; "
                "baseline_improvements.csv reports improvement over "
                "solved_count_only_baseline."
            ),
        },
        {
            "claim": "removing solved features causes the largest MAE increase",
            "source": drop_path.as_posix(),
            "evidence": (
                f"ablation summary marks {best_drop} as the largest one-group "
                "drop effect; verify per strategy/model in the CSV."
            ),
        },
        {
            "claim": (
                "forward-time gaps should be described as temporal "
                "generalization gaps, not automatic overfitting"
            ),
            "source": analysis_path.as_posix(),
            "evidence": "generalization_gap_notes contains temporal_generalization_gap wording.",
        },
    ]


def write_evidence_map(outputs_dir: Path, output_path: Path) -> None:
    """Write a claim-to-source evidence map."""
    rows = build_evidence_rows(outputs_dir)
    _write_text(
        output_path,
        "# Evidence Map\n\n"
        + markdown_table(
            ["Claim", "Source file", "Evidence to preserve"],
            [[row["claim"], row["source"], row["evidence"]] for row in rows],
        ),
    )


def _figure_metadata(filename: str) -> tuple[str, str, str]:
    """Return intended section, caption, and placement for one figure."""
    mapping = {
        "solved_count_hist_log.png": (
            "Data",
            "Distribution of log-transformed solved counts, showing strong skew.",
            "main paper",
        ),
        "test_mae_by_model.png": (
            "Results",
            "Test MAE by model for contest-grouped and forward-time evaluation.",
            "main paper",
        ),
        "within_200_by_model.png": (
            "Results",
            "Share of predictions within 200 rating points by model.",
            "main paper",
        ),
        "predicted_vs_actual_contest_grouped.png": (
            "Results",
            "Predicted versus actual ratings for the best contest-grouped model.",
            "appendix",
        ),
        "predicted_vs_actual_forward_time.png": (
            "Results",
            "Predicted versus actual ratings for the best forward-time model.",
            "appendix",
        ),
        "feature_drop_mae_change.png": (
            "Ablation Study",
            "MAE change when removing each feature group from all API features.",
            "main paper",
        ),
        "ablation_mae_by_feature_set_contest_grouped.png": (
            "Ablation Study",
            "Contest-grouped test MAE across ablation feature sets.",
            "main paper",
        ),
        "ablation_mae_by_feature_set_forward_time.png": (
            "Ablation Study",
            "Forward-time test MAE across ablation feature sets.",
            "main paper",
        ),
        "error_by_tag_top15_contest_grouped.png": (
            "Error Analysis",
            "Tags with highest mean absolute error in contest-grouped evaluation.",
            "main paper",
        ),
        "error_by_tag_top15_forward_time.png": (
            "Error Analysis",
            "Tags with highest mean absolute error in forward-time evaluation.",
            "appendix",
        ),
        "error_by_index_rank.png": (
            "Error Analysis",
            "Mean absolute error by problem index rank.",
            "main paper",
        ),
    }
    return mapping.get(
        filename,
        (
            "Appendix",
            f"Supplementary diagnostic figure: {filename}.",
            "appendix",
        ),
    )


def write_figure_checklist(paper_dir: Path, output_path: Path) -> None:
    """Write figure audit checklist for found and expected figures."""
    figure_dir = paper_dir / "figures"
    found = set(list_relative_files(figure_dir, "*.png"))
    filenames = sorted(found | set(EXPECTED_FIGURE_FILES))
    rows = []
    for filename in filenames:
        section, caption, placement = _figure_metadata(filename)
        rows.append([filename, section, filename in found, caption, placement])
    _write_text(
        output_path,
        "# Figure Checklist\n\n"
        + markdown_table(
            [
                "Filename",
                "Intended section",
                "Exists",
                "Caption draft",
                "Placement",
            ],
            rows,
        ),
    )


def _table_metadata(filename: str) -> tuple[str, str, str]:
    """Return intended section, caption, and placement for one table."""
    mapping = {
        "dataset_summary.md": (
            "Data",
            "Summary of dataset size, rating range, splits, and solved-count skew.",
            "main paper",
        ),
        "main_model_results.md": (
            "Results",
            "Test-set model ranking by MAE for both evaluation strategies.",
            "main paper",
        ),
        "baseline_improvements.md": (
            "Results",
            "MAE improvement of best full models over standard baselines.",
            "main paper",
        ),
        "ablation_drop_comparison.md": (
            "Ablation Study",
            "MAE change from removing each feature group.",
            "main paper",
        ),
        "error_by_tag_top.md": (
            "Error Analysis",
            "Highest-error tags for the best model under each strategy.",
            "main paper",
        ),
        "error_by_index_rank.md": (
            "Error Analysis",
            "Mean absolute error grouped by problem index rank.",
            "appendix",
        ),
    }
    return mapping[filename]


def write_table_checklist(paper_dir: Path, output_path: Path) -> None:
    """Write table audit checklist for found and expected tables."""
    table_dir = paper_dir / "tables"
    found = set(list_relative_files(table_dir, "*.md"))
    filenames = sorted(found | set(EXPECTED_TABLE_FILES))
    rows = []
    for filename in filenames:
        section, caption, placement = _table_metadata(filename)
        rows.append([filename, section, filename in found, caption, placement])
    _write_text(
        output_path,
        "# Table Checklist\n\n"
        + markdown_table(
            [
                "Filename",
                "Intended section",
                "Exists",
                "Caption draft",
                "Placement",
            ],
            rows,
        ),
    )


def build_revision_plan_text() -> str:
    """Return section-by-section expansion guidance."""
    return """# Revision Plan for an 8–12 Page Paper

The current paper is a short skeleton. It records the main pipeline and headline results, but each section needs expansion, citations, and tighter argumentation before it can function as a full research paper.

## Abstract

- Current weakness: too compressed; it reads like a project summary.
- Expand: add one sentence on motivation, one on data, one on evaluation, and one on the main empirical result.
- Cite: no figure/table citation needed in the abstract.
- Preserve numbers: 10,979 rated programming problems; rating range 800–3500; best-model MAE values from `model_ranking_test.csv`.
- Do not overclaim: avoid saying solved count measures intrinsic difficulty.

## Introduction

- Current weakness: lacks broader motivation and research questions.
- Expand: explain why difficulty prediction matters for practice, what official ratings represent, and why public API metadata is a constrained but reproducible setting.
- Cite: `dataset_summary.md` only if the introduction mentions dataset scale.
- Preserve numbers: dataset size and API-only constraint.
- Do not overclaim: do not claim this model replaces official ratings.

## Related Work

- Current weakness: missing from the current skeleton.
- Expand: add work on code-contest analytics, educational recommendation, difficulty estimation, and metadata/submission-behavior signals.
- Cite: external literature must be added manually by the researcher.
- Preserve numbers: none from project outputs.
- Do not overclaim: keep this section as positioning, not evidence for project-specific results.

## Data

- Current weakness: needs fuller explanation of API fields, filtering, and solved-count skew.
- Expand: describe raw endpoints, rated `PROGRAMMING` filter, split construction, missingness, and solve-count distribution.
- Cite: `dataset_summary.md`; `solved_count_hist_log.png`.
- Preserve numbers: 10,979 rows; rating range 800–3500; solved-count p50 around 4167, p99 around 73912, max around 700377; zero contest overlap; forward-time strict ordering.
- Do not overclaim: solved count reflects exposure and time as well as difficulty.

## Methods

- Current weakness: model and feature details are terse.
- Expand: define feature groups, target, baseline families, full models, split strategies, and metrics.
- Cite: `feature_group_definitions.csv`; `main_model_results.md` for metrics.
- Preserve numbers: feature counts from `feature_summary.json` and ablation definitions.
- Do not overclaim: describe models as baselines/structured regressors, not state-of-the-art.

## Results

- Current weakness: needs clearer comparison narrative.
- Expand: compare mean, index-only, tag-only, solved-count-only, ridge, random forest, and histogram gradient boosting across both splits.
- Cite: `main_model_results.md`; `baseline_improvements.md`; `test_mae_by_model.png`; `within_200_by_model.png`.
- Preserve numbers: solved-count-only ranking; best model per split; full-model improvement over solved-count-only.
- Do not overclaim: forward-time results should be discussed as temporal generalization behavior.

## Ablation Study

- Current weakness: currently states headline result without interpreting group removals carefully.
- Expand: explain feature groups and one-group drop design; compare all features to all-without-index, all-without-solved, all-without-tags, and all-without-points.
- Cite: `ablation_drop_comparison.md`; `feature_drop_mae_change.png`.
- Preserve numbers: removing solved features causes the largest MAE increase.
- Do not overclaim: ablations show predictive contribution, not causal importance.

## Error Analysis

- Current weakness: diagnostic observations need examples and grouping logic.
- Expand: discuss top errors, tag-level errors, and index-rank patterns; include a few qualitative examples from `top_error_cases.csv`.
- Cite: `error_by_tag_top.md`; `error_by_index_rank.md`; `error_by_tag_top15_contest_grouped.png`; `error_by_index_rank.png`.
- Preserve numbers: counts and MAE values from generated error tables.
- Do not overclaim: avoid claiming why a tag is hard without manual inspection.

## Limitations

- Current weakness: good start but should be more explicit.
- Expand: cover API-only data, lack of statement text, solved-count confounding, snapshot dependence, and temporal distribution shift.
- Cite: optional reference to solved-count skew and forward-time split.
- Preserve numbers: no new metrics needed.
- Do not overclaim: acknowledge preliminary and snapshot-specific nature without weakening reproducibility.

## Conclusion

- Current weakness: concise but should connect findings to next steps.
- Expand: restate main empirical findings and propose future work on text, temporal solve dynamics, and independent reproduction.
- Cite: no new citations necessary.
- Preserve numbers: dataset size and headline findings only.
- Do not overclaim: frame as reproducible baseline evidence.
"""


def build_final_outline_text() -> str:
    """Return detailed final outline with target word counts."""
    return """# Final Paper Outline and Target Word Counts

## Abstract (180–220 words)

- Motivation: predicting contest problem difficulty from public signals.
- Dataset and API-only constraint.
- Main methods: structured metadata, solved statistics, grouped/time splits.
- Main findings: solved-count-only is strongest simple baseline; full models improve; solved features dominate ablations.
- One conservative limitation sentence.

## Introduction (700–900 words)

- Practical importance of difficulty prediction.
- Why Codeforces ratings are a useful target.
- Reproducibility motivation for official API-only data.
- Research questions:
  1. How predictive are solved statistics and metadata?
  2. How do grouped and temporal evaluation differ?
  3. Which feature groups matter most?
- Contributions and paper structure.

## Related Work (500–700 words)

- Programming-contest analytics.
- Educational recommendation and difficulty estimation.
- Metadata and behavioral signals in prediction.
- Leakage-aware and temporal evaluation.
- Position this project as a reproducible structured-data baseline.

## Data (700–900 words)

- Codeforces API endpoints and snapshotting.
- Filtering to rated programming problems.
- Normalized tables and merged problem statistics.
- Dataset size, rating range, tag distribution, solved-count skew.
- Contest-grouped and forward-time split construction.
- Tables/figures to cite: `dataset_summary.md`, `solved_count_hist_log.png`.

## Methods (900–1200 words)

- Target variable and identifiers.
- Feature groups: index, solved, tags, points.
- Baselines and full models.
- Ablation models and feature-set definitions.
- Metrics: MAE, RMSE, R², within-100, within-200.
- Reproducibility and deterministic seeds.

## Results (900–1200 words)

- Main model ranking by test MAE.
- Compare contest-grouped and forward-time results.
- Discuss solved-count-only as strongest simple baseline.
- Discuss full-model improvement over solved-count-only.
- Interpret forward-time gaps as temporal generalization/distribution shift.
- Tables/figures: `main_model_results.md`, `baseline_improvements.md`, `test_mae_by_model.png`, `within_200_by_model.png`.

## Ablation Study (600–800 words)

- Explain all feature sets and one-group drops.
- Present drop comparison.
- Emphasize solved features as largest drop effect.
- Discuss smaller but nonzero metadata/tag contributions.
- Tables/figures: `ablation_drop_comparison.md`, `feature_drop_mae_change.png`.

## Error Analysis (500–700 words)

- Top absolute errors.
- Error by tag.
- Error by index rank.
- Qualitative discussion of why structured features may fail.
- Tables/figures: `top_error_cases.csv`, `error_by_tag_top.md`, `error_by_index_rank.md`.

## Limitations (400–600 words)

- API-only scope.
- No statement text or editorials.
- Solved-count confounding by age/exposure/popularity.
- Snapshot dependence.
- Forward-time distribution shift.
- Need independent reproduction.

## Conclusion (250–400 words)

- Recap dataset and reproducible pipeline.
- Recap strongest findings.
- Future work: text features, temporal solved trajectories, larger snapshots, independent replication.
"""


def run_paper_audit(
    *,
    paper_dir: Path,
    outputs_dir: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Path]:
    """Run the paper audit and write all audit artifacts."""
    logger = configure_logger(log_path)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = build_audit_summary(paper_dir)
        paths = {
            "audit_summary": output_dir / "audit_summary.json",
            "evidence_map": output_dir / "evidence_map.md",
            "figure_checklist": output_dir / "figure_checklist.md",
            "table_checklist": output_dir / "table_checklist.md",
            "revision_plan": output_dir / "revision_plan.md",
            "final_paper_outline": output_dir / "final_paper_outline.md",
        }
        _write_json(paths["audit_summary"], summary)
        write_evidence_map(outputs_dir, paths["evidence_map"])
        write_figure_checklist(paper_dir, paths["figure_checklist"])
        write_table_checklist(paper_dir, paths["table_checklist"])
        _write_text(paths["revision_plan"], build_revision_plan_text())
        _write_text(paths["final_paper_outline"], build_final_outline_text())
        logger.info(
            "Completed paper audit",
            extra={
                "event": "paper_audit_completed",
                "details": {
                    "paper_dir": paper_dir.as_posix(),
                    "output_dir": output_dir.as_posix(),
                    "paper_en_word_count": summary["paper_en_word_count"],
                    "paper_cn_word_count": summary["paper_cn_word_count"],
                },
            },
        )
        return paths
    except Exception:
        logger.exception("Paper audit failed", extra={"event": "paper_audit_failed"})
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the paper-audit command-line parser."""
    parser = argparse.ArgumentParser(
        description="Audit generated paper artifacts and evidence coverage."
    )
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_audit"),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("outputs/logs/paper_audit.log"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the paper audit CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        paths = run_paper_audit(
            paper_dir=args.paper_dir,
            outputs_dir=args.outputs_dir,
            output_dir=args.output_dir,
            log_path=args.log_path,
        )
    except (PaperAuditError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote paper audit summary: {paths['audit_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
