"""Summarize v6 semantic TF-IDF experiment outputs for paper drafting.

This module reads existing semantic TF-IDF metrics and summary artifacts. It
does not train models, rerun experiments, fetch data, or modify v5 results.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_METRICS_PATH: Final[Path] = Path(
    "outputs/semantic_tfidf/tables/semantic_tfidf_best_by_setting.csv"
)
DEFAULT_SUMMARY_PATH: Final[Path] = Path(
    "outputs/semantic_tfidf/summary/semantic_tfidf_summary.json"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path("outputs/semantic_tfidf/results_summary")
DEFAULT_LOG_PATH: Final[Path] = Path("outputs/logs/semantic_results.log")

RESEARCH_QUESTION: Final[str] = (
    "Can semantic problem-statement text features improve cold-start Codeforces "
    "difficulty prediction beyond metadata and lightweight statement-structure features?"
)
SETTING_ORDER: Final[tuple[str, ...]] = (
    "metadata_only",
    "text_light_only",
    "tfidf_text_only",
    "metadata_plus_text_light",
    "metadata_plus_tfidf",
    "metadata_plus_text_light_plus_tfidf",
    "full_api_reference",
)
IMPROVEMENT_COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    ("metadata_only", "metadata_plus_tfidf"),
    ("metadata_plus_text_light", "metadata_plus_text_light_plus_tfidf"),
    ("metadata_plus_tfidf", "metadata_plus_text_light_plus_tfidf"),
    ("tfidf_text_only", "metadata_plus_text_light_plus_tfidf"),
)
REQUIRED_METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "strategy",
    "feature_setting",
    "MAE",
    "RMSE",
    "R2",
    "within_100",
    "within_200",
    "validation_MAE",
    "selection_split",
    "report_split",
    "selection_rank",
)
CONSERVATIVE_LIMITATIONS: Final[tuple[str, ...]] = (
    "TF-IDF is classical bag-of-words modeling, not deep semantic understanding.",
    "Statement text extraction is approximate.",
    "Tags may be post-contest metadata, so this is metadata/statement cold-start, not strict pre-contest prediction.",
    "The full API reference in this v6 module uses ridge regression for comparison consistency.",
    "Generated outputs are local analysis artifacts.",
    "Historical v6 outcomes are retrospective under the 2026-07-10 public erratum.",
)
KEY_FINDINGS: Final[tuple[str, ...]] = (
    "TF-IDF alone is weak.",
    "Metadata + TF-IDF improves over metadata only.",
    "Metadata + text-light + TF-IDF improves over metadata + text-light.",
    "In the historical locked test report, the improvement is larger for forward-time than contest-grouped evaluation.",
    "The v6 full_api_reference is a ridge-based internal comparison and should not replace the historical v5 full API benchmark.",
)


class SemanticResultsError(RuntimeError):
    """Raised when semantic result summarization cannot proceed."""


class JsonLogFormatter(logging.Formatter):
    """Format semantic result logs as JSON Lines."""

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


def configure_logger(log_path: Path = DEFAULT_LOG_PATH) -> logging.Logger:
    """Create the dedicated structured semantic-results logger."""

    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.semantic_results")
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
    """Flush and close all handlers attached to the semantic-results logger."""

    for handler in tuple(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], table_name: str) -> None:
    """Raise a clear error when required columns are absent."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SemanticResultsError(f"{table_name} lacks columns: {missing}")


def load_metrics_csv(path: Path) -> pd.DataFrame:
    """Load validation-selected semantic settings and locked test reports."""

    if not path.exists():
        raise SemanticResultsError(f"Metrics file does not exist: {path}")
    metrics = pd.read_csv(path)
    _require_columns(metrics, REQUIRED_METRIC_COLUMNS, "metrics CSV")
    if "split_name" in metrics.columns:
        metrics = metrics.loc[metrics["split_name"].eq("test")].copy()
    if metrics.empty:
        raise SemanticResultsError("Metrics CSV contains no test rows.")
    if not metrics["selection_split"].eq("valid").all():
        raise SemanticResultsError("Metrics were not selected on validation rows.")
    if not metrics["report_split"].eq("test").all():
        raise SemanticResultsError("Metrics are not locked test reports.")
    if not pd.to_numeric(metrics["selection_rank"], errors="coerce").eq(1).all():
        raise SemanticResultsError("Metrics contain non-selected candidate rows.")
    if metrics.duplicated(["strategy", "feature_setting"]).any():
        raise SemanticResultsError(
            "Metrics contain duplicate strategy/feature-setting reports."
        )
    order = {setting: idx for idx, setting in enumerate(SETTING_ORDER)}
    metrics["_setting_order"] = metrics["feature_setting"].map(order).fillna(999)
    metrics = metrics.sort_values(
        ["strategy", "_setting_order", "feature_setting"],
        kind="mergesort",
    ).drop(columns=["_setting_order"])
    return metrics.reset_index(drop=True)


def load_summary_json(path: Path) -> dict[str, object]:
    """Load the semantic TF-IDF experiment summary JSON."""

    if not path.exists():
        raise SemanticResultsError(f"Summary file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SemanticResultsError("Summary JSON must contain an object.")
    return payload


def build_results_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build the paper-ready metrics table."""

    columns = [
        "strategy",
        "feature_setting",
        "model_name",
        "row_count",
        "feature_count",
        "uses_tfidf",
        "validation_MAE",
        "MAE",
        "RMSE",
        "R2",
        "within_100",
        "within_200",
    ]
    available = [column for column in columns if column in metrics.columns]
    output = metrics.loc[:, available].copy()
    return output.sort_values(["strategy", "feature_setting"], kind="mergesort")


def _metric_for_setting(group: pd.DataFrame, setting: str) -> pd.Series | None:
    """Return the single validation-selected row for one setting, if present."""

    rows = group.loc[group["feature_setting"].eq(setting)]
    if rows.empty:
        return None
    if len(rows) != 1:
        raise SemanticResultsError(
            f"Setting {setting!r} has {len(rows)} locked test rows; expected one."
        )
    return rows.iloc[0]


def compute_improvements(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute all requested MAE improvement comparisons."""

    rows: list[dict[str, object]] = []
    for strategy, group in metrics.groupby("strategy", sort=True):
        for baseline_setting, comparison_setting in IMPROVEMENT_COMPARISONS:
            baseline = _metric_for_setting(group, baseline_setting)
            comparison = _metric_for_setting(group, comparison_setting)
            if baseline is None or comparison is None:
                continue
            baseline_mae = float(baseline["MAE"])
            comparison_mae = float(comparison["MAE"])
            absolute = baseline_mae - comparison_mae
            rows.append(
                {
                    "strategy": strategy,
                    "baseline_setting": baseline_setting,
                    "comparison_setting": comparison_setting,
                    "baseline_MAE": round(baseline_mae, 6),
                    "comparison_MAE": round(comparison_mae, 6),
                    "absolute_MAE_improvement": round(absolute, 6),
                    "percent_MAE_improvement": round(
                        absolute / baseline_mae * 100.0 if baseline_mae else 0.0,
                        6,
                    ),
                }
            )
    output = pd.DataFrame(rows)
    if output.empty:
        raise SemanticResultsError("No requested improvement comparisons could be computed.")
    return output.sort_values(
        ["strategy", "baseline_setting", "comparison_setting"],
        kind="mergesort",
    ).reset_index(drop=True)


def _fmt_number(value: object, digits: int = 3) -> str:
    """Format numeric values for markdown tables."""

    if value is None or pd.isna(value):
        return ""
    number = float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Render a compact markdown table."""

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, separator]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(_fmt_number(value, 3))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _coverage_lines(summary: Mapping[str, object]) -> list[str]:
    """Build markdown bullets from summary coverage fields."""

    fields = [
        ("Input model-table rows", "input_model_table_rows"),
        ("Matched statement-text rows", "matched_statement_text_rows"),
        ("Matched statement-feature rows", "matched_statement_feature_rows"),
        ("Text available count", "text_available_count"),
        ("Text available rate", "text_available_rate"),
        ("TF-IDF max features", "tfidf_max_features"),
        ("TF-IDF ngram range", "tfidf_ngram_range"),
    ]
    lines: list[str] = []
    for label, key in fields:
        value = summary.get(key)
        if value is not None:
            lines.append(f"- {label}: `{value}`")
    return lines


def render_markdown_summary(
    metrics: pd.DataFrame,
    summary: Mapping[str, object],
    improvements: pd.DataFrame,
) -> str:
    """Render the paper-ready semantic results markdown summary."""

    lines = [
        "# v6 Semantic Statement Text Modeling Results",
        "",
        "## Research question",
        "",
        RESEARCH_QUESTION,
        "",
        "## Dataset and coverage",
        "",
        *_coverage_lines(summary),
        "",
        "## Metrics by split",
        "",
    ]
    metric_columns = [
        "feature_setting",
        "model_name",
        "validation_MAE",
        "MAE",
        "RMSE",
        "R2",
        "within_100",
        "within_200",
    ]
    for strategy, group in metrics.groupby("strategy", sort=True):
        lines.extend(
            [
                f"### {strategy}",
                "",
                _markdown_table(group.loc[:, metric_columns], metric_columns),
                "",
            ]
        )
    improvement_columns = [
        "strategy",
        "baseline_setting",
        "comparison_setting",
        "baseline_MAE",
        "comparison_MAE",
        "absolute_MAE_improvement",
        "percent_MAE_improvement",
    ]
    lines.extend(
        [
            "## Comparison summary",
            "",
            "The comparisons below focus on cold-start semantic text additions:",
            "",
            "- `metadata_only` vs `metadata_plus_tfidf`",
            "- `metadata_plus_text_light` vs `metadata_plus_text_light_plus_tfidf`",
            "- `tfidf_text_only` vs combined metadata/text-light/TF-IDF settings",
            "",
            _markdown_table(improvements.loc[:, improvement_columns], improvement_columns),
            "",
            "## Key findings",
            "",
        ]
    )
    lines.extend([f"{idx}. {finding}" for idx, finding in enumerate(KEY_FINDINGS, start=1)])
    lines.extend(
        [
            "",
            "## Conservative limitations",
            "",
        ]
    )
    lines.extend([f"- {limitation}" for limitation in CONSERVATIVE_LIMITATIONS])
    return "\n".join(lines) + "\n"


def build_key_findings_payload(
    metrics: pd.DataFrame,
    summary: Mapping[str, object],
    improvements: pd.DataFrame,
) -> dict[str, object]:
    """Build machine-readable key findings for downstream paper packaging."""

    selected_by_strategy: dict[str, dict[str, object]] = {}
    for strategy, group in metrics.groupby("strategy", sort=True):
        best = group.sort_values(
            ["validation_MAE", "feature_setting"],
            kind="mergesort",
        ).iloc[0]
        selected_by_strategy[str(strategy)] = {
            "feature_setting": str(best["feature_setting"]),
            "selection_split": "valid",
            "validation_MAE": float(best["validation_MAE"]),
            "report_split": "test",
            "test_MAE": float(best["MAE"]),
            "within_200": float(best["within_200"]),
        }
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "research_question": RESEARCH_QUESTION,
        "key_findings": list(KEY_FINDINGS),
        "conservative_limitations": list(CONSERVATIVE_LIMITATIONS),
        "coverage": {
            "input_model_table_rows": summary.get("input_model_table_rows"),
            "matched_statement_text_rows": summary.get("matched_statement_text_rows"),
            "matched_statement_feature_rows": summary.get(
                "matched_statement_feature_rows"
            ),
            "text_available_count": summary.get("text_available_count"),
            "text_available_rate": summary.get("text_available_rate"),
        },
        "validation_selected_setting_test_report": selected_by_strategy,
        "improvements": improvements.to_dict(orient="records"),
    }


def write_json(path: Path, payload: object) -> None:
    """Write pretty UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def plot_mae_comparison(metrics: pd.DataFrame, path: Path) -> None:
    """Create a paper-friendly MAE comparison by setting and split."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pivot = metrics.pivot(index="feature_setting", columns="strategy", values="MAE")
    existing_order = [setting for setting in SETTING_ORDER if setting in pivot.index]
    pivot = pivot.loc[existing_order]
    ax = pivot.plot(kind="bar", figsize=(13, 6))
    ax.set_title("v6 semantic TF-IDF MAE by feature setting")
    ax.set_xlabel("Feature setting")
    ax.set_ylabel("Test MAE")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Split strategy")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_improvement_comparison(improvements: pd.DataFrame, path: Path) -> None:
    """Create a paper-friendly improvement comparison figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure_data = improvements.copy()
    figure_data["comparison"] = (
        figure_data["baseline_setting"] + "\n→ " + figure_data["comparison_setting"]
    )
    pivot = figure_data.pivot(
        index="comparison",
        columns="strategy",
        values="absolute_MAE_improvement",
    )
    ax = pivot.plot(kind="bar", figsize=(13, 6))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("v6 semantic TF-IDF MAE improvements")
    ax.set_xlabel("Comparison")
    ax.set_ylabel("Absolute MAE improvement")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Split strategy")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_semantic_results_summary(
    *,
    metrics_path: Path,
    summary_path: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Path]:
    """Generate all v6 semantic results summary artifacts."""

    logger = configure_logger(log_path)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_dir = output_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)

        metrics = load_metrics_csv(metrics_path)
        summary = load_summary_json(summary_path)
        results_table = build_results_table(metrics)
        improvements = compute_improvements(metrics)
        markdown = render_markdown_summary(metrics, summary, improvements)
        key_findings = build_key_findings_payload(metrics, summary, improvements)

        paths = {
            "markdown": output_dir / "v6_semantic_results_summary.md",
            "results_table": output_dir / "v6_semantic_results_table.csv",
            "improvements": output_dir / "v6_semantic_improvements.csv",
            "key_findings": output_dir / "v6_semantic_key_findings.json",
            "mae_figure": figure_dir / "v6_semantic_mae_comparison.png",
            "improvement_figure": figure_dir / "v6_semantic_improvement_comparison.png",
        }
        paths["markdown"].write_text(markdown, encoding="utf-8")
        results_table.to_csv(paths["results_table"], index=False)
        improvements.to_csv(paths["improvements"], index=False)
        write_json(paths["key_findings"], key_findings)
        plot_mae_comparison(metrics, paths["mae_figure"])
        plot_improvement_comparison(improvements, paths["improvement_figure"])
        logger.info(
            "Completed semantic results summary",
            extra={
                "event": "semantic_results_completed",
                "details": {
                    "metrics_rows": len(metrics),
                    "improvement_rows": len(improvements),
                    "output_dir": output_dir.as_posix(),
                },
            },
        )
        return paths
    except Exception:
        logger.exception(
            "Semantic results summary failed",
            extra={"event": "semantic_results_failed"},
        )
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Summarize v6 semantic TF-IDF experiment outputs."
    )
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the semantic results summary CLI."""

    args = _build_argument_parser().parse_args(argv)
    try:
        paths = run_semantic_results_summary(
            metrics_path=args.metrics_path,
            summary_path=args.summary_path,
            output_dir=args.output_dir,
            log_path=args.log_path,
        )
    except (SemanticResultsError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote semantic results summary: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
