"""Select models on validation data and report untouched test metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_METRIC_COLUMNS: tuple[str, ...] = (
    "MAE",
    "RMSE",
    "R2",
    "within_100",
    "within_200",
)


class ModelSelectionError(RuntimeError):
    """Raised when validation selection cannot be audited safely."""


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ModelSelectionError(f"Metrics table lacks required columns: {missing}")


def build_validation_ranked_report(
    metrics: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    candidate_columns: Sequence[str],
    metric_columns: Sequence[str] = DEFAULT_METRIC_COLUMNS,
    selection_metric: str = "MAE",
    selection_split: str = "valid",
    report_split: str = "test",
) -> pd.DataFrame:
    """Rank candidates on validation MAE and attach their test metrics.

    The returned rows retain the report-split metric names (for example
    ``MAE``) and prefix selection-split metrics with ``validation_``. This
    keeps the final test report readable while making the selection evidence
    explicit and machine-auditable.
    """

    groups = list(group_columns)
    candidates = list(candidate_columns)
    metric_names = list(metric_columns)
    keys = [*groups, *candidates]
    if not groups:
        raise ModelSelectionError("At least one grouping column is required.")
    if not candidates:
        raise ModelSelectionError("At least one candidate column is required.")
    if selection_metric not in metric_names:
        raise ModelSelectionError(
            f"Selection metric {selection_metric!r} is not in metric_columns."
        )
    _require_columns(metrics, [*keys, "split_name", *metric_names])

    selection = metrics.loc[metrics["split_name"].eq(selection_split)].copy()
    report = metrics.loc[metrics["split_name"].eq(report_split)].copy()
    if selection.empty or report.empty:
        raise ModelSelectionError(
            f"Both {selection_split!r} and {report_split!r} rows are required."
        )
    for split_name, frame in ((selection_split, selection), (report_split, report)):
        duplicated = frame.duplicated(keys, keep=False)
        if duplicated.any():
            examples = frame.loc[duplicated, keys].head(3).to_dict(orient="records")
            raise ModelSelectionError(
                f"Split {split_name!r} has duplicate candidate rows: {examples}"
            )
        numeric_metrics = frame.loc[:, metric_names].apply(
            pd.to_numeric,
            errors="coerce",
        )
        finite_mask = np.isfinite(numeric_metrics.to_numpy(dtype=float))
        if not finite_mask.all():
            row_index, column_index = np.argwhere(~finite_mask)[0]
            bad_key = frame.iloc[int(row_index)].loc[keys].to_dict()
            bad_metric = metric_names[int(column_index)]
            raise ModelSelectionError(
                f"Split {split_name!r} has a non-finite {bad_metric} for "
                f"candidate {bad_key}."
            )

    selection_keys = set(map(tuple, selection.loc[:, keys].itertuples(index=False)))
    report_keys = set(map(tuple, report.loc[:, keys].itertuples(index=False)))
    if selection_keys != report_keys:
        missing_report = sorted(selection_keys - report_keys, key=str)[:3]
        missing_selection = sorted(report_keys - selection_keys, key=str)[:3]
        raise ModelSelectionError(
            "Validation and test candidates differ; "
            f"missing_test={missing_report}, missing_validation={missing_selection}."
        )

    validation_columns = selection.loc[:, [*keys, *metric_names]].rename(
        columns={column: f"validation_{column}" for column in metric_names}
    )
    output = report.merge(
        validation_columns,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    output["selection_split"] = selection_split
    output["report_split"] = report_split
    output["selection_metric"] = selection_metric
    validation_metric = f"validation_{selection_metric}"
    output = output.sort_values(
        [*groups, validation_metric, *candidates],
        kind="mergesort",
    ).reset_index(drop=True)
    output["selection_rank"] = (
        output.groupby(groups, sort=False).cumcount().add(1).astype(int)
    )
    return output


def select_rank_one(report: pd.DataFrame) -> pd.DataFrame:
    """Return validation-selected rows from a ranked validation/test report."""

    _require_columns(report, ("selection_rank",))
    selected = report.loc[report["selection_rank"].eq(1)].copy()
    if selected.empty:
        raise ModelSelectionError("Ranked report contains no selected candidates.")
    return selected.reset_index(drop=True)
