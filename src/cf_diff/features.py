"""Build lightweight modeling features from preprocessed Codeforces data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pandas as pd

from cf_diff import RANDOM_SEED

DEFAULT_CONFIG_PATH: Final[Path] = Path("configs/experiment.yaml")
DEFAULT_LOG_PATH: Final[Path] = Path("outputs/logs/features.log")
IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (
    "contest_id",
    "index",
    "name",
    "rating",
    "start_time_seconds",
)


class FeatureError(RuntimeError):
    """Raised when feature extraction cannot proceed safely."""


@dataclass(frozen=True)
class SplitRatios:
    """Store train, validation, and test proportions."""

    train: float = 0.7
    valid: float = 0.15
    test: float = 0.15


@dataclass(frozen=True)
class ExperimentConfig:
    """Store feature and split settings with reproducible defaults."""

    random_seed: int = RANDOM_SEED
    grouped_split: SplitRatios = SplitRatios()
    forward_time_split: SplitRatios = SplitRatios()
    include_points: bool = True
    include_tags: bool = True
    min_tag_frequency: int = 5


class JsonLogFormatter(logging.Formatter):
    """Format audit logs as JSON Lines."""

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
    """Create the dedicated structured feature-pipeline logger."""
    resolved_path = log_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cf_diff.features")
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
    """Flush and close all handlers on a dedicated logger."""
    for handler in tuple(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _parse_scalar(value: str) -> object:
    """Parse one scalar from the supported YAML subset."""
    stripped = value.strip()
    if not stripped:
        return {}
    if (
        len(stripped) >= 2
        and stripped[0] == stripped[-1]
        and stripped[0] in {"'", '"'}
    ):
        return stripped[1:-1]
    lowered = stripped.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return int(stripped)
    except ValueError:
        try:
            return float(stripped)
        except ValueError:
            return stripped


def parse_simple_yaml(text: str) -> dict[str, object]:
    """Parse a strict YAML subset containing nested scalar mappings only."""
    root: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(-1, root)]
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if "\t" in raw_line:
            raise FeatureError(
                f"YAML line {line_number} uses a tab; use spaces for indentation."
            )
        content = raw_line.split("#", maxsplit=1)[0].rstrip()
        if not content.strip():
            continue
        indentation = len(content) - len(content.lstrip(" "))
        stripped = content.strip()
        if ":" not in stripped:
            raise FeatureError(
                f"YAML line {line_number} must contain a key-value mapping."
            )
        key, raw_value = stripped.split(":", maxsplit=1)
        key = key.strip()
        if not key:
            raise FeatureError(f"YAML line {line_number} has an empty key.")

        while stack[-1][0] >= indentation:
            stack.pop()
        parent = stack[-1][1]
        if key in parent:
            raise FeatureError(
                f"YAML line {line_number} repeats key {key!r}."
            )
        parsed = _parse_scalar(raw_value)
        parent[key] = parsed
        if isinstance(parsed, dict):
            stack.append((indentation, parsed))
    return root


def _mapping(value: object, key: str) -> Mapping[str, object]:
    """Require a mapping-valued config section."""
    if not isinstance(value, dict):
        raise FeatureError(f"Config key {key!r} must be a mapping.")
    return value


def _read_ratios(
    config: Mapping[str, object],
    key: str,
    default: SplitRatios,
) -> SplitRatios:
    """Read and validate one split-ratio mapping."""
    raw = config.get(key)
    if raw is None:
        return default
    values = _mapping(raw, key)
    ratios = SplitRatios(
        train=float(values.get("train", default.train)),
        valid=float(values.get("valid", default.valid)),
        test=float(values.get("test", default.test)),
    )
    validate_split_ratios(ratios, key)
    return ratios


def validate_split_ratios(ratios: SplitRatios, name: str) -> None:
    """Require positive ratios summing to one."""
    values = (ratios.train, ratios.valid, ratios.test)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise FeatureError(f"{name} ratios must all be finite and positive.")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise FeatureError(f"{name} ratios must sum to 1.0.")


def load_experiment_config(
    path: Path = DEFAULT_CONFIG_PATH,
) -> ExperimentConfig:
    """Load the concise YAML config, using defaults when the file is absent."""
    defaults = ExperimentConfig()
    if not path.exists():
        return defaults

    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise FeatureError("Experiment config must contain a mapping.")
        raw_config: Mapping[str, object] = parsed
    else:
        raw_config = parse_simple_yaml(text)

    features_section = raw_config.get("features", {})
    if features_section is None:
        features_section = {}
    features = _mapping(features_section, "features")
    include_points = features.get(
        "include_points",
        raw_config.get("include_points", defaults.include_points),
    )
    include_tags = features.get(
        "include_tags",
        raw_config.get("include_tags", defaults.include_tags),
    )
    min_tag_frequency = features.get(
        "min_tag_frequency",
        raw_config.get(
            "min_tag_frequency",
            defaults.min_tag_frequency,
        ),
    )
    config = ExperimentConfig(
        random_seed=int(
            raw_config.get("random_seed", defaults.random_seed)
        ),
        grouped_split=_read_ratios(
            raw_config,
            "grouped_split",
            defaults.grouped_split,
        ),
        forward_time_split=_read_ratios(
            raw_config,
            "forward_time_split",
            defaults.forward_time_split,
        ),
        include_points=bool(include_points),
        include_tags=bool(include_tags),
        min_tag_frequency=int(min_tag_frequency),
    )
    if config.min_tag_frequency < 1:
        raise FeatureError("min_tag_frequency must be at least 1.")
    return config


def _normalize_tags(value: object) -> list[str]:
    """Normalize a Parquet list-like value to unique sorted strings."""
    if value is None:
        return []
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, (list, tuple, set)):
        return []
    return sorted({str(tag) for tag in value if str(tag)})


def _tag_column_name(tag: str, used: set[str]) -> str:
    """Create a stable collision-safe one-hot column name."""
    normalized = re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")
    normalized = normalized or "unnamed"
    candidate = f"tag__{normalized}"
    if candidate in used:
        digest = hashlib.sha256(tag.encode("utf-8")).hexdigest()[:8]
        candidate = f"{candidate}__{digest}"
    used.add(candidate)
    return candidate


def _required_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a clear error when required upstream columns are absent."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FeatureError(f"Input table lacks required columns: {missing}")


def build_model_table(
    source: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create deterministic identifiers, target, and lightweight features."""
    _required_columns(source, IDENTIFIER_COLUMNS)
    frame = source.copy()
    for optional in ("points", "tags", "solved_count"):
        if optional not in frame.columns:
            frame[optional] = pd.NA

    for column in ("contest_id", "rating", "start_time_seconds"):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).astype("Int64")
    if frame["contest_id"].isna().any() or frame["rating"].isna().any():
        raise FeatureError(
            "Rated modeling input must have non-null contest_id and rating."
        )
    frame["index"] = frame["index"].astype("string")
    frame["name"] = frame["name"].astype("string")

    index_letter = frame["index"].str.extract(
        r"^([A-Za-z]+)",
        expand=False,
    ).str.upper()
    index_number = frame["index"].str.extract(
        r"(\d+)$",
        expand=False,
    )
    frame["index_letter"] = index_letter.fillna("").astype("string")
    frame["index_number"] = pd.to_numeric(
        index_number,
        errors="coerce",
    ).fillna(0).astype("Int64")
    rank_mapping = {
        letter: rank
        for rank, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ", start=1)
    }
    frame["index_rank"] = (
        index_letter.str[0].map(rank_mapping).fillna(0).astype("Int64")
    )

    raw_points = pd.to_numeric(frame["points"], errors="coerce")
    raw_solved_count = pd.to_numeric(
        frame["solved_count"],
        errors="coerce",
    )
    missing_before_imputation = {
        "points": int(raw_points.isna().sum()),
        "solved_count": int(raw_solved_count.isna().sum()),
        "start_time_seconds": int(frame["start_time_seconds"].isna().sum()),
    }

    # Missing points mean "not supplied"; preserve that fact with has_points,
    # then use zero so downstream numeric estimators do not receive NaN.
    frame["has_points"] = raw_points.notna().astype("Int8")
    frame["points"] = raw_points.fillna(0.0).astype("Float64")

    # Missing solved counts are explicitly represented by a binary indicator.
    # The numeric value is zero-imputed before applying log1p.
    frame["solved_count_missing"] = raw_solved_count.isna().astype("Int8")
    frame["solved_count"] = (
        raw_solved_count.fillna(0).clip(lower=0).astype("Int64")
    )
    frame["log_solved_count"] = frame["solved_count"].map(
        lambda value: math.log1p(int(value))
    )

    normalized_tags = frame["tags"].map(_normalize_tags)
    frame["tag_count"] = normalized_tags.map(len).astype("Int64")
    tag_frequency = Counter(
        tag for tags in normalized_tags for tag in set(tags)
    )
    selected_tags = (
        sorted(
            tag
            for tag, frequency in tag_frequency.items()
            if frequency >= config.min_tag_frequency
        )
        if config.include_tags
        else []
    )
    used_columns: set[str] = set()
    tag_feature_map = {
        tag: _tag_column_name(tag, used_columns) for tag in selected_tags
    }
    for tag, column in tag_feature_map.items():
        frame[column] = normalized_tags.map(
            lambda tags, current=tag: int(current in tags)
        ).astype("Int8")

    structured_features = [
        "index_letter",
        "index_number",
        "index_rank",
        "tag_count",
        "solved_count",
        "solved_count_missing",
        "log_solved_count",
    ]
    if config.include_points:
        structured_features.extend(["has_points", "points"])
    feature_columns = [
        *structured_features,
        *tag_feature_map.values(),
    ]
    output_columns = list(IDENTIFIER_COLUMNS)
    for column in (
        "index_letter",
        "index_number",
        "index_rank",
        "has_points",
        "points",
        "tag_count",
        "solved_count",
        "solved_count_missing",
        "log_solved_count",
        *tag_feature_map.values(),
    ):
        if column not in output_columns:
            output_columns.append(column)
    model_table = frame.loc[:, output_columns]
    model_table = model_table.sort_values(
        ["start_time_seconds", "contest_id", "index"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    metadata: dict[str, object] = {
        "identifier_columns": list(IDENTIFIER_COLUMNS),
        "target_column": "rating",
        "feature_columns": feature_columns,
        "tag_feature_map": tag_feature_map,
        "missing_value_strategy": {
            "points": "zero-imputed with has_points indicator",
            "solved_count": (
                "zero-imputed with solved_count_missing indicator"
            ),
            "index_number": "missing numeric suffix encoded as 0",
            "index_rank": "missing alphabetic prefix encoded as 0",
        },
        "missing_before_imputation": missing_before_imputation,
    }
    return model_table, metadata


def write_json(path: Path, payload: object) -> None:
    """Write deterministic pretty UTF-8 JSON."""
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_features(
    input_path: Path,
    output_dir: Path,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
) -> dict[str, Path]:
    """Read processed problems and write the modeling feature artifacts."""
    logger = configure_logger(log_path)
    try:
        config = load_experiment_config(config_path)
        source = pd.read_parquet(input_path, engine="pyarrow")
        model_table, metadata = build_model_table(source, config)
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "model_table": output_dir / "model_table.parquet",
            "feature_columns": output_dir / "feature_columns.json",
            "feature_summary": output_dir / "feature_summary.json",
        }
        model_table.to_parquet(
            paths["model_table"],
            engine="pyarrow",
            index=False,
        )
        feature_columns_payload = {
            **metadata,
            "config": asdict(config),
        }
        write_json(paths["feature_columns"], feature_columns_payload)
        tag_columns = [
            column
            for column in model_table.columns
            if column.startswith("tag__")
        ]
        summary = {
            "input_path": input_path.resolve().as_posix(),
            "row_count": len(model_table),
            "column_count": len(model_table.columns),
            "feature_count": len(metadata["feature_columns"]),
            "tag_feature_count": len(tag_columns),
            "rating_min": int(model_table["rating"].min()),
            "rating_max": int(model_table["rating"].max()),
            "missing_before_imputation": metadata[
                "missing_before_imputation"
            ],
            "random_seed": config.random_seed,
        }
        write_json(paths["feature_summary"], summary)
        logger.info(
            "Completed Codeforces feature extraction",
            extra={
                "event": "features_completed",
                "details": summary,
            },
        )
        return paths
    except Exception:
        logger.exception(
            "Codeforces feature extraction failed",
            extra={"event": "features_failed"},
        )
        raise
    finally:
        close_logger(logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the feature command-line parser."""
    parser = argparse.ArgumentParser(
        description="Build Codeforces modeling features."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the feature extraction CLI."""
    args = _build_argument_parser().parse_args(argv)
    try:
        paths = generate_features(
            args.input_path,
            args.output_dir,
            config_path=args.config,
            log_path=args.log_path,
        )
    except (FeatureError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote model table: {paths['model_table']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
