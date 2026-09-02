"""Verify the committed locked-backtest evidence without modifying it."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from cf_diff import historical_statement_backtest as backtest


DEFAULT_RESULT_DIR = Path("outputs/historical_statement_backtest/test")
DEFAULT_RESULT_MANIFEST = DEFAULT_RESULT_DIR / "result_manifest.sha256"


class VerificationError(RuntimeError):
    """Raised when committed evidence cannot be verified or reproduced."""


def _parse_hash_manifest(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        raise VerificationError(f"Hash manifest does not exist: {path}")

    entries: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="ascii").splitlines(), start=1
    ):
        parts = raw_line.split()
        if len(parts) != 2:
            raise VerificationError(
                f"Invalid hash manifest line {line_number}: expected hash and filename."
            )
        expected_hash, filename = parts
        if (
            len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
            or Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
        ):
            raise VerificationError(f"Invalid hash manifest line {line_number}.")
        entries.append((expected_hash, filename))

    if not entries:
        raise VerificationError("Hash manifest is empty.")
    if len({filename for _digest, filename in entries}) != len(entries):
        raise VerificationError("Hash manifest contains duplicate filenames.")
    return entries


def verify_hash_manifest(manifest_path: Path, artifact_dir: Path) -> int:
    """Verify every file named by a strict, flat SHA-256 manifest."""

    entries = _parse_hash_manifest(manifest_path)
    for expected_hash, filename in entries:
        artifact_path = artifact_dir / filename
        if not artifact_path.is_file():
            raise VerificationError(f"Committed result artifact is missing: {filename}")
        actual_hash = backtest.sha256_file(artifact_path)
        if actual_hash != expected_hash:
            raise VerificationError(f"Committed result hash mismatch: {filename}")
    return len(entries)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"Cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"JSON artifact is not an object: {path}")
    return value


def _assert_values_close(expected: Any, actual: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise VerificationError(f"Recomputed JSON structure differs: {label}")
        for key in expected:
            _assert_values_close(expected[key], actual[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise VerificationError(f"Recomputed JSON list differs: {label}")
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            _assert_values_close(expected_item, actual_item, f"{label}[{index}]")
        return
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        if not math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-9):
            raise VerificationError(
                f"Recomputed numeric value differs at {label}: "
                f"expected {expected}, got {actual}"
            )
        return
    if expected != actual:
        raise VerificationError(
            f"Recomputed value differs at {label}: expected {expected!r}, got {actual!r}"
        )


def _compare_json(expected_path: Path, actual_path: Path) -> None:
    _assert_values_close(
        _load_json(expected_path),
        _load_json(actual_path),
        expected_path.name,
    )


def _compare_csv(expected_path: Path, actual_path: Path) -> None:
    expected = pd.read_csv(expected_path)
    actual = pd.read_csv(actual_path)
    if list(expected.columns) != list(actual.columns) or len(expected) != len(actual):
        raise VerificationError(f"Recomputed CSV shape differs: {expected_path.name}")

    for column in expected.columns:
        expected_column = expected[column]
        actual_column = actual[column]
        if pd.api.types.is_numeric_dtype(expected_column):
            if not pd.api.types.is_numeric_dtype(actual_column) or not np.allclose(
                expected_column.to_numpy(dtype=float),
                actual_column.to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-9,
                equal_nan=True,
            ):
                raise VerificationError(
                    f"Recomputed numeric CSV column differs: "
                    f"{expected_path.name}:{column}"
                )
        elif not expected_column.fillna("").astype(str).equals(
            actual_column.fillna("").astype(str)
        ):
            raise VerificationError(
                f"Recomputed CSV column differs: {expected_path.name}:{column}"
            )


def verify_locked_backtest(
    *,
    config_path: Path = backtest.DEFAULT_CONFIG_PATH,
    selection_dir: Path = backtest.DEFAULT_SELECTION_DIR,
    result_dir: Path = DEFAULT_RESULT_DIR,
    result_manifest_path: Path = DEFAULT_RESULT_MANIFEST,
) -> dict[str, int]:
    """Verify hashes, then reproduce the reported result in a temporary directory."""

    result_hash_count = verify_hash_manifest(result_manifest_path, result_dir)

    with tempfile.TemporaryDirectory(prefix="cf_locked_backtest_verify_") as temp_dir:
        generated_dir = Path(temp_dir) / "test"
        generated = backtest.run_test(
            config_path=config_path,
            selection_dir=selection_dir,
            output_dir=generated_dir,
        )
        _compare_json(result_dir / backtest.TEST_METRICS_FILENAME, generated["metrics"])
        _compare_json(result_dir / backtest.BOOTSTRAP_FILENAME, generated["bootstrap"])
        _compare_csv(
            result_dir / backtest.TEST_PREDICTIONS_FILENAME,
            generated["predictions"],
        )
        _compare_csv(result_dir / backtest.TOP10_FILENAME, generated["top10"])

    return {
        "verified_result_hashes": result_hash_count,
        "recomputed_test_artifacts": 4,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the committed locked-backtest hashes and reproduce its reported "
            "test artifacts in a temporary directory."
        )
    )
    parser.add_argument("--config", type=Path, default=backtest.DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--selection-dir", type=Path, default=backtest.DEFAULT_SELECTION_DIR
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--result-manifest", type=Path, default=DEFAULT_RESULT_MANIFEST
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = verify_locked_backtest(
            config_path=args.config,
            selection_dir=args.selection_dir,
            result_dir=args.result_dir,
            result_manifest_path=args.result_manifest,
        )
    except (VerificationError, backtest.HistoricalBacktestError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        "Verified "
        f"{report['verified_result_hashes']} committed result hashes and reproduced "
        f"{report['recomputed_test_artifacts']} locked test artifacts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
