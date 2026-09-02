"""Tests for the read-only locked-result verifier."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cf_diff import verify_locked_backtest as verifier


def test_hash_manifest_verifies_flat_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_bytes(b'{"value": 1}\n')
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "result_manifest.sha256"
    manifest.write_text(f"{digest}  {artifact.name}\n", encoding="ascii", newline="\n")

    assert verifier.verify_hash_manifest(manifest, tmp_path) == 1


@pytest.mark.parametrize("unsafe_name", ["../outside.json", "..\\outside.json"])
def test_hash_manifest_rejects_path_escape(tmp_path: Path, unsafe_name: str) -> None:
    digest = "0" * 64
    manifest = tmp_path / "result_manifest.sha256"
    manifest.write_text(f"{digest}  {unsafe_name}\n", encoding="ascii", newline="\n")

    with pytest.raises(verifier.VerificationError, match="Invalid hash manifest"):
        verifier.verify_hash_manifest(manifest, tmp_path)


def test_hash_manifest_detects_changed_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_text("changed", encoding="utf-8")
    manifest = tmp_path / "result_manifest.sha256"
    manifest.write_text(
        f"{'0' * 64}  {artifact.name}\n", encoding="ascii", newline="\n"
    )

    with pytest.raises(verifier.VerificationError, match="hash mismatch"):
        verifier.verify_hash_manifest(manifest, tmp_path)
