"""Create immutable raw Codeforces API snapshots with provenance metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cf_diff import RANDOM_SEED
from cf_diff.fetch_api import (
    API_BASE_URL,
    CodeforcesAPIError,
    fetch_contest_list,
    fetch_problemset_problems,
)

PROBLEMSET_DATASET: Final[str] = "problemset_problems"
CONTEST_DATASET: Final[str] = "contest_list"


class SnapshotError(RuntimeError):
    """Raised when a raw snapshot set cannot be created safely."""


def compute_sha256(data: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of the supplied bytes."""
    return hashlib.sha256(data).hexdigest()


def write_json(
    path: Path,
    payload: object,
    *,
    exclusive: bool = False,
) -> bytes:
    """Serialize and write pretty UTF-8 JSON, returning the exact file bytes."""
    data = (
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    mode = "xb" if exclusive else "wb"
    with path.open(mode) as output_file:
        output_file.write(data)
    return data


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def generate_utc_timestamp_token(timestamp: datetime | None = None) -> str:
    """Return a compact UTC timestamp token suitable for snapshot filenames."""
    value = timestamp if timestamp is not None else utc_now()
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware.")
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _format_utc_iso(timestamp: datetime) -> str:
    """Return a timezone-aware UTC ISO-8601 timestamp."""
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware.")
    return timestamp.astimezone(timezone.utc).isoformat(timespec="seconds")


def _relative_path(path: Path, output_root: Path) -> str:
    """Return a portable path relative to the snapshot output root."""
    return path.relative_to(output_root).as_posix()


def _build_provenance(
    *,
    dataset_name: str,
    cf_method: str,
    request_params: Mapping[str, object],
    fetched_at_utc: str,
    snapshot_path: Path,
    output_root: Path,
    snapshot_bytes: bytes,
    api_payload: Mapping[str, object],
    seed: int,
) -> dict[str, object]:
    """Build machine-readable provenance for one raw snapshot."""
    api_status = api_payload.get("status")
    if not isinstance(api_status, str):
        raise SnapshotError(
            f"{dataset_name} payload lacks a string top-level API status."
        )

    return {
        "dataset_name": dataset_name,
        "cf_method": cf_method,
        "request_url": f"{API_BASE_URL}/{cf_method}",
        "request_params": dict(request_params),
        "fetched_at_utc": fetched_at_utc,
        "saved_path": _relative_path(snapshot_path, output_root),
        "sha256": compute_sha256(snapshot_bytes),
        "byte_count": len(snapshot_bytes),
        "api_status": api_status,
        "seed": seed,
        "python_version": sys.version,
    }


def create_snapshot_set(
    output_root: Path,
    *,
    lang: str = "en",
    seed: int = RANDOM_SEED,
) -> dict[str, object]:
    """Fetch and persist one immutable pair of Codeforces raw snapshots."""
    problemset_payload = fetch_problemset_problems(lang=lang)
    contest_payload = fetch_contest_list(gym=False, lang=lang)

    run_timestamp = utc_now()
    timestamp_token = generate_utc_timestamp_token(run_timestamp)
    fetched_at_utc = _format_utc_iso(run_timestamp)

    output_root = output_root.resolve()
    snapshots_dir = output_root / "snapshots"
    provenance_dir = output_root / "provenance"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    snapshot_paths = {
        PROBLEMSET_DATASET: (
            snapshots_dir / f"{PROBLEMSET_DATASET}_{timestamp_token}.json"
        ),
        CONTEST_DATASET: snapshots_dir / f"{CONTEST_DATASET}_{timestamp_token}.json",
    }
    provenance_paths = {
        PROBLEMSET_DATASET: (
            provenance_dir
            / f"{PROBLEMSET_DATASET}_{timestamp_token}.meta.json"
        ),
        CONTEST_DATASET: (
            provenance_dir / f"{CONTEST_DATASET}_{timestamp_token}.meta.json"
        ),
    }

    timestamped_paths = tuple(snapshot_paths.values()) + tuple(
        provenance_paths.values()
    )
    existing_paths = [path for path in timestamped_paths if path.exists()]
    if existing_paths:
        joined_paths = ", ".join(str(path) for path in existing_paths)
        raise SnapshotError(
            "Refusing to overwrite existing timestamped snapshot artifacts: "
            f"{joined_paths}"
        )

    problemset_bytes = write_json(
        snapshot_paths[PROBLEMSET_DATASET],
        problemset_payload,
        exclusive=True,
    )
    contest_bytes = write_json(
        snapshot_paths[CONTEST_DATASET],
        contest_payload,
        exclusive=True,
    )

    provenance_by_dataset = {
        PROBLEMSET_DATASET: _build_provenance(
            dataset_name=PROBLEMSET_DATASET,
            cf_method="problemset.problems",
            request_params={"lang": lang},
            fetched_at_utc=fetched_at_utc,
            snapshot_path=snapshot_paths[PROBLEMSET_DATASET],
            output_root=output_root,
            snapshot_bytes=problemset_bytes,
            api_payload=problemset_payload,
            seed=seed,
        ),
        CONTEST_DATASET: _build_provenance(
            dataset_name=CONTEST_DATASET,
            cf_method="contest.list",
            request_params={"gym": "false", "lang": lang},
            fetched_at_utc=fetched_at_utc,
            snapshot_path=snapshot_paths[CONTEST_DATASET],
            output_root=output_root,
            snapshot_bytes=contest_bytes,
            api_payload=contest_payload,
            seed=seed,
        ),
    }

    for dataset_name, provenance in provenance_by_dataset.items():
        write_json(
            provenance_paths[dataset_name],
            provenance,
            exclusive=True,
        )

    manifest: dict[str, object] = {
        "created_at_utc": fetched_at_utc,
        "seed": seed,
        "entries": {
            dataset_name: {
                "snapshot_path": _relative_path(
                    snapshot_paths[dataset_name],
                    output_root,
                ),
                "provenance_path": _relative_path(
                    provenance_paths[dataset_name],
                    output_root,
                ),
            }
            for dataset_name in (PROBLEMSET_DATASET, CONTEST_DATASET)
        },
    }
    write_json(output_root / "manifest_latest.json", manifest)
    return manifest


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Create immutable raw Codeforces API snapshots."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory for snapshots, provenance, and the latest manifest.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Codeforces API language parameter (default: en).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Project random seed recorded in provenance (default: {RANDOM_SEED}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the raw Codeforces snapshot CLI."""
    args = _build_argument_parser().parse_args(argv)
    random.seed(args.seed)

    try:
        manifest = create_snapshot_set(
            args.output_root,
            lang=args.lang,
            seed=args.seed,
        )
    except (CodeforcesAPIError, SnapshotError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    entries = manifest["entries"]
    if not isinstance(entries, dict):
        raise AssertionError("Generated manifest entries must be a dictionary.")
    print(
        "Created Codeforces snapshot set: "
        f"{entries[PROBLEMSET_DATASET]['snapshot_path']}, "
        f"{entries[CONTEST_DATASET]['snapshot_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
