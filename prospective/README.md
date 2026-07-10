# Prospective blind evaluation

This directory holds the frozen model bundle, its manifest, and future
append-only ledgers for protocol `cf-difficulty-prospective-v1`.

The cohort is active only for contests starting on or after
`2026-07-12T00:00:00Z`. An empty ledger before the first eligible contest is the
correct initial state; historical rows must never be inserted as if they were
prospective.

See [`docs/RESEARCH_PROTOCOL_V1.md`](../docs/RESEARCH_PROTOCOL_V1.md) for the
frozen design and [`ledger/README.md`](ledger/README.md) for the operational
commands.

Tracked prospective artifacts use these locations:

- `model_bundle_v1.json`: transparent frozen preprocessing and coefficients;
- `model_freeze_manifest_v1.json`: hashes, source commit, cutoff, and training
  provenance;
- `inputs/`: sanitized T0 feature files with no ratings or behavioral fields;
- `predictions/`: exact model outputs before reveal;
- `ledger/predictions.jsonl`: public prediction commitments;
- `ledger/reveals.jsonl`: later official outcomes.

Do not recreate or replace the v1 model bundle after the first prediction.
