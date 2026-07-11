# Prospective v2 operations runbook (draft)

This runbook describes the guarded workflow proposed for
`cf-difficulty-prospective-v2`. It is **not an instruction to enroll contests
now**. The public protocol remains `draft`; production capture, commitment,
snapshot, and analysis commands intentionally reject it.

## Activation gates

Before changing the protocol to `frozen`:

1. merge the historical evaluation correction and governance changes in their
   dependency order;
2. retarget each stacked pull request to `main` after its prerequisite merges;
3. run the entire test suite from a clean, current `main` checkout on Linux and
   the operating machine;
4. run the synthetic end-to-end fixture, including deliberate tamper, late-run,
   missing-rating, non-programming, absent-key, and 30/200 threshold cases;
5. independently review the protocol fields, source hashes, witness workflow,
   fixed window times, and bootstrap golden values;
6. confirm that the planned start still leaves a review buffer; move it forward
   before freezing if it does not.

The real freeze remains a separate two-commit operation: first freeze the
protocol from clean `main`, then generate and commit the model bundle and freeze
manifest from that immutable protocol commit. Never backdate either action.

## Per-contest T0 commitment

Capture every visible index directly from public statement pages:

```powershell
python -m cf_diff.prospective_input `
  --protocol configs/prospective_protocol_v2.json `
  --contest-id 3000 `
  --indices A B C D `
  --contest-start-utc 2026-08-15T12:00:00Z `
  --output prospective/inputs/3000_t0_features.csv `
  --sidecar prospective/inputs/3000_t0_features.capture.json `
  --raw-dir prospective/raw/3000
```

Run the frozen primary and comparator models:

```powershell
python -m cf_diff.prospective_model predict `
  --protocol configs/prospective_protocol_v2.json `
  --model prospective/model_bundle_v2.json `
  --manifest prospective/model_freeze_manifest_v2.json `
  --input prospective/inputs/3000_t0_features.csv `
  --capture-sidecar prospective/inputs/3000_t0_features.capture.json `
  --contest-start-utc 2026-08-15T12:00:00Z `
  --output prospective/predictions/3000_predictions.csv
```

Append the commitment using explicit, hash-bound artifacts:

```powershell
python -m cf_diff.prospective_ledger prediction `
  --protocol configs/prospective_protocol_v2.json `
  --model prospective/model_bundle_v2.json `
  --manifest prospective/model_freeze_manifest_v2.json `
  --input prospective/inputs/3000_t0_features.csv `
  --capture-sidecar prospective/inputs/3000_t0_features.capture.json `
  --predictions prospective/predictions/3000_predictions.csv
```

The sanitized input, capture sidecar, prediction, and commitment event must
reach `main` before the T0 deadline. The dedicated push workflow must run; do
not use a skip directive. A local clock or Git commit date is not evidence.

After run attempt 1 finishes successfully, append its REST-verified witness:

```powershell
python -m cf_diff.prospective_ledger witness `
  --protocol configs/prospective_protocol_v2.json `
  --contest-id 3000 `
  --run-id 123456789
```

The command itself requests the frozen GitHub API endpoint and preserves the
exact response under `prospective/witnesses/`. It checks repository id/name,
`main`, push event, workflow name/path/hash, head commit, run attempt, status,
conclusion, referenced artifacts, and GitHub `created_at`. It accepts no local
response file or caller-supplied timestamp. A late run remains visible in the
chain but does not qualify the contest.

If capture, prediction, or publication fails, append a direct operational miss
instead of creating partial predictions. A direct miss is itself a base
coverage event: publish it to `main` and attach the same run-id witness so its
timeliness cannot be backdated. If later fixed evidence proves a committed
prediction invalid, append one objective invalidation; never edit or delete the
commitment.

## Fixed cohort-close snapshots

Start the census command before its first scheduled slot and leave it running:

```powershell
python -m cf_diff.prospective_snapshot acquire-census `
  --protocol configs/prospective_protocol_v2.json `
  --run-dir prospective/snapshots/v2/cohort-census
```

The census window is `[2027-03-01T00:04:59Z,
2027-03-02T00:04:59Z)`. The confirmatory outcome command is analogous:

```powershell
python -m cf_diff.prospective_snapshot acquire-outcome `
  --protocol configs/prospective_protocol_v2.json `
  --run-dir prospective/snapshots/v2/confirmatory-outcome
```

Its window is `[2027-03-04T00:04:59Z,
2027-03-05T00:04:59Z)`. Each window has exactly 48 scheduled half-hour slots,
a frozen 60-second start grace, one 30-second request per slot, and no 49th
request. Missing a slot invalidates the window. The first transport- and
structure-valid response is sealed even when its mappings, ratings, or sample
size are unfavorable.

Verify and append both selections to the observation chain:

```powershell
python -m cf_diff.prospective_snapshot verify `
  --protocol configs/prospective_protocol_v2.json `
  --kind cohort_census `
  --run-dir prospective/snapshots/v2/cohort-census

python -m cf_diff.prospective_ledger observation `
  --protocol configs/prospective_protocol_v2.json `
  --selection prospective/snapshots/v2/cohort-census/selection.json
```

Repeat those two commands with `confirmatory_outcome` and its run directory.

## Cohort mapping and confirmatory analysis

Finalize only from the two sealed selections and both event chains:

```powershell
python -m cf_diff.prospective_cohort `
  --protocol configs/prospective_protocol_v2.json `
  --census-run-dir prospective/snapshots/v2/cohort-census `
  --outcome-run-dir prospective/snapshots/v2/confirmatory-outcome `
  --commitment-ledger prospective/ledger/commitments.jsonl `
  --observation-ledger prospective/ledger/observations.jsonl `
  --output-dir prospective/cohort/v2
```

Every census contest must map exactly once to a prediction commitment or direct
miss, with no ledger-extra contest and no exclusion bucket. Official start and
the complete official index set are checked after snapshot selection. A newly
discovered mismatch must first receive an append-only invalidation. Finalized
outcomes retain every locked prediction key, including missing, non-programming,
absent, and already-invalidated rows.

The analysis command remains blocked until `2027-03-05T00:04:59Z`, and then
re-verifies both ledgers, mappings, hashes, time windows, and thresholds:

```powershell
python -m cf_diff.prospective_analysis `
  --protocol configs/prospective_protocol_v2.json `
  --input-manifest prospective/cohort/v2/confirmatory_analysis_input.json `
  --output prospective/results/v2/confirmatory_analysis.json
```

There are no CLI overrides for time, seed, resample count, endpoint, retry
schedule, quantile method, or success boundary. Fewer than 30 eligible contests
or 200 paired eligible problems produces an underpowered report condition and
blocks all aggregate metrics; the window is never extended post hoc.

## Required verification before every evidence push

```powershell
python -m cf_diff.prospective_ledger verify
python -m pytest -q
```

The GitHub witness workflow additionally compares the push against the prior
`main` commit. Event chains may only grow by a byte-identical prefix append;
published evidence cannot be modified, deleted, renamed, symlinked, or
case-collided. Once the protocol is frozen, its control source and workflow
cannot change in place either.
