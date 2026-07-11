# Prospective Research Protocol v2 — draft

Protocol ID: `cf-difficulty-prospective-v2`

Machine-readable source:
[`configs/prospective_protocol_v2.json`](../configs/prospective_protocol_v2.json)

Status: **draft**. The tentative eligibility window is 2026-08-15 through
2027-02-28 UTC. No contest may be enrolled while the protocol is a draft.

The v2 draft replaces v1 before enrollment. See the
[v1 retirement record](PROSPECTIVE_V1_RETIREMENT.md).

## Research question

Does a Ridge model using only contest index and lightweight structure extracted
from a public problem statement improve genuinely prospective Codeforces
rating prediction over an index-only Ridge comparator?

The paired problem-level improvement in absolute error remains the sole
confirmatory estimand. The planned interval is a 95% percentile interval from
10,000 contest-cluster bootstrap resamples with the pre-specified seed in the
machine-readable protocol. Secondary metrics and index bands are descriptive.

## T0 isolation boundary

For each contest, the capture command accepts only:

- an explicit numeric `contest_id`;
- the complete list of problem indices visible to the operator;
- the scheduled contest start in UTC; and
- the frozen protocol plus new output locations.

It constructs public problem-statement URLs directly. It does not query the
Codeforces problem metadata API or read a local metadata table. The model CSV
contains exactly `contest_id`, `index`, and the 41 frozen statement-structure
features. URLs, fetch status, timestamps, raw-page hashes, and errors are
stored only in a sidecar.

The predictor reads the CSV header or Parquet schema before row values. A label,
points, tags, solved counts, submissions, acceptance data, unexpected audit
field, caller-supplied derived index field, or changed column order causes an
immediate failure. After schema approval it loads only the exact allowlist.

Every capture and prediction output uses exclusive creation. One failed or
unparsed statement invalidates the whole contest input; partial prediction is
prohibited. The later ledger will record that contest as an operational miss.

## Freeze gates

The protocol may change from draft to frozen only after all of these are true:

1. the historical evaluation correction has merged;
2. capture and predictor tests pass on Linux and the local environment;
3. a synthetic dry run completes capture → prediction → verification without
   reading a metadata table;
4. the append-only ledger and public timestamp workflow are ready;
5. the tentative start still leaves a review buffer.

If a gate is not complete in time, the start date must move forward before
freezing. It must never be backdated.

Freezing is a two-commit operation:

1. change the protocol status to `frozen`, record its UTC freeze timestamp, and
   commit that protocol;
2. from that clean commit, generate the JSON model bundle and manifest, verify
   them, and commit the artifacts before the first eligible contest.

The model manifest records the full source commit, training-input hashes,
dependency-spec hash, exact runtime versions, training cutoff, and model hash.
The estimator locks Ridge alpha, intercept behavior, and the deterministic SVD
solver.

## Current scope boundary

This branch prepares capture, model freezing, prediction, and documentation.
It does not open the cohort. Append-only prediction/outcome ledgers, reveal
separation, public timestamp checks, and the final operational runbook belong
to the next sequential change.
