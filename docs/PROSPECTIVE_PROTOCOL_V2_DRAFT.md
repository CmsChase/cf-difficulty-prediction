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

The machine-readable protocol now fixes the implementation details that could
otherwise create analyst discretion: PCG64, one 10,000-by-contest draw, seed
`20260710`, numerically sorted contest-id clusters, expanded problem-level
cluster multiplicity, linear 2.5%/97.5% quantiles, and a strict lower-bound
greater-than-zero success rule. The command must stop before calculating any
aggregate when an integrity, time, or sample-size gate fails.

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
Neither operational CLI accepts a caller-supplied capture or prediction time.
Both use the system UTC clock and check it before and after publication; a
deadline crossing removes the model input or prediction.
The sanitized input, sidecar, prediction, and ledger commitment must also reach
the default branch before the deadline. A GitHub Actions push-event run URL,
id, and GitHub-created timestamp are the external witness; local commit dates
do not suffice.
The witness policy is bound to repository id `1280990637`, branch `main`, the
dedicated `prospective-witness.yml` workflow, its SHA-256, a push event, and
run attempt 1. The workflow-run REST response is retained as immutable public
evidence. Its GitHub `created_at` value is the deadline clock; a local timestamp
or Git commit date is never substituted.
Direct operational misses use the same base-event deadline and push witness;
missing or late miss evidence remains visibly untimely rather than backdated.
The later reveal step also verifies the operator-entered scheduled start and
complete index list. A mismatch invalidates the whole contest prediction set;
rows are never added or corrected after T0.

## Freeze gates

The protocol may change from draft to frozen only after all of these are true:

1. the historical evaluation correction has merged;
2. capture and predictor tests pass on Linux and the local environment;
3. a synthetic dry run completes capture → prediction → verification without
   reading a metadata table;
4. the append-only ledger and public timestamp workflow are ready;
5. deterministic confirmatory-analysis code and fixture tests are frozen,
   including clustered resampling, RNG, quantile, missingness, and duplicate
   handling, and the command refuses to run before cohort close;
6. the tentative start still leaves a review buffer.

If a gate is not complete in time, the start date must move forward before
freezing. It must never be backdated.

Freezing is a two-commit operation:

1. change the protocol status to `frozen`, record its UTC freeze timestamp, and
   commit that protocol;
2. from that clean commit, generate the JSON model bundle and manifest, verify
   them, and commit the artifacts before the first eligible contest.

The protocol becomes immutable in the first commit. Any later substantive
change, even before the first prediction, requires a new protocol ID and a new
future cohort.

The model manifest records the full source commit, training-input hashes,
dependency-spec hash, key runtime and numerical-library versions, training
cutoff, and model hash.
It also hashes the capture, prediction, ledger, snapshot, cohort, and analysis
modules; both workflows; and all prospective protocol/model/operation tests.
The protocol stores the PCG64 golden draw, replicate, interval, and quantile
values checked by those tests.
The estimator locks Ridge alpha, intercept behavior, and the deterministic SVD
solver.

These records make the committed model and future predictions hash-auditable.
They do not make the model independently rebuildable from this repository:
the historical training snapshots are local and only their hashes are public,
and transitive numerical-library details may differ across systems.

The 30-contest and 200-problem thresholds count only contests and problems with
locked paired predictions and a rating in the fixed post-close outcome
snapshot. Operational misses are reported in the coverage denominator but do
not count toward the thresholds or bootstrap. Earlier 72-hour reveal snapshots
are append-only audit records; the confirmatory outcome uses the first
successful hashed API snapshot in the fixed post-close window, and missing
ratings are not filled from later polls.

Silent whole-contest omission is checked independently at cohort close. The
census window is the half-open interval from `2027-03-01T00:04:59Z` through
`2027-03-02T00:04:59Z`. It has exactly 48 scheduled requests, at 30-minute
intervals with no request at the closing boundary. The first response that
passes only the frozen transport and basic-structure predicate is sealed,
even if its later mapping, ratings, or sample size are unfavorable. Every
in-window contest must map exactly once to predictions or an operational miss;
there is no discretionary contest-exclusion bucket.

The confirmatory rating snapshot also cannot be chosen opportunistically. Its
half-open window is `2027-03-04T00:04:59Z` through
`2027-03-05T00:04:59Z`, again with exactly 48 scheduled requests. The first
structure-valid response is retained even when ratings are missing. A missed
scheduled slot invalidates the window; an all-failure window cannot be replaced
with a later snapshot. Confirmatory analysis cannot begin before the latter
deadline.

## Current scope boundary

This change prepares the append-only commitment/observation chains, public
timestamp validation, fixed snapshot acquisition, cohort-integrity mapping,
and deterministic confirmatory analysis. It still does not open the cohort or
freeze a model. Those actions require merged prerequisites, a clean-main
synthetic dry run, independent review, and a separate explicit freeze commit.
