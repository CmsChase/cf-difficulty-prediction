# Prospective Research Protocol v1

Protocol ID: `cf-difficulty-prospective-v1`

Machine-readable source:
[`configs/prospective_protocol_v1.json`](../configs/prospective_protocol_v1.json)

Status: frozen on 2026-07-10. The eligible future cohort starts at
2026-07-12 00:00:00 UTC and closes at 2027-02-28 23:59:59 UTC.

## Purpose

Historical Codeforces labels are now treated as development data. This protocol
creates a genuinely future evaluation of whether lightweight information from
a public problem statement improves rating prediction beyond contest-index
position alone.

The primary model and comparator are fixed Ridge regressions. The primary model
uses only index position and the 41 numeric statement-structure fields already
defined by this repository. The comparator uses only index rank and numeric
suffix. Neither model may use rating, points, tags, solved counts, submission
statistics, or manually assigned difficulty labels at prediction time.

## Frozen sequence

1. Freeze a JSON model bundle from historical data and publish its SHA-256,
   source commit, feature allowlists, training cutoff, and input hashes in a
   model-freeze manifest.
2. For an eligible future contest, capture the public statements without
   joining official ratings.
3. Generate predictions with the frozen JSON bundle. The prediction command
   rejects files containing rating or post-publication behavioral columns.
4. Append each prediction to the prediction ledger. Every event includes the
   previous event hash, so deletion, insertion, reordering, or modification is
   detectable.
5. Commit and push the prediction ledger before querying ratings for this
   project.
6. Only then run the separate reveal path. It appends actual ratings and errors
   to a different hash-chained ledger and never edits prediction events.
7. Do not compute aggregate prospective performance until the cohort closes.

The ledger must be locked no later than 30 minutes after the scheduled contest
start. A rating reveal is rejected until at least 72 hours after that start.
Every eligible contest is included; an operational miss is recorded as a
`missed_contest` event with a reason instead of being silently omitted.

## Primary analysis

The confirmatory estimand is the paired mean improvement in absolute error:

`index-only absolute error - text-light absolute error`.

A positive value favors the statement model. The uncertainty interval is a
two-sided 95% percentile interval from 10,000 contest-cluster bootstrap
resamples with seed `20260710`. The confirmatory success rule requires the
entire interval to be above zero. All other metrics, subgroups, and plots are
secondary/descriptive.

The cohort must contain at least 30 eligible contests and 200 rated problems.
If either threshold is missed by the fixed end date, the result is reported as
underpowered; the date is not extended after looking at outcomes.

## Amendments and deviations

This protocol file is immutable after the first prospective prediction. A
substantive change requires a new protocol ID, model bundle, ledger, and future
cohort. Operational deviations are appended to public documentation without
deleting or rewriting the original evidence.

The protocol does not claim that a Git hash proves the prediction was made at a
particular real-world instant. Its purpose is to make later alteration evident;
the public Git commit time supplies the external timestamped record.
