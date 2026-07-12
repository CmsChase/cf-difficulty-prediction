# Historical Statement-Only Backtest Protocol

Status: design frozen before the new rerun.

This study is a **forward-time retrospective evaluation** of difficulty
prediction from public problem statements. It is not a prospective study and
not an independent preregistration. The protocol is committed before this
branch's outcome rerun, but the underlying contests and labels are historical
and earlier project work has already examined related data.

## Evidence boundary

Problem pages were collected after their contests. A current page may include
corrections or other edits that were unavailable when the contest began, so
this study cannot establish real-time performance. The legacy page cache also
contains decoded response text re-encoded as UTF-8 by the legacy cache writer,
not the byte-for-byte HTTP response. A SHA-256 digest will be recorded for each
stored file before parsing; this identifies the exact cached representation
used in the rerun, but neither filesystem timestamps nor these hashes prove
when the page was captured or what the server originally returned.

The analysis cohort is the rated historical problem table. Missing or
unparseable pages remain in the cohort: `statement_available` is zero, other
count features use their parser defaults, and missing numeric limits are
imputed from the relevant training partition. Capture and parse coverage will
be reported overall and for every split.

## Frozen split and predictors

Unique contest start-time buckets are sorted chronologically and allocated to
train, validation, and test at 70/10/20 using deterministic largest-remainder
counts. All contests that began at the same timestamp stay in the same bucket,
so neither a contest nor a concurrent start-time bucket crosses a boundary.
All three partitions must be nonempty and their timestamp ranges must be
strictly ordered.

Two feature sets are compared:

- **Comparator:** `index_rank` and `index_number`.
- **Primary:** the comparator plus the exact statement-structure allowlist in
  `configs/historical_statement_backtest.json`.

`contest_id` and the raw `index` are join, split, and reporting identifiers
only; they are never model features. The derived numeric index features above
are allowed. No ratings, points, tags, solved or acceptance counts,
submissions, participation, hacks, or similarly named fields may enter either
feature matrix. The runner must fail on an unexpected feature rather than
silently include it.

## Selection and final test

Both feature sets use Ridge regression with median imputation and standard
scaling learned only from the current training data. For each feature set,
alpha is selected from the frozen candidate grid using validation MAE only;
ties choose the lowest alpha. After selection, each model is refit on combined
training and validation contests. The test partition is then evaluated once.
Test results cannot be used to change features, alpha candidates, processing,
or the reported primary comparison.

The primary metric is MAE and the primary contrast is primary MAE minus
comparator MAE. RMSE and R-squared are secondary. A paired 10,000-resample
contest-cluster bootstrap reports a 95% percentile interval for the primary
contrast: test contests are sampled with replacement, all problems from each
sampled contest travel together, and both models use the same resample.

## Fixed outputs

The rerun will record the protocol hash, cache manifest and SHA-256 digests,
split membership, page coverage, selected alphas, one set of test predictions,
metrics, and bootstrap interval. Error analysis is the primary model's ten
largest test absolute errors, selected automatically. Ties are resolved by
contest start time, `contest_id`, and `index`; examples cannot be substituted
manually.

TF-IDF and other free-text models are outside this first confirmatory scope.
They may be reported later only as separately labeled exploratory work, without
changing this study's frozen primary result.
