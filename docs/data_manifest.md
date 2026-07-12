# Data and Artifact Manifest

This repository is a research codebase, not a hosted dataset release. It uses
local Codeforces API snapshots and locally cached Codeforces problem-page HTML
to build the reported v5 and v6 results.

Historical counts and outputs in this document are governed by the
[2026-07-10 public erratum](ERRATUM_2026-07-10.md). They are retrospective and
must not be described as a still-unseen final test.

Large raw API snapshots, original cached problem pages, and logs are
intentionally not committed. Compact sealed derived artifacts for the locked
historical statement-only backtest are committed and documented below. Other
large generated outputs generally remain local. As a result, rerunning the full
pipeline later against live Codeforces data may produce slightly different
numbers. The committed papers report results from the author's local frozen
snapshot.

## Stable counts reported in the papers

| Item | Count / value |
|---|---:|
| Rated programming problems in model table | 10,979 |
| Contests represented | 1,948 |
| Rating range | 800 to 3500 |
| Statement text extraction attempted rows | 10,979 |
| Statement text extracted success | 10,906 |
| Statement text missing/failure rows | 73 |
| Statement text availability rate | about 99.3% |
| v6 matched statement text rows | 10,979 |
| v6 usable text rows | 10,906 |
| TF-IDF max features | 20,000 |
| TF-IDF n-gram range | 1 to 2 |

## Data sources and local artifacts

The raw public API sources are:

- Codeforces `problemset.problems`
- Codeforces `contest.list`

Important local artifact paths used by the pipeline:

- Main model table:
  `data/processed/features/model_table.parquet`
- Statement text-light features:
  `data/processed/statement_features/statement_features.parquet`
- Statement text artifact:
  `data/processed/statement_text/statement_text.parquet`
- v6 metrics source:
  `outputs/semantic_tfidf/tables/semantic_tfidf_best_by_setting.csv`
- v6 summary source:
  `outputs/semantic_tfidf/summary/semantic_tfidf_summary.json`
- v6 paper:
  `paper/paper_v6_semantic_tfidf.md`
- v6 PDF:
  `paper/paper_v6_semantic_tfidf_final.pdf`
- Canonical corrected experiment config:
  `configs/experiment.yaml`
- Historical effective split config:
  `configs/experiment_legacy_v6.yaml`

The local raw snapshot metadata records provenance and hashes when present.
However, not all large artifacts are committed, so this repository alone should
not be interpreted as fully reproducing the exact paper numbers without the
author's local snapshot artifacts. The historical effective config documents
the old split ratios; using it with current code is an audit aid, not a complete
paper-reproduction recipe. Exact historical reproduction also requires the
matching Git tag and original local snapshot.

## Historical statement backtest cache seal

The statement-only backtest uses the local cache state sealed on 2026-07-12.
The committed [file manifest](../data/manifests/historical_statement_cache_v1.csv)
contains a relative path, stored-byte SHA-256 digest, size, filesystem mtime,
and conservative content classification for each of 10,979 entries. Its
[summary](../data/manifests/historical_statement_cache_v1_summary.json) records:

- 10,906 HTML entries, 71 PDF responses, and 2 empty files;
- 718,348,870 stored bytes in total;
- manifest SHA-256
  `c6314ae0297533d6747f85093f11c033fe06f91fec8f1f68f4d72c74485a3df0`.

These digests seal the current local representation only. The legacy cache
writer decoded response text and re-encoded it as UTF-8, so these are not hashes
of the original HTTP response bytes. Filesystem mtimes are recorded as local
provenance metadata but do not independently prove capture time or contest-time
page state.

## Historical statement-only backtest result artifacts

The frozen selection was committed at
`1260f4bd54db42ac985797bd99a1d462cc0289e5`; the committed locked test outputs were
committed at `12b85070180d9170981e8ff4c54b2c5c30980947`. The full interpretation,
including the evidence boundary and post-hoc error analysis, is in the
[result report](HISTORICAL_STATEMENT_BACKTEST_RESULTS.md).

The following compact evidence artifacts are committed:

- selection metadata and artifact hashes:
  [`selection_lock.json`](../outputs/historical_statement_backtest/selection/selection_lock.json)
  and [`selection_lock.sha256`](../outputs/historical_statement_backtest/selection/selection_lock.sha256);
- validation selection results:
  [`validation_metrics.csv`](../outputs/historical_statement_backtest/selection/validation_metrics.csv);
- frozen cohort inputs:
  [`prepared_dataset.parquet`](../outputs/historical_statement_backtest/selection/prepared_dataset.parquet),
  [`split_assignment.parquet`](../outputs/historical_statement_backtest/selection/split_assignment.parquet),
  and [`source_manifest.csv`](../outputs/historical_statement_backtest/selection/source_manifest.csv);
- final metrics and uncertainty:
  [`test_metrics.json`](../outputs/historical_statement_backtest/test/test_metrics.json)
  and [`paired_bootstrap.json`](../outputs/historical_statement_backtest/test/paired_bootstrap.json);
- split-level statement coverage:
  [`coverage.csv`](../outputs/historical_statement_backtest/test/coverage.csv);
- row-level audit outputs:
  [`test_predictions.csv`](../outputs/historical_statement_backtest/test/test_predictions.csv)
  and [`primary_top10.csv`](../outputs/historical_statement_backtest/test/primary_top10.csv);
- final output digests:
  [`result_manifest.sha256`](../outputs/historical_statement_backtest/test/result_manifest.sha256).

The split contains 7,375 / 1,136 / 2,468 train, validation, and test rows;
1,416 / 177 / 355 contests; and 1,107 / 158 / 316 complete contest start-time
buckets. Parsed-statement coverage is 7,352 / 7,375, 1,132 / 1,136, and
2,422 / 2,468, respectively.

These files make the reported rerun auditable within the repository, but they
do not change the underlying provenance limit: the statement cache is
post-contest, normalized rather than raw HTTP, and drawn from historical data
that had prior project exposure. This statement-only result is not prospective
and does not replace the legacy full-API headline metrics.

The committed prepared dataset is enough to recompute the reported model
selection and test metrics. The 718 MB page cache itself is not committed, so
the repository alone cannot reproduce the HTML-to-feature extraction step. A
separately licensed data archive would be required for end-to-end extraction
reproduction.

## Reviewer interpretation

This manifest exists to make the artifact situation explicit. It documents which
counts are reported from the frozen local run and which large files remain local
for size and reproducibility reasons. It does not claim to be a complete
archival release of the Codeforces data or cached HTML corpus.
