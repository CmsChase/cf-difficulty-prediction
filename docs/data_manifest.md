# Data and Artifact Manifest

This repository is a research codebase, not a hosted dataset release. It uses
local Codeforces API snapshots and locally cached Codeforces problem-page HTML
to build the reported v5 and v6 results.

Large raw data, cached HTML pages, logs, and generated experiment outputs are
intentionally not committed. As a result, rerunning the full pipeline later
against live Codeforces data may produce slightly different numbers. The
committed papers report results from the author's local frozen snapshot.

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

The local raw snapshot metadata records provenance and hashes when present.
However, not all large artifacts are committed, so this repository alone should
not be interpreted as fully reproducing the exact paper numbers without the
author's local snapshot artifacts.

## Reviewer interpretation

This manifest exists to make the artifact situation explicit. It documents which
counts are reported from the frozen local run and which large files remain local
for size and reproducibility reasons. It does not claim to be a complete
archival release of the Codeforces data or cached HTML corpus.
