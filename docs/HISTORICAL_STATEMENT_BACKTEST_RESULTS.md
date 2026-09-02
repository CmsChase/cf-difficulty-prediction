# Historical Statement-Only Backtest Results

Status: final committed locked test result, 2026-07-12.

This is a **forward-time retrospective evaluation** of two Ridge models. The
comparator uses only problem-index features; the primary model adds the frozen
allowlist of statement-structure features. The primary model reduced test MAE
from **477.2151 to 401.2704**, a difference of **-75.9447 rating points**
(about **15.9%**). A paired contest-cluster bootstrap placed the 95% interval
for that difference at **[-86.7339, -65.0783]**.

This result supports the narrower claim that statement structure adds
predictive signal in this historical dataset under the frozen evaluation. It
does not establish prospective performance.

## Evidence anchors

- Selection was sealed before the test run in commit
  `1260f4bd54db42ac985797bd99a1d462cc0289e5`.
- The committed locked test outputs were recorded in commit
  `12b85070180d9170981e8ff4c54b2c5c30980947`.
- Both feature sets selected `alpha = 0.01` using validation MAE only.
- The test contrast was calculated over 2,468 problems in 355 contests. The
  uncertainty interval used 10,000 paired resamples clustered by contest.
- The protocol intentionally retains its pre-run wording; this separate report
  records the completed execution and results.
- The execution environment was Python 3.12.13, NumPy 2.3.5, pandas 2.3.3,
  scikit-learn 1.9.0, SciPy 1.18.0, and PyArrow 21.0.0. These versions were
  recorded after execution, not restored from an exact environment lock.

The frozen [protocol](HISTORICAL_STATEMENT_BACKTEST.md) and
[configuration](../configs/historical_statement_backtest.json) define the
split, exact feature allowlists, selection rule, test policy, uncertainty
calculation, and automatic error-analysis rule.

## Cohort, split, and coverage

Complete contest start-time buckets were allocated chronologically; neither a
contest nor a shared start timestamp crosses a boundary.

| Split | Rows | Contests | Start-time buckets | Parsed statements | Coverage |
|---|---:|---:|---:|---:|---:|
| Train | 7,375 | 1,416 | 1,107 | 7,352 | 99.69% |
| Validation | 1,136 | 177 | 158 | 1,132 | 99.65% |
| Test | 2,468 | 355 | 316 | 2,422 | 98.14% |
| **Total** | **10,979** | **1,948** | **1,581** | **10,906** | **99.34%** |

Missing or unparseable pages remained in the cohort. The model received
`statement_available = 0`, parser defaults for other count fields, and numeric
imputation learned from the relevant training partition.

## Validation-only selection

| Setting | Features | Selected alpha | Validation MAE |
|---|---|---:|---:|
| Comparator | `index_rank`, `index_number` | 0.01 | 493.6116 |
| Primary | Comparator + frozen statement-structure allowlist | 0.01 | 422.2488 |

After selection, each model was refit on train plus validation. No feature,
alpha, clipping rule, or processing choice was changed after the test was
opened.

## Final test result

| Setting | MAE | RMSE | R-squared |
|---|---:|---:|---:|
| Comparator | 477.2151 | 575.4347 | 0.5005 |
| Primary | **401.2704** | **503.2983** | **0.6179** |

The frozen primary contrast is
`primary MAE - comparator MAE = -75.9447`. Negative values favor the primary
model. Relative to the comparator MAE, the reduction is about 15.9%.
That percentage is relative only to the predeclared index-only comparator; it
is not an improvement claim over the legacy full-API model or the state of the
art.

The paired contest-cluster bootstrap result was:

- point estimate: -75.9447;
- 95% percentile interval: **[-86.7339, -65.0783]**;
- clusters: 355 test contests;
- resamples: 10,000, using the frozen PCG64 seed.

The interval is descriptive uncertainty for this locked historical cohort. It
does not remove the provenance and prior-exposure limitations below.

## Automatic Top-10 error analysis

The frozen rule selected the primary model's ten largest absolute test errors;
no example was manually substituted. This analysis is necessarily
**post-hoc**.

| Rank | Problem | Rating | Prediction | Absolute error |
|---:|---|---:|---:|---:|
| 1 | [1912/L — LOL Lovers](https://codeforces.com/problemset/problem/1912/L) | 800 | 3,793.6404 | 2,993.6404 |
| 2 | [2038/N — Fixing the Expression](https://codeforces.com/problemset/problem/2038/N) | 800 | 3,503.5058 | 2,703.5058 |
| 3 | [2073/L — Boarding Queue](https://codeforces.com/problemset/problem/2073/L) | 1,300 | 3,615.3657 | 2,315.3657 |
| 4 | [2172/M — Maximum Distance To Port](https://codeforces.com/problemset/problem/2172/M) | 1,300 | 3,433.1363 | 2,133.1363 |
| 5 | [2181/M — Medical Parity](https://codeforces.com/problemset/problem/2181/M) | 1,700 | 3,792.4850 | 2,092.4850 |
| 6 | [2041/M — Selection Sort](https://codeforces.com/problemset/problem/2041/M) | 2,000 | 4,058.4470 | 2,058.4470 |
| 7 | [1906/M — Triangle Construction](https://codeforces.com/problemset/problem/1906/M) | 1,700 | 3,681.1086 | 1,981.1086 |
| 8 | [1912/K — Kim's Quest](https://codeforces.com/problemset/problem/1912/K) | 1,800 | 3,752.1220 | 1,952.1220 |
| 9 | [2038/J — Waiting for...](https://codeforces.com/problemset/problem/2038/J) | 800 | 2,752.0112 | 1,952.0112 |
| 10 | [2045/M — Mirror Maze](https://codeforces.com/problemset/problem/2045/M) | 1,800 | 3,731.9566 | 1,931.9566 |

Manual post-hoc inspection identified all ten as ICPC-style problems; all are
overpredictions. A likely explanation is structural: unlike typical Codeforces
rounds, ICPC problem letters are identifiers and are not ordered by difficulty.
The unbounded linear Ridge model can also extrapolate beyond the observed
rating range when an index and statement-feature combination is unusual. For
2073/L, the cached response is a PDF rather than parsable HTML
(`pdf_not_html`), so its HTML statement features are missing as specified by
the frozen policy.

These observations describe failure modes; they do not authorize a retroactive
fix. Clipping predictions, changing how ICPC indices are encoded, adding a
contest-type interaction, or changing the model after viewing the test would
invalidate the locked comparison. Any such change belongs in a separately
specified future experiment.

## Evidence boundary and non-claims

- The problem pages were collected after the contests. Current pages may
  contain corrections or edits that were unavailable at contest start.
- The sealed cache contains response text decoded and re-encoded as UTF-8 by a
  legacy cache writer. It is a normalized local representation, not the raw
  byte-for-byte HTTP response.
- Earlier project work had already examined these historical contests and
  related data. Freezing this protocol before the rerun limits new analytical
  discretion, but it is not an independent preregistration.
- Git history records one committed locked evaluation after the public
  selection commit. The filesystem guard prevents overwriting the same output
  path, but cannot prove that no unrecorded execution occurred elsewhere.
- Chronological splitting makes this a forward-time retrospective backtest,
  not a prospective or genuinely unseen future cohort.
- This is a separate statement-only result. It does **not** replace or correct
  the legacy full-API headline metrics in the historical papers.
- The prepared dataset is sufficient to recompute the reported selection and
  test metrics, but the 718 MB page cache is not committed. A third party cannot
  reproduce the HTML-to-feature extraction from this repository alone. A
  separately licensed archival release would be needed for that end-to-end
  reproduction step.

## Committed result artifacts

The committed prepared-data-to-result path can be checked without overwriting
these artifacts by running `python -m cf_diff.verify_locked_backtest` from the
repository root after setting `PYTHONPATH=src`. The verifier does not recreate
HTML-derived features because the page cache is not committed.

- [Selection lock](../outputs/historical_statement_backtest/selection/selection_lock.json)
  and its [SHA-256 file](../outputs/historical_statement_backtest/selection/selection_lock.sha256)
- [Validation metrics](../outputs/historical_statement_backtest/selection/validation_metrics.csv)
- [Split assignment](../outputs/historical_statement_backtest/selection/split_assignment.parquet)
- [Prepared dataset](../outputs/historical_statement_backtest/selection/prepared_dataset.parquet)
- [Per-page source manifest](../outputs/historical_statement_backtest/selection/source_manifest.csv)
- [Final test metrics](../outputs/historical_statement_backtest/test/test_metrics.json)
- [Paired bootstrap output](../outputs/historical_statement_backtest/test/paired_bootstrap.json)
- [Coverage by split](../outputs/historical_statement_backtest/test/coverage.csv)
- [Test predictions](../outputs/historical_statement_backtest/test/test_predictions.csv)
- [Automatic Top-10 errors](../outputs/historical_statement_backtest/test/primary_top10.csv)
- [Final-result SHA-256 manifest](../outputs/historical_statement_backtest/test/result_manifest.sha256)
- [Sealed cache manifest](../data/manifests/historical_statement_cache_v1.csv)
  and [summary](../data/manifests/historical_statement_cache_v1_summary.json)
