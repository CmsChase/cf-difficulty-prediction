# Codeforces Difficulty Prediction

A reproducible machine-learning study of Codeforces problem difficulty prediction using public API metadata, solved-count behavior, cold-start evaluation, exposure-aware analysis, and rolling temporal validation.

<!-- project-showcase-start -->
## Project Snapshot

| Experiment | Main setting | Best / key result | Interpretation |
|---|---|---:|---|
| Contest-grouped prediction | HGB full API | MAE 166.9, within ±200 = 69.7% | Generalizes to unseen contests |
| Forward-time prediction | RF full API | MAE 152.5, within ±200 = 71.2% | Tests chronological generalization |
| Solved-only baseline | solved-count only | MAE 227-274 | Solved behavior is the strongest simple public signal |
| Cold-start prediction | metadata only | MAE 318-332 | New-problem prediction is much harder |
| Rolling temporal validation | full API + age norm | average MAE 146.0 | Best across rolling chronological folds |

## Key Figures

![Test MAE by model](paper/figures/test_mae_by_model.png)

![Feature drop MAE change](paper/figures/feature_drop_mae_change.png)

![Rolling-window MAE](paper/figures/rolling_window_mae.png)

## Project Overview

**Problem.** Can Codeforces problem difficulty be predicted from public API metadata and solved-count behavior?

**Dataset.** The final modeling table contains 10,979 rated Codeforces programming problems from 1,948 contests.

**Method.** The project uses a reproducible Python pipeline covering API data collection, preprocessing, feature engineering, grouped and temporal evaluation, baseline models, ablations, robustness experiments, exposure-aware analysis, and rolling-window temporal validation.

**Key finding.** Solved-count behavior is the strongest simple public signal, but it is not a pure difficulty measure. Cold-start prediction remains much harder, and exposure-aware features help most when combined with full API metadata.

**Limitation.** Full API prediction uses post-publication solved statistics. Age-normalized features are simple exposure proxies, not full solve-curve models.

**Why it matters.** The project separates post-publication prediction from cold-start prediction and shows how difficulty, exposure, popularity, and time interact in competitive-programming data.
<!-- project-showcase-end -->

## Research Question

How well can Codeforces problem difficulty be predicted across post-publication and cold-start settings using public metadata, solved-count statistics, and lightweight problem-statement structure features?

This project separates two prediction scenarios:

- **Post-publication prediction:** solved-count statistics are available after a problem has been published and attempted by users.
- **Cold-start prediction:** solved-count statistics are unavailable, so the model must rely on metadata and problem-statement structure features.

The main goal is not only to maximize prediction accuracy, but also to understand how different information sources contribute to difficulty prediction and where solved-count-based models fail under realistic cold-start conditions.


<!-- v5-statement-cold-start-start -->
## v5 Statement Text-Light Cold-Start Results

This extension tests whether lightweight problem-statement structure features can improve cold-start Codeforces difficulty prediction beyond API metadata alone.

| Split | Setting | Best model | MAE | Within ?200 | Interpretation |
|---|---|---|---:|---:|---|
| Contest-grouped | metadata only | HGB | 317.1 | 40.1% | Cold-start API metadata baseline |
| Contest-grouped | metadata + text-light | HGB | 284.0 | 46.2% | +33.0 MAE improvement |
| Forward-time | metadata only | HGB | 331.4 | 34.3% | Chronological cold-start baseline |
| Forward-time | metadata + text-light | HGB | 289.1 | 42.6% | +42.3 MAE improvement |

Key result: text-light features alone are weak, but they add useful complementary signal when combined with metadata. The strongest v5 conclusion is that lightweight statement-structure features improve cold-start prediction beyond API metadata alone.

![Statement cold-start MAE comparison](paper/figures/statement_cold_start_mae_comparison.png)

![Statement feature coverage](paper/figures/statement_feature_coverage.png)

Statement feature coverage:
- 10,979 model-table rows
- 10,906 parsed statement pages
- 73 missing-statement rows
- 99.3% statement feature coverage
- missing-statement rows handled by imputation
<!-- v5-statement-cold-start-end -->

## Reproducing the v5 Statement Text-Light Extension

The v5 extension adds lightweight problem-statement structure features for cold-start difficulty prediction.

### 1. Extract statement text-light features

```powershell
python -m cf_diff.statement_features `
  --feature-path data/processed/features/model_table.parquet `
  --cache-dir data/raw/codeforces/problem_pages `
  --output-dir data/processed/statement_features `
  --sleep-seconds 2.5 `
  --timeout 30 `
  --log-path outputs/logs/statement_features.log
```

For a small smoke test:

```powershell
python -m cf_diff.statement_features `
  --feature-path data/processed/features/model_table.parquet `
  --cache-dir data/raw/codeforces/problem_pages `
  --output-dir data/processed/statement_features `
  --sleep-seconds 2.5 `
  --timeout 30 `
  --max-pages 200 `
  --log-path outputs/logs/statement_features.log
```

### 2. Run statement cold-start experiments

```powershell
python -m cf_diff.statement_cold_start `
  --feature-path data/processed/features/model_table.parquet `
  --statement-feature-path data/processed/statement_features/statement_features.parquet `
  --output-dir outputs/statement_cold_start `
  --log-path outputs/logs/statement_cold_start.log
```

### 3. Run tests

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

Current tested state: `79 passed`.

The raw cached HTML files and generated output files are local reproducibility artifacts and are not committed to the repository.



## Research question

Can Codeforces official problem ratings be predicted from public platform signals, and how much do metadata, solved statistics, and lightweight statement-structure features contribute under post-publication and cold-start settings?

## Main findings

- The processed snapshot contains 10,979 rated `PROGRAMMING` problems from 1,948 contests.
- Ratings range from 800 to 3500.
- Solved-count-only is the strongest simple baseline in both contest-grouped and forward-time evaluation.
- Full models improve over solved-count-only, showing that metadata and tags add complementary signal.
- Removing solved features causes the largest MAE increase in ablation studies.
- Cold-start metadata-only prediction is substantially harder than post-publication prediction.
- Age-normalized solved-count features are useful in full models, but they are only a partial exposure adjustment.

## Repository structure

```text
configs/                 Experiment configuration
src/cf_diff/             Pipeline modules
tests/                   Unit tests
paper/                   Paper drafts, figures, tables, references
outputs/paper_tables/    Small paper-ready CSV tables
```

Large generated data, raw API snapshots, trained model files, caches, and logs should not be committed.

## Setup

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

## Reproducible pipeline

Run from the repository root.

```powershell
python -m cf_diff.fetch_api --output-root data/raw/codeforces --lang en --sleep-seconds 2.1 --timeout 30 --latest-dir data/raw/codeforces/latest
python -m cf_diff.preprocess --raw-dir data/raw/codeforces/latest --interim-dir data/interim --processed-dir data/processed --log-path outputs/logs/preprocess.log
python -m cf_diff.features --config configs/experiment.yaml --input-path data/processed/rated_programming_problems.parquet --output-dir data/processed/features --log-path outputs/logs/features.log
python -m cf_diff.splits --config configs/experiment.yaml --input-path data/processed/features/model_table.parquet --output-dir data/processed/splits --log-path outputs/logs/splits.log
python -m cf_diff.eda --config configs/experiment.yaml --processed-path data/processed/rated_programming_problems.parquet --feature-path data/processed/features/model_table.parquet --contest-split-path data/processed/splits/contest_grouped_split.parquet --time-split-path data/processed/splits/forward_time_split.parquet --output-dir outputs/eda --log-path outputs/logs/eda.log
python -m cf_diff.baselines --config configs/experiment.yaml --feature-path data/processed/features/model_table.parquet --feature-columns-path data/processed/features/feature_columns.json --contest-split-path data/processed/splits/contest_grouped_split.parquet --time-split-path data/processed/splits/forward_time_split.parquet --output-dir outputs/baselines --log-path outputs/logs/baselines.log
python -m cf_diff.analysis --config configs/experiment.yaml --feature-path data/processed/features/model_table.parquet --feature-columns-path data/processed/features/feature_columns.json --contest-split-path data/processed/splits/contest_grouped_split.parquet --time-split-path data/processed/splits/forward_time_split.parquet --baseline-metrics-dir outputs/baselines/metrics --baseline-predictions-dir outputs/baselines/predictions --output-dir outputs/analysis --log-path outputs/logs/analysis.log
python -m cf_diff.ablations --config configs/experiment.yaml --feature-path data/processed/features/model_table.parquet --feature-columns-path data/processed/features/feature_columns.json --contest-split-path data/processed/splits/contest_grouped_split.parquet --time-split-path data/processed/splits/forward_time_split.parquet --output-dir outputs/ablations --log-path outputs/logs/ablations.log
python -m cf_diff.robustness --config configs/experiment.yaml --processed-path data/processed/rated_programming_problems.parquet --feature-path data/processed/features/model_table.parquet --feature-columns-path data/processed/features/feature_columns.json --contest-split-path data/processed/splits/contest_grouped_split.parquet --time-split-path data/processed/splits/forward_time_split.parquet --output-dir outputs/robustness --log-path outputs/logs/robustness.log
```

## Tests

```powershell
python -m pytest -q
```

Current tested state: `79 passed`.

## Paper

The final v4 English paper is available under `paper/`.

Main files:
- `paper/paper_v4_full_en.md`
- `paper/paper_v4_full_en_final.pdf`

The v4 paper extends the earlier robustness version by adding exposure-aware analysis and rolling-window temporal validation. It distinguishes post-publication prediction from cold-start prediction, treats age-normalized solved count as a partial exposure proxy, and uses rolling temporal validation to test whether results remain stable across chronological folds.

## Exposure-Aware Extension

This branch adds an exposure-aware analysis of solved-count signals. The main project shows that public solved-count features are highly predictive for Codeforces rating, but solved count is not a pure difficulty signal. It can also reflect problem age, platform exposure, contest popularity, archival reuse, and participant behavior.

The extension studies this issue in three ways:

1. **Age-bucket analysis**  
   Problems are grouped by age buckets: `0-1y`, `1-3y`, `3-5y`, and `5y+`. The analysis compares metadata-only features, raw solved-count features, age-normalized solved-count features, metadata plus age-normalized features, and full API features plus age-normalized features.

2. **Popularity-difficulty mismatch analysis**  
   The module identifies examples where official rating and age-normalized popularity do not align cleanly, such as hard problems with high solves per day or easy problems with unusually low exposure.

3. **Exposure correlation summary**  
   The analysis reports correlations between official rating and exposure-related signals, including raw solved count, log solved count, problem age, solves per day, and log solves per day.

The extension remains conservative. Age normalization is treated as a simple proxy for exposure, not a complete correction. Mismatch examples are diagnostic rather than causal evidence, and the analysis does not use submission-level time series.

### Run the exposure-aware analysis

    python -m cf_diff.exposure_analysis `
      --config configs/experiment.yaml `
      --feature-path data/processed/features/model_table.parquet `
      --feature-columns-path data/processed/features/feature_columns.json `
      --contest-split-path data/processed/splits/contest_grouped_split.parquet `
      --time-split-path data/processed/splits/forward_time_split.parquet `
      --output-dir outputs/exposure `
      --log-path outputs/logs/exposure_analysis.log

### Main outputs

    outputs/exposure/summary/exposure_summary.json
    outputs/exposure/tables/age_bucket_metrics.csv
    outputs/exposure/tables/popularity_difficulty_mismatch_examples.csv
    outputs/exposure/tables/exposure_correlation_summary.csv
    outputs/exposure/figures/age_bucket_mae_by_feature_set.png
    outputs/exposure/figures/rating_vs_log_solves_per_day.png

The two main figures are also included in `paper/figures/` on this branch.
