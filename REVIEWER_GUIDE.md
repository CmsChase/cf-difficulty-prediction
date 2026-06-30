# Reviewer Guide

This is the two-minute map for the final v5 research version of
`cf-difficulty-prediction`.

## Project goal

Predict official Codeforces problem ratings from public Codeforces data while
keeping the pipeline reproducible, auditable, and careful about leakage.

## Research question

How well can Codeforces problem difficulty be predicted from:

- official API metadata,
- post-publication solved statistics, and
- lightweight problem-statement structure features for cold-start prediction?

## Key contributions

- A reproducible Python pipeline from raw Codeforces API snapshots to processed
  parquet tables, features, splits, baselines, ablations, robustness checks, and
  paper artifacts.
- Two leakage-aware evaluation protocols: contest-grouped and forward-time.
- A clear distinction between post-publication prediction and cold-start
  prediction.
- v5 statement text-light experiments that test whether simple statement
  structure features improve cold-start prediction beyond API metadata alone.

## Main results

- Final modeling data: 10,979 rated `PROGRAMMING` problems from 1,948 contests.
- Rating range: 800 to 3500.
- Post-publication full API results:
  - contest-grouped: HGB MAE 166.9, within +/-200 = 69.7%;
  - forward-time: random forest MAE 152.5, within +/-200 = 71.2%.
- Solved-count-only is the strongest simple baseline, but full API models improve
  over it.
- Cold-start metadata-only prediction is much harder, around MAE 318-332.
- v5 metadata + statement text-light improves cold-start prediction:
  - contest-grouped: MAE 317.1 -> 284.0;
  - forward-time: MAE 331.4 -> 289.1.
- Statement feature coverage is 99.3%: 10,906 parsed pages out of 10,979 rows,
  with 73 missing-statement rows handled by imputation.

## Post-publication vs cold-start

Post-publication settings use solved-count behavior observed after problems have
been available to contestants. These features are strong but include exposure,
age, popularity, and participation effects.

Cold-start settings exclude solved behavior and ask what can be predicted from
metadata and lightweight statement structure before solve statistics accumulate.
The full API reference is therefore not a cold-start result.

## Reproducibility commands

Install dependencies and run tests:

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m pytest -q
```

Run the v5 statement cold-start experiment after statement features exist:

```powershell
python -m cf_diff.statement_cold_start `
  --feature-path data/processed/features/model_table.parquet `
  --statement-feature-path data/processed/statement_features/statement_features.parquet `
  --output-dir outputs/statement_cold_start `
  --log-path outputs/logs/statement_cold_start.log
```

Final paper artifacts:

- [`paper/paper_v5_full_en.md`](paper/paper_v5_full_en.md)
- [`paper/paper_v5_full_en_final.pdf`](paper/paper_v5_full_en_final.pdf)

## Limitations

- Current cold-start is metadata cold-start, not strict pre-contest cold-start.
- Codeforces ratings are treated as labels, not as perfect ground truth.
- Solved counts are predictive but confounded by exposure and time.
- Age-normalized features are simple proxies, not full solve-curve models.
- Statement text-light features are approximate HTML-derived structure features,
  not semantic embeddings or deep NLP.
- The v5 cold-start improvements do not prove semantic understanding of problem
  difficulty.
