# Reviewer Guide

This is the two-minute map for `cf-difficulty-prediction`, a reproducible
research project on predicting Codeforces problem difficulty.

> **Start with the public erratum.** Historical tags `v1.0` through `v2.0`
> used effective 70/15/15 splits despite different declared ratios, and several
> winners were selected using test MAE. Historical numbers below are retained
> as retrospective evidence, not an unseen confirmatory test.

## What the project studies

The project predicts official Codeforces problem ratings from public data:
official API metadata, post-publication solved statistics, and problem-statement
features. It separates post-publication prediction from cold-start prediction.

Cold-start here means **no solved-count behavior**, not strict pre-contest
prediction. Tags, metadata, and statement availability may not exactly match a
real pre-contest setting.

## Main contributions

- A reproducible Python pipeline from raw Codeforces API snapshots to processed
  parquet tables, features, splits, baselines, ablations, robustness checks, and
  papers.
- Leakage-aware contest-grouped and forward-time evaluation.
- v5 statement text-light experiments showing that simple statement-structure
  features improve cold-start prediction.
- v6 semantic TF-IDF experiments showing that classical statement-text features
  add signal beyond metadata and text-light structure features, while remaining
  much weaker than deep semantic understanding.

## Historical results (retrospective)

- Final modeling data: 10,979 rated `PROGRAMMING` problems from 1,948 contests.
- Rating range: 800 to 3500.
- v5 full API post-publication results:
  - contest-grouped: HGB MAE 166.9, within +/-200 = 69.7%;
  - forward-time: random forest MAE 152.5, within +/-200 = 71.2%.
- Solved-count-only is the strongest simple baseline, but full API models
  improve over it.
- v5 metadata + statement text-light improves cold-start prediction:
  - contest-grouped: MAE 317.1 -> 284.0;
  - forward-time: MAE 331.4 -> 289.1.
- v6 metadata + TF-IDF improves over metadata only, and metadata + text-light +
  TF-IDF improves over metadata + text-light. v6 does not replace the canonical
  v5 full API benchmark.

## Read these files first

- [`docs/ERRATUM_2026-07-10.md`](docs/ERRATUM_2026-07-10.md) for the public correction.
- [`docs/RESEARCH_PROTOCOL_V1.md`](docs/RESEARCH_PROTOCOL_V1.md) for the future blind test.
- [`README.md`](README.md) for the project overview and commands.
- [`paper/paper_v5_full_en_final.pdf`](paper/paper_v5_full_en_final.pdf) for
  the historical v5 paper.
- [`paper/paper_v6_semantic_tfidf_final.pdf`](paper/paper_v6_semantic_tfidf_final.pdf)
  for the separate semantic TF-IDF extension paper.
- [`docs/data_manifest.md`](docs/data_manifest.md) for what data and generated
  artifacts are local rather than committed.

## Reproducibility check

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m pytest -q
```

Large raw snapshots, cached HTML pages, logs, and generated experiment outputs
are intentionally not committed. The committed papers report the author's local
frozen snapshot; rerunning against live Codeforces data later may produce small
differences.

## Limitations

- Codeforces ratings are treated as labels, not perfect ground truth.
- Solved counts are predictive but confounded by exposure, age, popularity, and
  participation.
- Cold-start experiments exclude solved behavior but are not strict pre-contest
  forecasts.
- Statement text-light and TF-IDF features are approximate HTML/text-derived
  signals, not deep language understanding.
