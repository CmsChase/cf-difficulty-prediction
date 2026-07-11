# Quick Overview

> **Research status:** The numbers below are retained as retrospective results.
> A 2026-07-10 code audit identified a split-configuration mismatch and
> test-informed model selection; see the
> [public erratum](docs/ERRATUM_2026-07-10.md).
>
> A replacement prospective protocol is now in
> [pre-enrollment draft](docs/PROSPECTIVE_PROTOCOL_V2_DRAFT.md). The earlier
> v1 proposal was [superseded before enrollment](docs/PROSPECTIVE_V1_RETIREMENT.md).

This project studies a simple question: can Codeforces problem difficulty be predicted from public problem-level information?

The short answer is yes, but the answer depends heavily on what information is allowed. After a problem has been published, solved-count behavior is very useful. For a colder setting where solved counts are removed, prediction becomes much harder, so the project also tests whether problem-statement features can help.

## Main idea

The project builds a reproducible machine-learning pipeline around Codeforces rated programming problems. It uses public metadata, contest information, tags, solved statistics, and features extracted from problem statements.

The project separates two settings:

- **Post-publication prediction:** solved-count behavior is available.
- **Cold-start prediction:** solved-count behavior is removed.

Cold-start in this repo means **no solved-count behavior**, not strict pre-contest prediction. Tags, metadata, and statement availability may not exactly match a real pre-contest setting.

## Main dataset

The processed model table contains:

- 10,979 rated programming problems
- 1,948 contests
- Official rating range from 800 to 3500
- 10,906 usable extracted statement-text rows, about 99.3% coverage

Large raw API snapshots, cached HTML pages, logs, and generated experiment outputs are not committed. See [`docs/data_manifest.md`](docs/data_manifest.md) for details.

## Historical results (retrospective)

The strongest simple post-publication signal is solved count. This is useful, but it also reflects exposure, age, visibility, and popularity, not only intrinsic difficulty.

The main full-feature models improve over simple baselines:

- Contest-grouped full API model: about 166.9 MAE
- Forward-time full API model: about 152.5 MAE

The cold-start setting is harder, but statement features help.

Statement-structure features improve cold-start prediction:

- Contest-grouped: metadata-only MAE 317.1 → metadata + statement-structure MAE 284.0
- Forward-time: metadata-only MAE 331.4 → metadata + statement-structure MAE 289.1

TF-IDF statement-text features add another smaller but useful improvement:

- Contest-grouped: metadata-only MAE 340.5 → metadata + TF-IDF MAE 311.1
- Forward-time: metadata-only MAE 365.4 → metadata + TF-IDF MAE 325.4
- Contest-grouped: metadata + statement-structure MAE 310.6 → metadata + statement-structure + TF-IDF MAE 298.5
- Forward-time: metadata + statement-structure MAE 335.8 → metadata + statement-structure + TF-IDF MAE 316.2

TF-IDF alone is weak. Its value is mainly as an extra signal combined with metadata and statement-structure features.

## What to read first

Recommended order:

1. [`docs/ERRATUM_2026-07-10.md`](docs/ERRATUM_2026-07-10.md) for the governance correction.
2. [`README.md`](README.md) for the project overview and corrected pipeline.
3. [`docs/PROSPECTIVE_PROTOCOL_V2_DRAFT.md`](docs/PROSPECTIVE_PROTOCOL_V2_DRAFT.md) for the future blind-test design.
4. [`paper/paper_v5_full_en_final.pdf`](paper/paper_v5_full_en_final.pdf) for the historical full study.
5. [`paper/paper_v6_semantic_tfidf_final.pdf`](paper/paper_v6_semantic_tfidf_final.pdf) for the historical TF-IDF experiment.
6. [`docs/error_analysis.md`](docs/error_analysis.md) for model failure patterns.
7. [`docs/data_manifest.md`](docs/data_manifest.md) for data and artifact limitations.

## What the project does not claim

This project does not claim to predict the “true” intrinsic difficulty of every problem. Codeforces ratings are platform labels, and solved counts are affected by exposure.

The text experiments also do not mean the model understands algorithms. Statement-structure features and TF-IDF help because they capture broad patterns in statements, not because they solve the problems or understand the key insight.

The best way to read the project is as a careful, reproducible study of what different public signals can and cannot tell us about Codeforces difficulty.
