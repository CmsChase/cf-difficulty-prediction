# Prospective v1 retirement record

Status: **superseded before enrollment on 2026-07-11**

The proposed `cf-difficulty-prospective-v1` protocol was published on the
historical staging branch, but it was not used to enroll a contest or issue a
prospective prediction. Its planned cohort start was 2026-07-12 00:00:00 UTC.

The original records remain unchanged and publicly inspectable:

- archived snapshot tag: `archive/research-governance-2026-07-10`;
- [v1 protocol at commit `79fa9ed`](https://github.com/CmsChase/cf-difficulty-prediction/blob/79fa9ed/configs/prospective_protocol_v1.json)
- [v1 model bundle at commit `79fa9ed`](https://github.com/CmsChase/cf-difficulty-prediction/blob/79fa9ed/prospective/model_bundle_v1.json)
- [v1 freeze manifest at commit `79fa9ed`](https://github.com/CmsChase/cf-difficulty-prediction/blob/79fa9ed/prospective/model_freeze_manifest_v1.json)

The v1 protocol file SHA-256 is
`2562c136df52f7e2e48e626109d51101f0c355c9b541936c7bcb564ab027c548`.
At retirement, the staging tree contained no prediction events, outcome
events, or populated prediction files; the relevant directories contained
only placeholder documentation.

## Why it was superseded

The research question and confirmatory estimand remain useful. The operational
workflow did not yet meet the intended isolation standard:

- there was no dedicated command that generated a T0 feature file directly
  from explicit contest and problem keys;
- the predictor inspected forbidden columns only after loading the full input
  table;
- multi-contest inputs and overwrite attempts were not rejected;
- capture provenance was not cryptographically linked to each prediction.

Starting the cohort before closing those gaps would make the prospective claim
harder to defend. The replacement v2 protocol therefore remains a draft until
the capture, prediction, ledger, and dry-run gates are complete.

This record does not edit or reinterpret the frozen v1 file. It documents a
pre-enrollment replacement in a later commit.
