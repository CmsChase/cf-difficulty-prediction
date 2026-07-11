# Prospective evaluation workspace

The prospective workflow is under pre-enrollment construction.

- Current machine-readable protocol:
  [`configs/prospective_protocol_v2.json`](../configs/prospective_protocol_v2.json)
  (**draft**)
- Human-readable draft:
  [`docs/PROSPECTIVE_PROTOCOL_V2_DRAFT.md`](../docs/PROSPECTIVE_PROTOCOL_V2_DRAFT.md)
- Prior v1 disposition:
  [`docs/PROSPECTIVE_V1_RETIREMENT.md`](../docs/PROSPECTIVE_V1_RETIREMENT.md)

No prospective cohort is active and no v2 model is frozen yet. The capture and
prediction commands intentionally reject the draft protocol.

## Planned command chain after freeze

Capture a complete contest directly from public statement pages:

```powershell
python -m cf_diff.prospective_input `
  --protocol configs/prospective_protocol_v2.json `
  --contest-id 3000 `
  --indices A B C D `
  --contest-start-utc 2026-08-15T12:00:00Z `
  --output prospective/inputs/3000_t0_features.csv `
  --sidecar prospective/inputs/3000_t0_features.capture.json `
  --raw-dir prospective/raw/3000
```

Run only the verified frozen model and bind the prediction to that capture:

```powershell
python -m cf_diff.prospective_model predict `
  --protocol configs/prospective_protocol_v2.json `
  --model prospective/model_bundle_v2.json `
  --manifest prospective/model_freeze_manifest_v2.json `
  --input prospective/inputs/3000_t0_features.csv `
  --capture-sidecar prospective/inputs/3000_t0_features.capture.json `
  --contest-start-utc 2026-08-15T12:00:00Z `
  --output prospective/predictions/3000_predictions.csv
```

Until the ledger change is merged, these commands are for synthetic dry runs
only. Do not use them to enroll a real contest.

Raw HTML is retained locally for capture audit and ignored by Git. Sanitized
feature inputs, sidecars, frozen model artifacts, and later ledger events are
the public evidence.
