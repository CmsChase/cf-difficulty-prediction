# Ledger workflow

Prediction events and rating reveals live in different JSON Lines files:

- `predictions.jsonl` is written before ratings are queried or joined.
- `reveals.jsonl` is written later and refers back to immutable prediction-event
  hashes.

Both files are append-only SHA-256 chains. Run the verifier before and after
every append. Commit and push `predictions.jsonl` before running the reveal
command. Never squash away published prediction-ledger history.

Lock one contest no later than 30 minutes after its scheduled start:

```powershell
python -m cf_diff.prospective_model predict `
  --protocol configs/prospective_protocol_v1.json `
  --model prospective/model_bundle_v1.json `
  --manifest prospective/model_freeze_manifest_v1.json `
  --input prospective/inputs/CONTEST_t0_features.csv `
  --output prospective/predictions/CONTEST_predictions.csv `
  --contest-start-utc 2026-07-12T12:00:00Z

python -m cf_diff.prospective_ledger record `
  --predictions prospective/predictions/CONTEST_predictions.csv `
  --protocol configs/prospective_protocol_v1.json `
  --manifest prospective/model_freeze_manifest_v1.json `
  --ledger prospective/ledger/predictions.jsonl

python -m cf_diff.prospective_ledger verify
```

Then commit and push the sanitized input, prediction CSV, and ledger append.
Do not amend, rebase away, or force-push a public prediction commitment.

If an eligible contest is missed, record it explicitly:

```powershell
python -m cf_diff.prospective_ledger missed `
  --contest-id CONTEST `
  --contest-start-utc 2026-07-12T12:00:00Z `
  --reason "documented operational reason" `
  --protocol configs/prospective_protocol_v1.json `
  --ledger prospective/ledger/predictions.jsonl
```

After the frozen 72-hour delay, prepare a CSV with exactly
`contest_id,index,official_rating`, then append outcomes to the separate chain:

```powershell
python -m cf_diff.prospective_ledger reveal `
  --ratings prospective/outcomes/CONTEST_official_ratings.csv `
  --protocol configs/prospective_protocol_v1.json `
  --prediction-ledger prospective/ledger/predictions.jsonl `
  --reveal-ledger prospective/ledger/reveals.jsonl
```

GitHub CI verifies both hash chains and requires every previously committed
ledger byte to remain an exact prefix of the proposed ledger.
