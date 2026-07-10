# Frozen prediction batches

Each eligible contest gets one CSV produced by `cf_diff.prospective_model`.
The file contains primary and index-only comparator predictions, feature-row
hashes, model/protocol identifiers, and timestamps—never official ratings or
errors. Commit it together with the corresponding append to
`ledger/predictions.jsonl`.
