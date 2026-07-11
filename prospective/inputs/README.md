# Prospective input contract

This directory will contain one immutable sanitized T0 feature CSV and one
capture sidecar per eligible contest after the v2 protocol is frozen.

The feature CSV is model input. It contains only:

1. `contest_id`
2. `index`
3. the 41 protocol-locked statement-structure features, in frozen order

The sidecar is audit evidence, not model input. It records the explicit
operator inputs, direct statement URLs, timestamps, request policy, fetch and
parse status, raw-page hashes, the exact CSV schema, row count, and CSV hash.

Capture files are never overwritten. A failed capture writes a failure
sidecar but no model CSV.
