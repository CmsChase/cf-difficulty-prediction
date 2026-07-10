# Official rating snapshots

After the frozen 72-hour delay, store the exact reveal source as a CSV with only
`contest_id,index,official_rating`. The reveal ledger commits the CSV SHA-256
and copies the outcomes into its own append-only chain. Never place outcomes in
a T0 input or prediction file.
