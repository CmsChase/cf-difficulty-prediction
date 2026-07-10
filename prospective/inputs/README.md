# Sanitized T0 inputs

Future contest feature files belong here. They may contain `contest_id`,
`index`, statement-parser audit fields, and the frozen numeric statement
features. They must not contain rating, points, tags, solved counts, submission
or acceptance statistics, or manually assigned difficulty labels.

The predictor derives `index_rank` and `index_number` from `index` and ignores
caller-supplied versions of those two fields.
