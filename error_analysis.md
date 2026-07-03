# Error Analysis

After getting the main results, I checked the kinds of cases where the model can still make large mistakes. This was useful because the overall MAE numbers do not show what the model actually misunderstands.

The main pattern is simple: the model can capture broad signals, but it still does not understand the core trick or reasoning step of a programming problem.

## Case 1: Short statements with hard ideas

Some difficult problems have short statements, few samples, and simple-looking input formats. These can be hard for statement-length features because the problem does not look complex on the surface.

A short statement can still hide a difficult observation. For example, the solution may require a clever invariant, a non-obvious greedy proof, a graph transformation, or a dynamic programming state that is not directly visible from the wording.

This is a weakness of both statement-structure features and TF-IDF. The text may contain common words, but the actual difficulty comes from the idea needed to solve the problem.

## Case 2: Long statements that are not actually very hard

The opposite problem also happens. Some statements are long because they contain a story, many variables, several cases, or a detailed input format. The model may treat this as a sign of difficulty, even when the algorithmic idea is not extremely advanced.

This is why statement length, sample count, number count, and section length are useful but limited. They can describe how complicated the statement looks, but they cannot always tell whether the required algorithm is hard.

## Case 3: Tag ambiguity

Official tags are helpful, but the same tag can appear across a wide difficulty range.

For example, a `binary search` problem can be easy if the monotonic condition is direct. It can be much harder if binary search is combined with dynamic programming, graph reasoning, or a difficult feasibility check.

The same issue appears with tags like `math`, `graphs`, `dp`, and `greedy`. A tag tells the model the rough topic, but not how hidden the observation is or how many ideas must be combined.

This is one reason tag-only prediction is weak, and why even metadata plus text features still makes mistakes.

## Case 4: Solved count and exposure

Solved count is one of the strongest signals in the project, but it is not a pure difficulty measure.

A problem can have many solves because it is old, famous, included in practice lists, from a popular contest, or commonly recommended to beginners. Another problem can have fewer solves because it is newer, less visible, or from a less-used contest, even if its official rating is not extremely high.

This means the post-publication models can learn exposure patterns as well as difficulty. That is useful for prediction, but it is not the same as understanding the problem.

This is why the project keeps post-publication prediction and cold-start prediction separate.

## Case 5: Text features help, but only as extra signals

The statement feature experiments improved cold-start prediction, but they did not solve the whole problem.

Statement-structure features can tell whether a problem has a long statement, many examples, many numeric tokens, or certain visible keywords. TF-IDF can capture words and phrases such as graph, tree, query, array, substring, probability, or shortest path.

However, neither feature type can reliably identify the main solution insight. They are useful because they add weak but complementary signals to metadata, not because they fully represent algorithmic difficulty.

## What this means

The model is good at learning broad patterns:

- Later contest indices are often harder.
- Some tags are associated with higher ratings.
- Solved behavior is strongly related to rating after publication.
- Statement structure and wording add extra cold-start signal.

But it still misses many problem-specific reasons for difficulty:

- hidden observations
- tricky proofs
- unusual reductions
- implementation traps
- misleadingly simple statements
- exposure effects in solved counts

This is the main reason the project should be read as a difficulty-prediction study, not as a model that truly understands programming problems.
