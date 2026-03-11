Reviewer’s concern: Open-ended evaluation with cosine similarity and a single 0.55 threshold is not sound (cosine ≠ entailment, model not specified, threshold ad hoc).

Your response:

Entailment vs similarity
We agree that cosine measures topical similarity, not correctness. We now use NLI-based correctness as the main measure: the generated answer must entail the gold answer (BART-MNLI). Cosine is only a secondary check.

Models
We use the same models as in the main experiment: BART-MNLI for NLI and paraphrase-mpnet-base-v2 for cosine. Both are named in the paper (Section 5.3 and 6.8).

No single threshold
We no longer use one “0.55” cutoff. We report accuracy at several NLI thresholds (0.15–0.5) and several cosine thresholds (0.5–0.7) as a sensitivity analysis. For example, at NLI τ = 0.5 we get 38.6% accuracy and mean NLI score 0.40; cosine results are given at 0.5–0.7 for transparency.

Bottom line
Open-ended correctness is now defined by entailment (NLI), with the same NLI and embedding models as in the main experiment, and multiple thresholds instead of a single ad hoc value.
