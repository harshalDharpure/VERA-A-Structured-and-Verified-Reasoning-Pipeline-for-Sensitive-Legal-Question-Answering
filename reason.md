# Open-ended evaluation methodology (for paper / reviewer response)

## 1. Why not cosine similarity alone?

- **Cosine similarity** between sentence embeddings measures **topical/lexical proximity**, not semantic correctness or entailment.
- Example: *"I love books"* vs *"I do not love books"* can have high cosine similarity (shared words) but opposite meaning. A single threshold (e.g. 0.55) is not a sound correctness criterion.
- We therefore treat cosine as **secondary**: we report it at multiple thresholds for transparency and specify the embedding model (see below), but we do **not** use it as the sole or primary measure of correctness.

## 2. Primary measure: NLI (Natural Language Inference) entailment

- **NLI** models (we use **BART-MNLI**, `facebook/bart-large-mnli`) take a **premise** and a **hypothesis** and output three-way probabilities: **entailment**, **neutral**, **contradiction**.
- We use **P(entailment)** as the score: *premise entails hypothesis*.
- **Answer-level NLI (primary for correctness)**  
  - **Premise** = generated answer  
  - **Hypothesis** = gold (reference) answer  
  - If the model’s answer **entails** the gold answer, we consider it consistent with the reference. If it **contradicts** the gold (e.g. “I do not love books” vs “I love books”), entailment probability is low.  
  - This directly addresses the reviewer’s concern: negation and contradiction yield low scores, unlike with cosine.

- **Reasoning-level NLI (supplementary)**  
  - **Premise** = generated reasoning  
  - **Hypothesis** = gold reasoning  
  - Measures whether the model’s justification is consistent with (entails) the reference justification.

## 3. What is an “NLI threshold”?

- For each pair we get a single score in **[0, 1]** (probability of entailment).
- **Threshold** τ: we count the answer as **correct** for that pair if **NLI_score ≥ τ**.
  - **τ = 0.5**: at least 50% entailment probability required (conservative).
  - **Higher τ** (e.g. 0.5, 0.6): stricter → fewer counted correct, higher precision.
  - **Lower τ** (e.g. 0.2, 0.3): more lenient → more counted correct, higher recall.
- We **do not** pick one “empirically chosen” threshold. We report **accuracy at multiple thresholds** (sensitivity analysis) so readers see the full curve and can interpret results without relying on a single ad hoc value.

## 4. How are the thresholds chosen?

- **No single threshold is calibrated as “the” correctness rule.** We report:
  - **Cosine:** 0.5, 0.55, 0.6, 0.65, 0.7 (for comparison and transparency).
  - **NLI:** 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5 (and optionally more).
- **Rationale:**
  - **Sensitivity analysis:** Show how accuracy changes with threshold so the method is not tied to one number.
  - **Conservative reference:** For NLI, τ = 0.5 is a natural reference (majority entailment) but we do not claim it is “calibrated”; we present the full curve.
  - **Reproducibility:** All models and thresholds are specified; others can replicate or use different thresholds.

Optional (if you have a small labeled dev set): you can add a sentence that you performed threshold sensitivity over the above grid and, if applicable, chose a reference threshold (e.g. 0.5) for reporting after inspecting the curve, without claiming full calibration.

## 5. Models used (to state in the paper)

- **Embedding (cosine):** `sentence-transformers/paraphrase-mpnet-base-v2` (so cosine is clearly specified and reproducible).
- **NLI:** `facebook/bart-large-mnli`. Premise/hypothesis order: **premise = generated**, **hypothesis = gold**; we use the **entailment** probability (index 2 in MNLI logits).

## 6. What to report in the paper

- **Primary:** Accuracy at multiple **NLI (answer-level)** thresholds, and optionally mean NLI answer score. Emphasize that correctness is based on **entailment**, not similarity.
- **Secondary:** Cosine similarity reported at multiple thresholds, with embedding model named; clarify that it is used only as a supplementary/exploratory measure.
- **Supplementary:** NLI on reasoning (optional), with the same multi-threshold reporting.
- **Methodology sentence (example):**  
  *“For open-ended evaluation we use NLI-based correctness: the model’s answer is counted correct at threshold τ if the probability that it entails the gold answer (BART-MNLI) is ≥ τ. We report accuracy at multiple τ (sensitivity analysis) and do not rely on a single empirically chosen threshold. Cosine similarity (paraphrase-mpnet-base-v2) is reported at several thresholds for transparency but is not used as the primary correctness criterion.”*
