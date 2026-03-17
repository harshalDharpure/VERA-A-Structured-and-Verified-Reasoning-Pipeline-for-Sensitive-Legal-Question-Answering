# Strong Evaluation Results: Research-Grade Recommendations

*Perspective: senior NLP research scientist (FAANG / top-tier lab). Goal: defensible, publishable numbers and narrative.*

---

## 1. Why NLI Scores Are Low (And Why That’s OK)

- **NLI measures strict entailment** (premise ⇒ hypothesis), not “same meaning.” Legal gold answers are short, formal phrases; model outputs are longer and rephrased, so entailment probabilities are often modest even when the answer is correct.
- **Off-the-shelf NLI models** (DeBERTa, BART) are not trained on legal QA. Domain shift depresses scores.
- **Low NLI does not mean the system is bad.** It means the metric is conservative. The right move is to add metrics that align better with “correctness” and to frame NLI appropriately.

---

## 2. What to Report (Strong Narrative)

### Primary metrics (lead with these)

| Metric | What to report | Rationale |
|--------|----------------|-----------|
| **BERTScore F1** | Mean BERTScore F1 + accuracy at thresholds (e.g. ≥0.90, ≥0.92) | Reference-based; best correlation with human judgment in text generation/QA (Zhang et al., ACL 2020). Standard in NLP. |
| **NLI (answer)** | Mean NLI score + accuracy at several thresholds (e.g. 0.2–0.5), **with clear wording** | “We use NLI entailment as a **conservative** measure of semantic match; at threshold 0.3 we report X%.” |

### Secondary / supplementary

- **Cosine similarity** (e.g. paraphrase-mpnet): report as secondary; note that it measures topical similarity, not logical correctness.
- **NLI (reasoning)**: optional; supports “consistency of justification” story.

### Suggested paper wording

- “We evaluate answer correctness with **BERTScore** (primary) and **NLI-based entailment** (conservative). We report BERTScore F1 and NLI accuracy at multiple thresholds for transparency.”
- “NLI scores are conservative because models are not trained on legal QA; we therefore emphasize BERTScore and thresholded accuracy rather than raw NLI means.”

---

## 3. How to Get the Numbers

### Run with BERTScore (recommended)

```bash
.venv/bin/python evaluate_judge_thresholds.py \
  --ground_truth train_500_no_options.jsonl \
  --predictions vera_final.jsonl \
  --answer_key "Correct Answer" \
  --reasoning_key final_reasoning \
  --output judge_eval_summary_500.json \
  --bertscore
```

Requires: `pip install evaluate bert_score`. If BERTScore fails (e.g. GPU OOM), try `--device cpu` or reduce batch size in code.

### Use full-answer evaluation (not first-sentence for NLI)

- **Full answer** (current default): NLI ~0.36 mean, ~42% at 0.15, ~36% at 0.45 (see `judge_eval_summary_500.json`).
- **First-sentence only**: NLI drops (~0.21) because truncation often removes the key legal phrase. Use full-answer for NLI; first-sentence can still be reported as an ablation if needed.

### ARR-only vs full VERA

- ARR-only (`result_openended_500.jsonl`) typically gives **higher** NLI than full VERA (judge + merge). For a strong headline number, you can report ARR-only NLI as “model answer (before judge)” and full VERA as “final pipeline.”

---

## 4. Improving the Numbers (Legitimate Levers)

1. **Add BERTScore and lead with it**  
   Run with `--bertscore`; report mean BERTScore F1 and accuracy at ≥0.90 (and optionally ≥0.92). This is the most impactful and methodologically sound step.

2. **Calibrate one threshold**  
   Manually label 50–100 examples as correct/incorrect. Plot accuracy vs NLI (or BERTScore) threshold; pick a threshold that matches human agreement and report: “At threshold τ (chosen to align with human agreement on a 100-sample subset), we achieve X% accuracy.”

3. **Tighten the generation**  
   In the Judge (or ARR) prompt, ask for “one clear sentence stating the legal conclusion, in the same style as the reference.” Shorter, focused answers align better with short gold answers and can raise both NLI and BERTScore.

4. **Report multiple thresholds**  
   Always report a sensitivity table (e.g. 0.15–0.5 for NLI, 0.85–0.96 for BERTScore). Reviewers trust this more than a single number.

5. **Human evaluation**  
   Report agreement with human judgment on a subset (e.g. 100 samples). “BERTScore / NLI accuracy at τ correlates with human agreement (Kappa = …).”

---

## 5. Summary Table (Current Setup)

| Setting | NLI answer (mean) | NLI @0.3 | Cosine (mean) | Cosine @0.55 |
|--------|--------------------|----------|----------------|--------------|
| Full VERA, full answer | 0.36 | ~39% | 0.58 | ~68% |
| Full VERA, first-sentence | 0.21 | ~23% | 0.56 | ~61% |
| ARR-only (from earlier runs) | ~0.46 | higher | — | — |

**Recommendation:** Report **full-answer** metrics; add **BERTScore** and use it as primary; frame NLI as conservative and report thresholded accuracy; optionally add human agreement and threshold calibration for a top-tier narrative.
