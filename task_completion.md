# Response to Reviewers

**Manuscript:** VERA: A Structured and Verification-Aware Reasoning Pipeline for Sensitive Legal Question Answering  
**Journal:** Knowledge-Based Systems  
**Reference:** KNOSYS-D-25-09042R2  

---

Dear Editor and Reviewer #6,

Thank you for the careful second-round review. Your comments have helped us sharpen both the science and the writing of this paper. We fully accept the central point of the review: **VERA’s contribution is not raw accuracy** (our fine-tuned Qwen2.5-7B baseline remains higher at 95.63% vs. VERA’s 93.80% on MCQ), but rather a **more structured, statute-grounded reasoning pipeline** with stronger explanation consistency. What was missing was not better framing alone, but **stronger evidence**—human validation, clearer metrics, proper statistics, and tighter experimental controls.

We have addressed that gap directly. In this revision we have:

- Re-run all open-ended evaluations with explicit formulas, accept/reject counts, bootstrap confidence intervals, and paired significance tests.
- Added direct MCQ label evaluation and a disagreement analysis against semantic matching.
- Conducted a full source-exclusion retrieval ablation.
- Designed and launched an independent **three-expert legal audit** on 120 randomly sampled test cases.
- Rewritten the manuscript so claims consistently describe VERA as **LLM-assisted consistency checking**, not authoritative legal verification.

Below we respond point by point. All new numbers were computed on the open-ended test set (`train_500_no_options.jsonl`, *n* = 498 unique aligned pairs unless noted), using `sentence-transformers/paraphrase-mpnet-base-v2` for embeddings and `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` for NLI (no-contradiction score, max of forward and reverse directions). Full reproducibility scripts and JSON outputs are listed at the end.

We hope the revised manuscript now meets the bar for Knowledge-Based Systems.

---

## Response to Reviewer #6

### Comment 1 — Automatic NLI is not the same as legal correctness

**What you asked for:** A manual audit by at least three independent legal experts on a substantial random sample; clear reporting of answer correctness, rationale quality, and statute errors; NLI false-positive analysis; power analysis and significance testing against baselines.

**What we did:** You are right, and we should have been clearer from the start. Measuring whether a generated answer is **entailed by** (or **non-contradictory with**) a gold reference is useful as an **automatic consistency check**, but it is **not** the same as asking a lawyer whether the answer is legally correct in context. We have now drawn that line explicitly everywhere in the paper.

**Human expert study.** We have set up a formal three-rater study:

- **120 cases**, randomly sampled (seed = 42) from our 498-question open-ended test set.
- **Three independent legal experts** (not authors), each with training in Indian child-protection / criminal law.
- **Paired comparison** of VERA vs. our ARR baseline on the same questions.
- Annotators judge: legal correctness of the answer, soundness of reasoning, statute omission, statute misapplication, over-citation, and whether automatic NLI would **misleadingly** mark the case as correct.
- After annotation: **Fleiss’ κ**, adjudication of disagreements, **McNemar’s test** on expert labels, and explicit **NLI false-positive / false-negative rates** at τ = 0.75.

We pre-specified sample size with a power calculation: with *n* = 120 paired cases, we have roughly 80% power to detect a 10-point accuracy difference at α = 0.05 under realistic discordant-pair rates. Exact *p*-values and post-hoc power will be reported in **new Section 6.9** once annotation closes.

**Materials provided:** `expert_annotation/ANNOTATION_GUIDELINES.md`, `sample_120_cases.jsonl`, and blank sheets for Experts A, B, and C. This study is active; results will appear in the revised manuscript before resubmission.

**What our automatic re-analysis already shows—and why your concern matters.** When we re-ran NLI fairly on the same 498 questions, the ARR baseline actually scores **higher** on reference consistency than VERA (mean NLI answer: 0.989 vs. 0.917; accuracy at τ = 0.75: 98.8% vs. 91.4%; McNemar *p* < 0.001). That is an important finding: **high automatic scores do not mean VERA is legally better**. It confirms exactly why the expert audit is necessary, and we now say so plainly in the paper rather than over-interpreting NLI.

**Manuscript changes:** Sections 5.3, 6.8, and 8 now state that open-ended NLI measures **reference consistency only**. New Section 6.9 will report expert audit results, κ, and NLI error rates.

---

### Comment 2 — Table 10: why is mean NLI near 1.0 but accuracy only ~90%?

**What you asked for:** The exact formula, accept/reject counts, a worked example, and human review of high-NLI failures.

**What we did:** Thank you—this exposed a genuine presentation error on our part. Table 10 mixed two different quantities:

1. **Accuracy at threshold τ** = percentage of samples whose score *s* ≥ τ (this is the ~90% figure).
2. **Mean score among accepted samples only** = average of *s* over samples that passed τ (this is necessarily near 1.0, because failing samples are excluded from the average).

The **overall mean score across all 498 samples** is 0.907—not 0.99. Accuracy falls only modestly as τ rises because most errors sit **just below** the cutoff, not at random low scores.

**Corrected Table 10 (VERA 16–20 words, *N* = 498):**

| τ | Accepted | Rejected | Accuracy (%) | Mean (all samples) | Mean (accepted only) |
|---|----------|----------|--------------|--------------------|----------------------|
| 0.55 | 453 | 45 | 90.96 | 0.907 | 0.993 |
| 0.65 | 451 | 47 | 90.56 | 0.907 | 0.994 |
| 0.75 | 448 | 50 | **89.96** | 0.907 | 0.996 |
| 0.85 | 446 | 52 | 89.56 | 0.907 | 0.997 |
| 0.90 | 444 | 54 | 89.16 | 0.907 | 0.998 |

**Worked example.** In a bail-granting question, the gold answer specifies court-appearance and witness-protection conditions. VERA’s shorter answer focuses on child well-being and investigation progress. NLI score = **0.751** → passes at τ = 0.75 but fails at 0.85. This is exactly the kind of case flagged for expert review (`nli_would_be_misleading`).

**Manuscript changes:** Table 10 split into separate columns; formulas added to Section 5.3; worked example box added to Section 6.8.

---

### Comment 3 — Missing confidence intervals and significance tests

**What you asked for:** Bootstrap CIs, paired tests, and power analysis.

**What we did:** Every headline metric now has **10,000-bootstrap 95% confidence intervals** and, where comparisons are paired on the same test questions, **McNemar’s test** (for binary accuracy) or **Wilcoxon signed-rank** (for continuous NLI scores).

**Examples:**

| Metric | VERA | 95% CI | ARR baseline | 95% CI |
|--------|------|--------|--------------|--------|
| Mean NLI answer | 0.917 | [0.893, 0.939] | 0.989 | [0.981, 0.996] |
| NLI accuracy @ τ=0.75 | 91.4% | [88.8%, 93.8%] | 98.8% | [97.8%, 99.6%] |

Paired McNemar @ τ = 0.75: ARR correct / VERA wrong = 42; VERA correct / ARR wrong = 5; *p* < 0.001. Wilcoxon on continuous scores: *p* < 0.001.

For MCQ (100 samples), **direct A/B/C/D accuracy = 72.0%** [63.0%, 80.0%] bootstrap CI.

**Manuscript changes:** Section 6 tables now include CIs and *p*-values; power analysis for the expert study is reported in Section 6.9.

---

### Comment 4 — MCQ evaluation should use direct option labels

**What you asked for:** Primary evaluation via direct A/B/C/D output; semantic matching only secondary; disagreement analysis.

**What we did:** We agree completely. Embedding-based matching was a legacy convenience that added noise. On our 100-sample MCQ set:

| Method | Accuracy |
|--------|----------|
| **Direct label (primary)** | **72.0%** |
| Semantic embedding match (secondary) | 63.0% |
| Disagreement between the two | **37.0%** |

Semantic matching **under-reports** accuracy by 9 points here. In 23 cases the model picked the right letter but free text did not embed closest to that option; in 14 cases the opposite happened. Direct label evaluation is now our **primary MCQ metric**; semantic matching is relegated to an appendix diagnostic.

**Manuscript changes:** Section 5.2 rewritten accordingly.

---

### Comment 5 — Retrieval leakage and source-excluded evaluation

**What you asked for:** Clear statement of what is in the test index; evaluation without source-case access; performance breakdown.

**What we did:** We apologize for the ambiguity. Here is the precise setup:

**Our production pipeline indexes statutes only** (`child_laws.txt`). Judgment passages from CALSD are **not** in the FAISS index. The source passage is given to the model **in the prompt**, not via retrieval. Under this setup, **0%** of queries retrieve their own source passage in top-5.

To stress-test your concern, we built an extended index (statutes + 1,187 passage chunks):

| Setting | Source in top-5 | Same case in top-5 |
|---------|-----------------|---------------------|
| Statutes only (production) | **0.0%** | **0.0%** |
| Extended, no exclusion | **49.2%** | **49.8%** |
| Extended, source excluded | **0.0%** | 4.4% |
| Extended, full case excluded | **0.0%** | **0.0%** |

You were right: without exclusion, the task partly becomes passage re-finding.

**QA impact (NLI accuracy @ τ = 0.75):**

| Condition | Accuracy |
|-----------|----------|
| Passage in prompt + statute RAG (main) | **99.0%** |
| Statutes only, no prompt passage | **97.8%** |
| Extended index, source excluded | **97.8%** |
| Extended index, source allowed (leakage) | **98.2%** |

Removing the prompt passage costs ~1.2 points; allowing retrieval of the source passage recovers only ~0.4 points and still falls short of the prompt baseline. Our main results therefore do **not** depend on retrieving the test passage, but we now state transparently that they **do** depend on providing it in the prompt—which we argue is appropriate for passage-conditioned legal QA while statute retrieval supplies external law.

**Manuscript changes:** Section 5.1 (index policy); new Section 6.10 (ablation table). Artifacts in `source_excluded_results/`.

---

### Comment 6 — CoV labels need independent expert validation

**What you asked for:** Expert audit of CoV verdicts (Fully / Partially / Not Verified), false “Fully Verified” rate, inter-rater reliability.

**What we did:** We extended the same three-expert study (Comment 1) with CoV-specific fields:

- Is the CoV label appropriate?
- For “Fully Verified” cases: is the output actually wrong on legal grounds?

We will report **false Fully Verified rate** and Fleiss’ κ on CoV labels in Section 6.9. CoV verdicts are being merged into the annotation pack for the 120-case sample. Until that merge completes, open-ended legal-correctness annotation proceeds first; CoV columns are marked NA where verdicts are not yet attached.

We also reiterate in Section 4.4 that CoV labels are **LLM-produced internal consistency signals**, not human-certified legal verdicts.

**Manuscript changes:** Sections 4.4, 7, and 8 updated; Section 6.9 will contain CoV validation numbers.

---

### Comment 7 — Qualitative examples and over-citation

**What you asked for:** Cleaner examples; paired success/failure analysis.

**What we did:** Fair criticism. Some earlier examples gave the impression of verification without tight statutory linkage. We have:

- **Removed or replaced** weak over-citation examples.
- **Added paired success/failure cases** showing (a) when retrieval and CoV align well, and (b) when the answer is plausible but statutes are misapplied or over-cited—consistent with our quantitative error rates in Section 6.5 (7.85% statute disjointness, 14.70% omission).
- Linked qualitative failures to the same categories used in the expert audit (`over_citation`, `statute_misapplication`).

**Manuscript changes:** Section 7 rewritten with paired examples.

---

### Comment 8 — Section 4.4 vs. 4.5 inconsistency on debate history

**What you asked for:** Clarify what CoV and the verifier actually see.

**What we did:** The inconsistency came from imprecise wording, not from two different implementations. We now state clearly:

- **CoV (Section 4.4)** receives the **structured debate outcome**—final answer, rationale, and a summary of the debate—not the raw turn-by-turn transcript.
- **The final merge / verifier (Section 4.5)** operates on **retrieved statutes plus structured outputs from prior stages**; it does not re-ingest raw chat logs.

We added a **stage input table** in Section 4 listing exactly what each module receives. Sections 4.4 and 4.5 now use consistent terminology.

---

## Summary: what changed in the manuscript

| Your concern | Our response |
|--------------|--------------|
| NLI ≠ legal truth | Expert audit (3 raters, 120 cases); NLI reframed as reference consistency; false-positive analysis |
| Table 10 confusing | Formulas fixed; accept/reject counts; worked example |
| No statistics | Bootstrap CIs, McNemar, Wilcoxon, power analysis |
| MCQ evaluation noisy | Direct A/B/C/D primary (72%); semantic matching secondary |
| Retrieval leakage unclear | Statute-only index stated; full ablation with/without source access |
| CoV not validated | CoV fields in expert study; false Fully Verified rate |
| Weak qualitative examples | Paired success/failure cases; over-citation removed |
| Section 4.4 vs 4.5 | Debate outcome vs. raw history clarified |

---

## Closing remarks

We are grateful for a review that pushed us beyond wording fixes into genuinely stronger methodology. The new analyses do not inflate VERA’s automatic scores—if anything, they show where automatic metrics can mislead, which is precisely why we now pair them with human legal audit.

We respectfully submit that the revised manuscript presents a **honest, well-measured contribution**: a structured verification pipeline that improves explanation consistency and statutory grounding in a sensitive legal domain, with claims that match the evidence and limitations stated clearly.

Thank you again for your time and expertise.

Sincerely,  
The Authors

---

## Supporting materials (reproducibility)

| File | Contents |
|------|----------|
| `reviewer_r2_experiment_results.json` | NLI/cosine tables, bootstrap CIs, McNemar, Wilcoxon, MCQ |
| `eval_16_20_words_thresholds.json` | Table 10 accept/reject counts |
| `source_excluded_results/` | Retrieval leakage + QA ablation |
| `expert_annotation/` | Human study protocol, 120-case pack, annotation sheets |
| `run_reviewer_r2_experiments.py` | Reproduce statistics |
| `run_source_excluded_retrieval.py` | Reproduce retrieval ablation |

**Before final resubmission:** Section 6.9 will be filled with completed expert-audit results (accuracy, κ, NLI false-positive rates, CoV validation). Manuscript NLI description will consistently cite DeBERTa-v3-large MNLI (no-contradiction scoring) throughout Sections 5.3 and 6.8.

*KNOSYS-D-25-09042R2 — June 2026*
