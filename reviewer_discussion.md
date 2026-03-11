# Response to Reviewers
## KNOSYS-D-25-09042R1 — VERA: A Structured and Verified Reasoning Pipeline for Sensitive Legal Question Answering

---

## Editor comment: Revise

**Response:** We thank the Editor for the opportunity to revise. We have addressed each reviewer comment below and have revised the manuscript, checklist, and supplementary materials accordingly. All references, tables, and figures are cited in the text; figures are provided in editable format at 300 dpi where applicable.

---

## Reviewer #5

### Comment 1 — Trade-off between multi-stage verification cost and perceived utility

**Reviewer’s concern:** Table 8 shows that the VERA pipeline increases inference time by approximately 200% (from 13s to 40s) compared to the base ARR stage, while Table 5 shows that final accuracy does not improve significantly (and in some cases slightly decreases) relative to a fine-tuned single-model baseline. The reviewer asks whether the improvement in "explanation logic" (NLI scores) justifies the longer wait times for real-world users, and requests a discussion of the trade-off between the cost of multi-model verification and its impact on user trust in legal aid contexts.

**Our response:** We agree that this trade-off should be made explicit. The main benefit of the full pipeline is not a large gain in raw accuracy (which is already high for fine-tuned baselines on CALSD) but rather a substantial gain in **explanation consistency and statutory grounding**, as reflected in the NLI entailment scores (e.g. 0.7984 for VERA vs. 0.66–0.67 for baselines in Table 5). In legal aid contexts, trust often depends on whether the system's *reasoning* is aligned with the retrieved law and with human expectations of legal justification, not only on whether the chosen option is correct. We have therefore added a short subsection (or paragraph) in the manuscript that:

- States clearly that the full pipeline trades **inference time and compute** (Table 8: ~13s for ARR vs. ~40s for full VERA) for **explanation quality** (NLI) and **consistency with retrieved context**, rather than for a large accuracy gain over fine-tuned baselines.
- Discusses **when the full pipeline is justified**: e.g. high-stakes or ambiguous cases where statutory grounding and auditability matter more than latency; and notes that for simple or low-ambiguity cases the full sequence may be unnecessary (consistent with our existing "Limitations" discussion on adaptive routing).
- Acknowledges that we have not yet measured **perceived utility** or **user trust** in a user study; we state this as a limitation and a direction for future work (e.g. deployment studies with legal aid workers or experts).

**Changes in the manuscript:** We have added/revised text in Section 6 (e.g. after Table 8 or in a short "Inference cost and utility" paragraph) and, if needed, in Section 8 (Limitations) to reflect the above. The revised wording avoids overclaiming that the extra cost always translates to proportionally higher user trust and instead frames the pipeline as a tool that prioritises explanation quality and verifiability when those are needed.

---

### Comment 2 — Textual Faithfulness vs. Judicial Truth / "logical self-verification trap"

**Reviewer’s concern:** The framework successfully isolates models to break circular validation, but a concern remains about a "logical self-verification trap": underlying 8B/7B models share similar pre-training and scale-related limitations and may converge on "plausible but doctrinally incorrect" logic. The reviewer asks the authors to clarify that VERA ensures **Textual Faithfulness** (alignment with provided evidence) rather than **absolute Doctrinal Correctness** (deep legal reasoning).

**Our response:** We accept this distinction and have made it explicit in the manuscript. VERA's Chain-of-Verification (CoV) and VERA merge operate only over the **retrieved context C** and the structured outputs from prior stages. The verifier checks whether the answer and rationale are **consistent with the provided passages and debate outcome**, not whether they are doctrinally correct in an absolute sense. Our own Section 6.5 (Statute Misapplication and Doctrinal Fallacy Analysis) already shows that a non-trivial proportion of outputs still exhibit statute disjointness (7.85%) and statutory omission (14.70%), including cases where the final answer is correct but the legal grounding is wrong. This supports the reviewer's point: we improve **alignment with the given text** and **factual consistency** (e.g. NLI with gold rationales), but we do **not** guarantee correct application of law in the sense of judicial or doctrinal truth.

**Changes in the manuscript:** We have added a clear statement (in Section 4.4 and/or in Section 8 (Limitations)) to the following effect:

- **VERA ensures textual faithfulness:** alignment of the answer and rationale with the **retrieved legal context** and with the structured debate/verification outputs. It does **not** guarantee **doctrinal correctness** or deep judicial reasoning; models may still produce plausible but doctrinally incorrect logic, as evidenced by the residual statute misapplication and omission rates reported in Section 6.5. We have toned any remaining language that could be read as claiming "absolute" or "judicial" correctness.

This clarifies the system boundary and addresses the "self-verification trap" concern without undermining the contribution (improved consistency, auditability, and statutory grounding relative to single-model baselines).

---

## Reviewer #6

### Comment 1 — Explainability vs. plausibility: "who verifies?" and strength of claims

**Reviewer’s concern:** The response reframed "explainability" as "verified legal faithfulness" via CoV, but the core issue remains: **who verifies** that the explanation is faithful rather than merely plausible? The "verification" is produced by an LLM-based pipeline. That may improve consistency with retrieved text but does not establish faithfulness in the interpretability sense, because the verifier is not a ground-truth authority and remains susceptible to hallucination and LLM judge failures. In a high-stakes legal setting, claiming "faithful, verified explainability" without human expert validation is unsubstantiated. The manuscript itself reports residual doctrinal errors (e.g. statute misapplication, statutory omission), including cases where the answer is correct but the legal grounding is wrong, which contradicts any implication of reliably "fully verified" legal reasoning. The reviewer requests: (a) toning down claims to something like "LLM-assisted consistency checking" unless the stronger claim is supported by external validation; (b) human expert validation of the verifier (correctness of CoV labels and of corrected answers/rationales; inter-rater reliability; verifier error rates)—not only "Case 1" expert evaluation; (c) clarity on what is actually deterministic (e.g. only the final argmax over scores vs. the LLM-mediated legal judgment that produces those scores).

**Our response:** We thank the reviewer for this important methodological point. We have revised the manuscript and our claims as follows.

**(a) Toning down claims**  
We have replaced or qualified phrasing that implies **guaranteed** "faithful" or "verified" explainability. Where appropriate we now use formulations such as **"LLM-assisted consistency checking"** or **"consistency with retrieved context"** and reserve "verified" for the **internal** pipeline output (e.g. "Fully/Partially/Not Verified" as labels produced by the CoV module), with an explicit caveat that these labels are **not** validated by human experts. We have also tightened the Abstract and Conclusion so they do not imply that the system delivers human-validated "verified explainability" in the sense of a ground-truth authority.

**(b) Human expert validation of the verifier**  
We acknowledge that a full study in which independent legal experts evaluate (i) the correctness of CoV verification labels (Fully/Partially/Not Verified) and (ii) the correctness of the verifier's corrected answers/rationales—with inter-rater reliability and error rates (e.g. false Fully-Verified, missed errors)—would be necessary to support a stronger claim. Such a study was not conducted in the current work; our expert evaluation (Case Study 1 and appendix) assesses **output quality** of selected cases, not the **accuracy of the CoV verifier itself**. We have therefore:

- Stated in the Limitations (Section 8) that the CoV labels and verifier outputs have **not** been validated by independent legal experts and that the pipeline provides LLM-assisted consistency checking rather than human-verified faithfulness.
- Added to Future Work the need for **human expert validation of the verifier**: e.g. expert annotation of a sample of CoV verdicts and corrected answers, with inter-rater reliability and reported error rates, before claiming "verified explainability" in the strong sense.

**(c) What is deterministic**  
We have clarified in Section 4.4.2 (and, if needed, in a short footnote or parenthesis):

- **Deterministic** refers to the **final answer selection** when the answer is corrected: given the verifier's **option-wise support assessments** (Supported / Not Supported), the corrected answer is chosen by a deterministic rule (e.g. arg max over options marked as supported, or an equivalent rule stated in the paper).  
- The **verification verdict** (Fully / Partially / Not Verified) and the **option-wise support assessments** are **produced by an LLM** (the verifier model). They are therefore **not** deterministic and are subject to the usual limitations of LLM-based judging (hallucination, inconsistency, etc.). We have removed or qualified any wording that could imply that the *entire* verification process is deterministic; only the **mapping from verifier outputs to the final corrected answer** is deterministic.

**Changes in the manuscript:**  
Revisions have been made in: Abstract; Section 4.4 (CoV description and wording around "deterministic" and "verified"); Section 6.5 (doctrinal analysis, already consistent with our caveats); Section 7 (expert evaluation, clarifying that it does not validate the verifier); Section 8 (Limitations and Future Work). The goal is to align the narrative with what the system actually does (LLM-assisted consistency checking and improved alignment with retrieved context) and to avoid methodologically misleading claims about human-verified or ground-truth "faithful explainability" until such validation is provided.

---

### Comment 2 — Open-ended evaluation via cosine similarity and 0.55 threshold

**Reviewer’s concern:** The open-ended evaluation marked an answer correct if cosine similarity between sentence embeddings exceeded 0.55, with the threshold chosen "empirically" via manual inspection. The reviewer states that this is not an acceptable correctness measure because: (i) embedding/cosine similarity measures topical proximity, not entailment (e.g. negation can still yield high similarity); (ii) the embedding model was not clearly specified; (iii) the 0.55 threshold appears ad hoc without principled calibration, sensitivity analysis, or error analysis.

**Our response:** We agree with the reviewer and have revised the evaluation methodology and the manuscript accordingly.

**(i) Entailment vs. proximity**  
We now use **NLI-based correctness as the primary measure** for open-ended answers. We apply BART-MNLI (facebook/bart-large-mnli) with the **generated answer as premise** and the **gold (reference) answer as hypothesis**, and use **P(entailment)**. Thus, contradictions and negations receive low scores. **Cosine similarity** is retained only as a **secondary, exploratory** measure and is **not** used as the sole correctness criterion.

**(ii) Model specification**  
Both models are now explicitly stated in the paper: for NLI we use **facebook/bart-large-mnli**; for cosine we use **sentence-transformers/paraphrase-mpnet-base-v2** (same as in MCQ option matching, Section 5.1.1). This ensures full reproducibility.

**(iii) Threshold justification**  
We no longer rely on a single empirically chosen threshold. We report **accuracy at multiple NLI thresholds** (τ = 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5) and at **multiple cosine thresholds** (0.5, 0.55, 0.6, 0.65, 0.7) as a **sensitivity analysis**. Following prior NLI-based evaluation practice, we report accuracy across this range rather than selecting a single fixed cutoff, allowing readers to observe system behaviour under different strictness levels; we do **not** claim a single "calibrated" threshold.

**Changes in the manuscript:**  
A new **Section 5.3 (Open-Ended Answer Evaluation)** describes the above protocol, including the distinction between NLI (primary) and cosine (secondary) and the rationale for multiple thresholds. **Section 6.8** has been updated to report results under this protocol (NLI-based accuracy at several τ, mean NLI score, and cosine-based accuracy at several thresholds for transparency). The previous wording that relied on a single 0.55 cosine threshold has been removed.

---

## Summary of manuscript revisions

| Reviewer | Point | Main revisions |
|----------|--------|----------------|
| #5 | 1 | New/expanded discussion of inference cost vs. explanation utility; when full pipeline is justified; limitation on perceived utility/user trust. |
| #5 | 2 | Explicit distinction: VERA ensures **textual faithfulness** (alignment with retrieved context), not **doctrinal correctness**; reference to Section 6.5. |
| #6 | 1 | Claims toned down to "LLM-assisted consistency checking"; CoV labels and verifier outputs stated as not human-validated; "deterministic" limited to final answer selection; expert validation of verifier added to Future Work. |
| #6 | 2 | Open-ended evaluation: NLI primary, cosine secondary; both models named; multiple thresholds reported (sensitivity analysis); Section 5.3 and 6.8 revised. |

We believe these revisions address the reviewers' concerns in a methodologically sound way and hope the manuscript is now suitable for acceptance.

---

*End of response to reviewers*
