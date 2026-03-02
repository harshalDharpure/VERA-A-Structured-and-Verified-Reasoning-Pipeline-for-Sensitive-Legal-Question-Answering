# Simple Explanation for Mentor — VERA Project & Review Response

## 1. What is this project? (In one sentence)

We built **VERA**: a multi-step AI pipeline that answers **legal multiple-choice questions** about Indian child protection law (POCSO, etc.), and gives **explainable answers** that can be checked against the law.

---

## 2. How does the pipeline work? (Simple flow)

Think of it like a **chain of experts**:

1. **RAG (retrieval):** From a big pile of laws and court texts, we **fetch the bits that are relevant** to the question.
2. **ARR (main model):** One AI (Qwen) **reads that text and picks an answer** (A/B/C/D) and writes a short reasoning.
3. **Critique:** Another AI acts as a **critic** and says what might be wrong with that answer.
4. **Defense:** Another AI **defends** the first answer against the critic.
5. **Judge:** A **judge AI** reads both sides and decides: who is right? What is the final answer?
6. **CoV (verification):** A **verifier** checks that the final answer really matches the law text.
7. **VERA:** We **combine** the verified reasoning and the judge’s reasoning into one clear, final answer.

So: we don’t just ask one model once; we use **several steps** so the answer is more accurate and easier to trust.

---

## 3. Where do I run the code? (Two folders — use only one)

- **`QA_CHild_legal/`**  
  This is the **main project folder**. All runnable code (notebooks, scripts, config, train/test data) is here.  
  **We run the full pipeline from this folder only.**

- **`QA_CHild_legal/all_data/`**  
  This is **only data**: PDFs of laws, court case texts, etc. (unzipped from `all_data.zip`).  
  **We do not run any pipeline from here.** The pipeline in `QA_CHild_legal/` *reads* from this data when it needs it (e.g. for RAG).

**Summary for mentor:**  
“We have one codebase in `QA_CHild_legal/`. The folder `all_data/` is just input data for that codebase; we don’t run the pipeline inside `all_data/`.”

---

## 4. What did the reviewer ask? (Cosine similarity threshold)

The reviewer is asking about a **number (0.5 or 0.55)** used to decide if an answer is “correct” in **one specific experiment** — the **open-ended** one.

- In the **main experiment** (multiple-choice, 93.8% accuracy):  
  The model outputs text; we compare it to the four options (A, B, C, D) using **cosine similarity** and choose the **option that is most similar** (no fixed threshold).

- In the **open-ended experiment** (Section 6.8, 500 questions):  
  We remove the options; the model gives a free-text answer. To decide “correct/incorrect,” we compute **cosine similarity** between the model’s answer and the correct option text. In the paper we said: **if similarity > 0.55, we count it as correct.**  
  That **0.55** is the “threshold” the reviewer is asking about.

**Summary for mentor:**  
“The threshold is used only in the open-ended evaluation (500 questions), not in the main MCQ accuracy. The reviewer wants us to justify or clarify this threshold (e.g. why 0.55, or report also at 0.5).”

---

## 5. How do we “defend” it? (What we are doing)

We are **not** re-running the whole pipeline. We are only:

1. **Clarifying in the paper/response:**  
   “The 0.55 threshold is used **only** for the open-ended evaluation; the main MCQ result uses **argmax** (most similar option), not a threshold.”

2. **Justifying 0.55:**  
   “We chose 0.55 by checking a validation set: above 0.55 the answers matched the correct option well; below that they were often partial or wrong.”

3. **Adding a small sensitivity check:**  
   We take the **same** open-ended model answers we already have and recompute accuracy using **different thresholds** (e.g. 0.45, 0.5, 0.55, 0.6). We add a table showing that results don’t change much — so the exact choice of 0.55 is not fragile. If the reviewer asked for 0.5, we also report accuracy at 0.5.

**Summary for mentor:**  
“We defend the threshold by (a) clarifying where it is used, (b) explaining how we chose 0.55, and (c) showing a small table that accuracy is stable for thresholds around 0.5–0.6. We do **not** need to re-run the full pipeline; we only re-evaluate the existing open-ended outputs with different thresholds.”

---

## 6. One-paragraph version you can say to your mentor

“Our project is VERA: a multi-step AI pipeline for legal multiple-choice QA in Indian child protection law. We run everything from the folder **QA_CHild_legal/**; **all_data/** is just data (laws, cases), not a second codebase. A reviewer asked about the **cosine similarity threshold**. That threshold is used **only** in the open-ended experiment (500 questions), not in the main 93.8% MCQ result. We will defend it by clarifying this in the paper, explaining that 0.55 was chosen on a validation set, and adding a short table showing accuracy at several thresholds (e.g. 0.45, 0.5, 0.55, 0.6) so they can see the result is stable. We don’t need to re-run the full pipeline — only re-evaluate the open-ended answers with different thresholds.”

---

## 7. Key terms (for your own understanding)

| Term | Simple meaning |
|------|----------------|
| **RAG** | Retrieve relevant law/case text before answering. |
| **ARR** | Analyze → Reason → Respond (structured way to use the retrieved text). |
| **CDA** | Critique–Debate–Adjudicate (critic, defender, judge). |
| **CoV** | Chain-of-Verification (check answer against the law). |
| **VERA** | Final step that merges verified + judge reasoning. |
| **Cosine similarity** | A number between -1 and 1 saying how similar two texts are (higher = more similar). |
| **Threshold (0.55)** | In open-ended evaluation only: we call the answer “correct” if similarity to the gold answer is above 0.55. |
| **argmax** | Pick the option (A/B/C/D) that has the **highest** similarity to the model’s answer — no threshold. |

You can share this file with your mentor so they have the full picture in one place.
