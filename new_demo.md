# McNemar Test @ τ = 0.75 — Simple Explanation

## Setup

On the same **498 questions**, each system either **passes** or **fails** the Natural Language Inference (NLI) evaluation at a threshold of **τ = 0.75**.

- **Pass:** NLI score ≥ 0.75 (counts as "correct" under this automatic metric)
- **Fail:** NLI score < 0.75

McNemar's test compares **VERA** and **ARR** using these paired pass/fail outcomes.

---

# What is the "difference"?

The difference is **not simply the overall accuracy** (e.g., 98.8% vs. 91.4%).

Instead, McNemar focuses only on the questions where the two systems **disagree**.

| Type | Meaning | Count |
|------|---------|------:|
| ARR correct, VERA wrong | ARR passed @ 0.75, VERA failed | **42** |
| VERA correct, ARR wrong | VERA passed @ 0.75, ARR failed | **5** |
| Both same | Both passed or both failed | Ignored |

Total discordant pairs:

```text
42 + 5 = 47
```

These are the only questions used by McNemar's test.

### Difference in simple words

Among the questions where the systems disagree:

- ARR wins **42** times.
- VERA wins **5** times.

So the comparison is:

```text
42 vs 5
```

This is a very large imbalance in favor of ARR.

---

# What is a p-value?

A **p-value** answers the following question:

> "If the two systems were actually equally good, how likely is it that we would observe a difference this large (or even larger) purely by random chance?"

Common interpretation:

| p-value | Interpretation |
|---------|----------------|
| < 0.05 | Statistically significant |
| < 0.01 | Strong evidence |
| < 0.001 | Very strong evidence |

### Your result

```text
p < 0.001
```

This means:

- It is **very unlikely** that the observed **42 vs 5** split happened by chance.
- Therefore, the difference between ARR and VERA is statistically significant.

If your significance threshold is **0.01**, then:

```text
0.001 < 0.01
```

So your result easily meets that criterion.

---

# How McNemar Works (Intuition)

McNemar ignores all questions where both systems agree.

It only considers the **47 disagreement cases**.

If both systems were equally good, we would expect something close to:

```text
23 vs 24
```

Instead, we observe:

```text
42 vs 5
```

This strong imbalance is converted into a **p-value**.

Since:

```text
p < 0.001
```

the imbalance is far too large to be explained by random chance alone.

---

# How to Interpret Your Results

### McNemar @ τ = 0.75

```text
ARR correct, VERA wrong : 42
VERA correct, ARR wrong : 5
p-value                 : < 0.001
```

### Plain English Interpretation

On the same **498 questions**, ARR satisfies the NLI consistency criterion at **τ = 0.75** significantly more often than VERA.

Among the questions where the systems disagree:

- ARR performs better on **42** questions.
- VERA performs better on only **5** questions.

The result:

```text
p < 0.001
```

shows that this difference is statistically significant and is very unlikely to be due to random variation.

**Conclusion:** ARR significantly outperforms VERA on this automatic NLI pass/fail metric.

---

# What McNemar Does NOT Mean

| McNemar Says | McNemar Does NOT Say |
|--------------|----------------------|
| ARR performs better than VERA on NLI pass/fail at τ = 0.75 | ARR is legally more correct |
| The observed difference is statistically significant | VERA is useless overall |
| The improvement is unlikely to be due to chance | NLI is perfect ground truth |

---

# One-Sentence Version for a Paper

> McNemar's test on 498 paired open-ended items at τ = 0.75 yielded 42 cases where only ARR passed versus 5 where only VERA passed (p < 0.001), indicating that ARR significantly outperforms VERA on reference-consistency NLI at this threshold.

---

# Quick Memory Trick

Remember:

```text
42 vs 5
```

means:

> **Who wins when they disagree?**

**Answer:** ARR wins almost every time.

And:

```text
p < 0.001
```

means:

> **Could this difference just be luck?**

**Answer:** Almost certainly **no**.
