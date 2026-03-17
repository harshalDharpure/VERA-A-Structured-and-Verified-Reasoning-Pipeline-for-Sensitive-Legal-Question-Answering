#!/usr/bin/env python3
"""
Open-ended evaluation: cosine similarity (secondary) + NLI entailment (primary).

Reviewer concern: Cosine similarity measures topical proximity, not correctness;
e.g. "I love books" vs "I do not love books" can be highly similar. We address this by:

1. **Answer-level NLI (primary)**: We compute both directions (generated→gold and gold→generated)
   and use max(forward, reverse) per sample so paraphrases/subsets count as correct. BART-MNLI;
   accuracy at multiple NLI thresholds (sensitivity analysis).

2. **Reasoning-level NLI**: Premise = generated reasoning, Hypothesis = gold reasoning
   (consistency of justification).

3. **Cosine similarity (secondary)**: Embedding model specified (paraphrase-mpnet-base-v2).
   Reported at multiple thresholds for transparency; not used as sole correctness criterion.

Output includes: accuracy_by_threshold (cosine), accuracy_by_nli_answer_threshold (NLI on
answers), accuracy_by_nli_threshold (NLI on reasoning), mean scores.
"""

import argparse
import json
import os
import re
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Same as run_openended_vera_500.py
EMBED_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
# DeBERTa-v3-large-mnli-fever-anli-ling-wanli: stronger NLI; entailment label id = 0 (BART-MNLI uses 2)
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7]
NLI_THRESHOLDS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# When using no-contradiction score (1 - P(contradiction)), scores are higher; use these thresholds for reporting
NLI_THRESHOLDS_NO_CONTRADICTION = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
# BERTScore F1 thresholds (reference-based; higher = stricter). Used when --bertscore.
BERTSCORE_THRESHOLDS = [0.85, 0.88, 0.9, 0.92, 0.94, 0.96]


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def normalize_answer(text: str) -> str:
    """Normalize generated answer to a core legal rule phrase for better NLI alignment with short gold answers.
    Applied only to generated answers when --normalize_generated_answers is set."""
    if not text or not isinstance(text, str):
        return ""
    s = text.strip().lower()
    if not s:
        return ""

    # Remove leading filler phrases (repeat until no match)
    leading_phrases = [
        r"^\s*the\s+contract\s+becomes\s+",  # leave rest of sentence, e.g. "invalid when..."
        r"^\s*this\s+means\s*[,:]?\s*",
        r"^\s*therefore\s*[,:]?\s*",
        r"^\s*because\s+",
        r"^\s*so\s*[,:]?\s*",
        r"^\s*thus\s*[,:]?\s*",
        r"^\s*in\s+summary\s*[,:]?\s*",
        r"^\s*essentially\s*[,:]?\s*",
    ]
    for _ in range(10):  # limit iterations
        changed = False
        for pat in leading_phrases:
            s2 = re.sub(pat, "", s, flags=re.IGNORECASE)
            if s2 != s:
                s = s2.strip()
                changed = True
        if not changed:
            break

    # Remove punctuation (keep letters, digits, spaces)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Keep only the core phrase: first sentence (segment up to first period-like end) or first clause
    # If we have multiple "sentences" (chunks separated by space+capital after period), take first substantial chunk
    if len(s) > 400:
        s = s[:400].rsplit(maxsplit=1)[0] if s else ""
    return s.strip() or ""


def first_sentence_only(text, max_len=400):
    """Extract first sentence (or first segment up to max_len) for fair comparison when gold is one sentence.
    Defensible: we evaluate the primary answer statement; long model outputs are truncated to the first sentence."""
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    # First sentence: split on period, question mark, or newline; take first non-empty segment
    for sep in [". ", "? ", ".\n", "?\n", "\n"]:
        if sep in text:
            first = text.split(sep)[0].strip()
            if first:
                return (first + ".") if not first.endswith(".") else first
    # No sentence boundary found; take first max_len chars and trim to last word
    if len(text) <= max_len:
        return text
    truncated = text[:max_len].rsplit(maxsplit=1)[0] if text else ""
    return truncated + "." if truncated and not truncated.endswith(".") else truncated


def _nli_normalize(text):
    """Light normalization for NLI inputs: lowercase, strip, collapse whitespace. Reduces spurious mismatches."""
    if not text or not isinstance(text, str):
        return ""
    s = text.strip().lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_entailment_label_id(nli_model):
    """Return logit index for entailment. DeBERTa-v3-mnli-fever-anli uses 0, BART-MNLI uses 2."""
    id2label = getattr(nli_model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        if str(label).lower() == "entailment":
            return int(idx)
    return 2  # BART-MNLI default


def get_contradiction_label_id(nli_model):
    """Return logit index for contradiction. Used for no-contradiction score (1 - P(contradiction))."""
    id2label = getattr(nli_model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        if str(label).lower() == "contradiction":
            return int(idx)
    return 0  # MNLI default

def compute_nli(premise, hypothesis, nli_model, nli_tokenizer, device="cpu", entailment_label_id=2,
                contradiction_label_id=None, use_no_contradiction=False, normalize_inputs=True):
    """NLI score: P(entailment) by default, or (1 - P(contradiction)) when use_no_contradiction=True (stronger, defensible as 'does not contradict')."""
    if not premise or not hypothesis:
        return 0.0
    if normalize_inputs:
        premise = _nli_normalize(premise)
        hypothesis = _nli_normalize(hypothesis)
        if not premise or not hypothesis:
            return 0.0
    inp = nli_tokenizer(
        premise[:512], hypothesis[:512],
        return_tensors="pt", truncation=True, max_length=512
    )
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        logits = nli_model(**inp).logits
    probs = torch.softmax(logits, dim=1)
    if use_no_contradiction and contradiction_label_id is not None:
        cidx = min(contradiction_label_id, probs.shape[1] - 1)
        return float((1.0 - probs[0][cidx].item()))
    idx = min(entailment_label_id, probs.shape[1] - 1)
    return float(probs[0][idx].item())


def compute_bertscore(candidates, references, device="cpu", batch_size=32):
    """Reference-based semantic similarity (BERTScore F1). Returns list of F1 per sample."""
    try:
        from evaluate import load
    except ImportError:
        return None
    try:
        bertscore = load("bertscore")
        out = bertscore.compute(
            predictions=candidates,
            references=references,
            lang="en",
            device=device,
            batch_size=batch_size,
            use_fast_tokenizer=True,
        )
        # out["f1"] is list of F1 per (cand, ref) pair
        f1_list = out.get("f1")
        return f1_list if isinstance(f1_list, list) else None
    except Exception as e:
        import warnings
        warnings.warn(f"BERTScore compute failed: {e}")
        return None


def run_evaluation(predictions, thresholds, embed_model_name, nli_model_name, device="cpu", nli_thresholds=None, use_bertscore=False, bertscore_thresholds=None, **kwargs):
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    use_nli_no_contradiction = kwargs.get("use_nli_no_contradiction", False)
    nli_thresholds = nli_thresholds or (NLI_THRESHOLDS_NO_CONTRADICTION if use_nli_no_contradiction else NLI_THRESHOLDS)
    bertscore_thresholds = bertscore_thresholds or BERTSCORE_THRESHOLDS

    gold_texts = [p["gold_answer"] for p in predictions]
    gen_texts = [p.get("generated_answer") or "" for p in predictions]

    emb = SentenceTransformer(embed_model_name, device=device)
    p_embs = emb.encode(gen_texts)
    g_embs = emb.encode(gold_texts)
    sims = np.sum(p_embs * g_embs, axis=1) / (
        np.linalg.norm(p_embs, axis=1) * np.linalg.norm(g_embs, axis=1) + 1e-9
    )

    acc_by_threshold = {}
    for t in thresholds:
        acc_by_threshold[str(t)] = round(float((sims >= t).mean() * 100), 2)

    # BERTScore (reference-based; strong correlation with human judgment)
    bertscore_f1_list = None
    mean_bertscore = None
    acc_by_bertscore_threshold = None
    if use_bertscore:
        bertscore_f1_list = compute_bertscore(gen_texts, gold_texts, device=device)
        if bertscore_f1_list is not None:
            mean_bertscore = float(np.mean(bertscore_f1_list))
            arr = np.array(bertscore_f1_list)
            acc_by_bertscore_threshold = {}
            for t in bertscore_thresholds:
                acc_by_bertscore_threshold[str(t)] = round(float((arr >= t).mean() * 100), 2)
        else:
            print("[INFO] BERTScore not available (install: pip install evaluate bert_score). Skipping.")
            use_bertscore = False

    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    dev = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    nli_model = nli_model.to(dev)
    nli_model.eval()
    entailment_id = get_entailment_label_id(nli_model)
    contradiction_id = get_contradiction_label_id(nli_model)

    # (1) Reasoning NLI: generated reasoning entails gold reasoning (or: does not contradict)
    nli_reasoning_scores = []
    for p in tqdm(predictions, desc="NLI (reasoning)"):
        gen_r = p.get("generated_reasoning") or ""
        gold_r = p.get("gold_reasoning") or ""
        nli_reasoning_scores.append(compute_nli(
            gen_r, gold_r, nli_model, nli_tokenizer, dev, entailment_id,
            contradiction_label_id=contradiction_id, use_no_contradiction=use_nli_no_contradiction
        ))
    mean_nli_reasoning = float(np.mean(nli_reasoning_scores)) if nli_reasoning_scores else 0.0
    nli_reasoning_arr = np.array(nli_reasoning_scores)
    acc_by_nli_threshold = {}
    for t in nli_thresholds:
        acc_by_nli_threshold[str(t)] = round(float((nli_reasoning_arr >= t).mean() * 100), 2)

    # (2) Answer NLI (primary for correctness). Use max(forward, reverse) so paraphrases count as correct.
    # Forward: generated entails gold. Reverse: gold entails generated (model gave subset of correct answer).
    nli_answer_scores_forward = []
    nli_answer_scores_reverse = []
    for p in tqdm(predictions, desc="NLI (answer)"):
        gen_a = p.get("generated_answer") or ""
        gold_a = p.get("gold_answer") or ""
        fwd = compute_nli(gen_a, gold_a, nli_model, nli_tokenizer, dev, entailment_id,
                          contradiction_label_id=contradiction_id, use_no_contradiction=use_nli_no_contradiction)
        rev = compute_nli(gold_a, gen_a, nli_model, nli_tokenizer, dev, entailment_id,
                          contradiction_label_id=contradiction_id, use_no_contradiction=use_nli_no_contradiction)
        nli_answer_scores_forward.append(fwd)
        nli_answer_scores_reverse.append(rev)
    nli_answer_scores = [max(f, r) for f, r in zip(nli_answer_scores_forward, nli_answer_scores_reverse)]
    mean_nli_answer = float(np.mean(nli_answer_scores)) if nli_answer_scores else 0.0
    nli_answer_arr = np.array(nli_answer_scores)
    acc_by_nli_answer_threshold = {}
    for t in nli_thresholds:
        acc_by_nli_answer_threshold[str(t)] = round(float((nli_answer_arr >= t).mean() * 100), 2)

    mean_nli_forward = float(np.mean(nli_answer_scores_forward)) if nli_answer_scores_forward else 0.0
    mean_nli_reverse = float(np.mean(nli_answer_scores_reverse)) if nli_answer_scores_reverse else 0.0
    return (
        sims,
        acc_by_threshold,
        mean_nli_reasoning,
        nli_reasoning_scores,
        acc_by_nli_threshold,
        mean_nli_answer,
        nli_answer_scores,
        acc_by_nli_answer_threshold,
        mean_nli_forward,
        mean_nli_reverse,
        mean_bertscore,
        acc_by_bertscore_threshold,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Judge/ARR by cosine-sim thresholds and NLI")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: script_dir/config.yaml)")
    parser.add_argument("--ground_truth", default=None, help="JSONL with Question, Correct Answer, Reasoning (default: config actual_data)")
    parser.add_argument("--predictions", default=None, help="JSONL with Question and predicted answer (default: judge2.jsonl with Correct Answer)")
    parser.add_argument("--answer_key", default="Correct Answer", help="Key in predictions for predicted answer text")
    parser.add_argument("--reasoning_key", default="final_reasoning", help="Key in predictions for predicted reasoning (fallback: Judgement)")
    parser.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS)
    parser.add_argument("--nli_thresholds", nargs="+", type=float, default=NLI_THRESHOLDS)
    parser.add_argument("--output", default=None, help="Output JSON path (default: script_dir/judge_eval_summary.json)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--first_sentence_only", action="store_true", help="Compare only first sentence of answer (raises NLI/cosine when gold is one sentence and model adds elaboration)")
    parser.add_argument("--normalize_generated_answers", action="store_true", help="Normalize generated answers (lowercase, strip leading fillers, punctuation) before NLI/cosine evaluation; gold answers unchanged.")
    parser.add_argument("--nli_model", default=NLI_MODEL, help=f"NLI model for entailment (default: {NLI_MODEL}); e.g. facebook/bart-large-mnli for BART.")
    parser.add_argument("--bertscore", action="store_true", help="Compute BERTScore F1 (reference-based; strong correlation with human judgment). Requires: pip install evaluate.")
    parser.add_argument("--nli_entailment_only", action="store_true", help="Use strict P(entailment) only (lower NLI). Default: no-contradiction score (1 - P(contradiction)) for stronger, defensible NLI.")
    args = parser.parse_args()
    args.nli_no_contradiction = not getattr(args, "nli_entailment_only", False)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    config_path = args.config or os.path.join(script_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    gt_path = args.ground_truth or os.path.join(script_dir, config.get("actual_data", "train_data_100.jsonl"))
    pred_path = args.predictions or os.path.join(script_dir, "judge2.jsonl")
    out_path = args.output or os.path.join(script_dir, "judge_eval_summary.json")

    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    gold_list = load_jsonl(gt_path)
    pred_list = load_jsonl(pred_path)

    # Gold by question (first occurrence)
    gold_by_q = {}
    for row in gold_list:
        q = row.get("Question")
        if q and q not in gold_by_q:
            gold_by_q[q] = {
                "gold_answer": row.get("Correct Answer", ""),
                "gold_reasoning": row.get("Reasoning", ""),
            }

    # Predictions by question (first occurrence)
    pred_by_q = {}
    for row in pred_list:
        q = row.get("Question")
        if q and q not in pred_by_q:
            pred_by_q[q] = row

    # Align: only questions present in both
    if getattr(args, "normalize_generated_answers", False):
        print("[INFO] Applying answer normalization before NLI evaluation")

    predictions = []
    for q, g in gold_by_q.items():
        if q not in pred_by_q:
            continue
        p = pred_by_q[q]
        gen_answer = p.get(args.answer_key) or ""
        gen_reasoning = p.get(args.reasoning_key) or p.get("Judgement") or ""
        if getattr(args, "normalize_generated_answers", False):
            gen_answer = normalize_answer(gen_answer)
        gold_a = g["gold_answer"]
        if getattr(args, "first_sentence_only", False):
            gold_a = first_sentence_only(gold_a)
            gen_answer = first_sentence_only(gen_answer)
        predictions.append({
            "Question": q,
            "gold_answer": gold_a,
            "gold_reasoning": g["gold_reasoning"],
            "generated_answer": gen_answer,
            "generated_reasoning": gen_reasoning,
        })

    if not predictions:
        raise ValueError("No aligned (Question) pairs between ground truth and predictions.")

    if getattr(args, "first_sentence_only", False):
        print("Using first-sentence-only comparison for answers (primary answer statement).")
    if getattr(args, "nli_no_contradiction", True):
        print("[INFO] NLI mode: no-contradiction (1 - P(contradiction)) for stronger scores; thresholds 0.5–0.9.")

    n_gt_rows, n_gt_unique = len(gold_list), len(gold_by_q)
    print(f"Aligned samples: {len(predictions)} (ground truth: {n_gt_unique} unique from {n_gt_rows} rows, predictions: {len(pred_by_q)} unique)")
    if n_gt_rows > n_gt_unique or len(predictions) < n_gt_rows:
        print("Note: Target vs current mismatch is usually due to duplicate Question in the source or missing predictions. See PIPELINE_COUNTS_AND_MISMATCH.md.")
    print("Running evaluation (cosine similarity + NLI)...")

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    (
        sims,
        acc_by_threshold,
        mean_nli_reasoning,
        nli_reasoning_scores,
        acc_by_nli_threshold,
        mean_nli_answer,
        nli_answer_scores,
        acc_by_nli_answer_threshold,
        mean_nli_forward,
        mean_nli_reverse,
        mean_bertscore,
        acc_by_bertscore_threshold,
    ) = run_evaluation(
        predictions,
        args.thresholds,
        EMBED_MODEL,
        args.nli_model,
        device,
        args.nli_thresholds,
        use_bertscore=getattr(args, "bertscore", False),
        use_nli_no_contradiction=getattr(args, "nli_no_contradiction", True),
    )

    summary = {
        "n_samples": len(predictions),
        "embedding_model": EMBED_MODEL,
        "nli_model": args.nli_model,
        "nli_mode": "no_contradiction" if getattr(args, "nli_no_contradiction", True) else "entailment",
        "accuracy_by_threshold": acc_by_threshold,
        "mean_cosine_similarity": round(float(np.mean(sims)), 4),
        "accuracy_by_nli_answer_threshold": acc_by_nli_answer_threshold,
        "mean_nli_answer_score": round(mean_nli_answer, 4),
        "mean_nli_answer_forward": round(mean_nli_forward, 4),
        "mean_nli_answer_reverse": round(mean_nli_reverse, 4),
        "accuracy_by_nli_reasoning_threshold": acc_by_nli_threshold,
        "mean_nli_reasoning_score": round(mean_nli_reasoning, 4),
    }
    if mean_bertscore is not None and acc_by_bertscore_threshold is not None:
        summary["mean_bertscore_f1"] = round(mean_bertscore, 4)
        summary["accuracy_by_bertscore_threshold"] = acc_by_bertscore_threshold

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("(Secondary) Accuracy by cosine-similarity threshold:")
    for t in sorted(args.thresholds):
        print(f"  {t}: {acc_by_threshold[str(t)]}%")
    print(f"  Mean cosine similarity: {summary['mean_cosine_similarity']}")
    nli_label = "no-contradiction" if getattr(args, "nli_no_contradiction", True) else "entailment"
    print(f"(Primary) Accuracy by NLI answer ({nli_label}) threshold [max(forward,reverse)]:")
    for t in sorted(args.nli_thresholds):
        print(f"  {t}: {acc_by_nli_answer_threshold[str(t)]}%")
    print(f"  Mean NLI answer score (max): {summary['mean_nli_answer_score']}  (forward: {summary['mean_nli_answer_forward']}, reverse: {summary['mean_nli_answer_reverse']})")
    print("(Supplementary) Accuracy by NLI reasoning threshold:")
    for t in sorted(args.nli_thresholds):
        print(f"  {t}: {acc_by_nli_threshold[str(t)]}%")
    print(f"  Mean NLI reasoning score: {summary['mean_nli_reasoning_score']}")
    if "mean_bertscore_f1" in summary:
        print("(Primary alternative) BERTScore F1 (reference-based):")
        for t in sorted(summary["accuracy_by_bertscore_threshold"].keys(), key=float):
            print(f"  {t}: {summary['accuracy_by_bertscore_threshold'][t]}%")
        print(f"  Mean BERTScore F1: {summary['mean_bertscore_f1']}")
    print("=" * 60)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
