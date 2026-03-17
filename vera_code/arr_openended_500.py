#!/usr/bin/env python3
"""
Open-ended ARR for 500 no-options: generate Answer + Reasoning from Question + Passage + retrieved_context.
Output: result_openended_500.jsonl (Question, Correct Answer, main_model_reasoning).
Uses config_500.yaml when run with --config config_500.yaml.
"""
import json
import os
import sys
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_script_dir)

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(_script_dir, "config_500.yaml"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    config = load_config(args.config)

    data_path = os.path.join(_script_dir, config["actual_data"])
    rag_path = os.path.join(_script_dir, config["rag_retrieved_context"])
    out_path = os.path.join(_script_dir, config["save_file_main_model"])

    data_list = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))
    context_dict = {}
    with open(rag_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            q = entry.get("Question")
            if q is not None:
                context_dict[q] = entry.get("retrieved_context") or ""

    system_prompt = """You are a legal QA assistant. Answer based only on the retrieved context and passage.
Output strictly this JSON (no markdown):
{"Answer": "<one direct legal answer>", "Reasoning": "<brief explanation>"}"""

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print("Loading model", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def extract_json(s):
        s = (s or "") + "}"
        stack = []
        start = -1
        for i, c in enumerate(s):
            if c == "{":
                if not stack:
                    start = i
                stack.append(c)
            elif c == "}":
                if stack:
                    stack.pop()
                    if not stack:
                        return s[start : i + 1]
        return None

    output_log = []
    if os.path.isfile(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            output_log = [json.loads(line) for line in f]
    processed = set(e.get("Question") for e in output_log)
    print(f"Resuming: {len(processed)} already in {out_path}")

    for i, row in enumerate(data_list):
        q = row.get("Question")
        if not q or q in processed:
            continue
        passage = row.get("Passage", "")[:2000]
        ctx = context_dict.get(q, "")[:3000]
        user = f"Passage:\n{passage}\n\nRetrieved context:\n{ctx}\n\nQuestion: {q}\n\nProvide JSON with Answer and Reasoning."
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tokenizer([text], return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=512)
        dec = tokenizer.batch_decode([out[0][inp.input_ids.shape[1]:]], skip_special_tokens=True)[0]
        raw = extract_json(dec)
        answer, reasoning = "", ""
        if raw:
            try:
                obj = json.loads(raw)
                answer = obj.get("Answer", obj.get("answer", "")) or ""
                reasoning = obj.get("Reasoning", obj.get("reasoning", "")) or ""
            except Exception:
                pass
        if not answer:
            answer = dec[:500] if dec else "No answer"
        if not reasoning:
            reasoning = "No reasoning"
        rec = {"Question": q, "Correct Answer": answer, "main_model_reasoning": reasoning}
        output_log.append(rec)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}")

    print("Done.", out_path)

if __name__ == "__main__":
    main()
