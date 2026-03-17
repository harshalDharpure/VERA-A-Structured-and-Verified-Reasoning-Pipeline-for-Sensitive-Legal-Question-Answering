"""
Critique step: load ARR output + RAG context, run Challenger model, write Critique.jsonl.
Run from QA_CHild_legal: python critique.py
"""
import json
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import pipeline
from transformers import GenerationConfig

warnings.filterwarnings("ignore")

# Script dir and config
_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_script_dir)

with open(os.path.join(_script_dir, "config.yaml"), "r") as file:
    config = yaml.safe_load(file)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)

# Optional: filter judge1.jsonl (skip if not needed)
data_file = config.get("actual_data", "train_data.jsonl")
data_path = os.path.join(_script_dir, data_file)
if os.path.isfile(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data_questions = set(json.loads(line).get("Question", "") for line in f if line.strip())
else:
    data_questions = set()

if os.path.isfile(os.path.join(_script_dir, "judge1.jsonl")):
    seen = set()
    filtered = []
    with open(os.path.join(_script_dir, "judge1.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            q = entry.get("Question")
            if q in data_questions and q not in seen:
                seen.add(q)
                filtered.append(entry)
    with open(os.path.join(_script_dir, "judge1.jsonl"), "w", encoding="utf-8") as f:
        for entry in filtered:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Filtered judge1.jsonl to {len(filtered)} entries.")
else:
    print("judge1.jsonl not found; skipping filter.")

# RAG context (Question -> retrieved_context)
contexts_file = config.get("rag_retrieved_context", "rag_retrieved_questions.jsonl")
context_path = os.path.join(_script_dir, contexts_file)
context_dict = {}
if contexts_file.endswith(".jsonl"):
    with open(context_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            q = entry.get("Question")
            if q is not None:
                context_dict[q] = entry.get("retrieved_context") or ""
else:
    context_df = pd.read_excel(context_path)
    if "Question" in context_df.columns and "retrieved_context" in context_df.columns:
        context_dict = dict(zip(context_df["Question"], context_df["retrieved_context"]))
    else:
        raise ValueError("Excel must have 'Question' and 'retrieved_context' columns.")

# ARR output (Question -> main_model_reasoning and model's answer for Initial Answer)
contexts_file1 = config.get("save_file_main_model", "result_qwen7b_main_architecture05032025.jsonl")
arr_path = os.path.join(_script_dir, contexts_file1)
context_dict1 = {}
arr_answer_dict = {}  # Question -> model answer (use as Initial Answer so we critique/defend/judge model, not gold)
with open(arr_path, "r", encoding="utf-8") as cf:
    for line in cf:
        entry = json.loads(line)
        q = entry.get("Question")
        if q is not None:
            context_dict1[q] = entry.get("main_model_reasoning") or ""
            arr_answer_dict[q] = entry.get("Correct Answer", "")

# Data list
data_list = []
with open(data_path, "r", encoding="utf-8") as df:
    for line in df:
        line = line.strip()
        if line:
            data_list.append(json.loads(line))

print(f"Loaded {len(data_list)} data points, {len(context_dict1)} ARR outputs, {len(context_dict)} RAG contexts.")


def get_data_point_with_context_list(data_entry):
    record = data_entry.copy()
    question = record.get("Question")
    record["retrieved_context"] = context_dict.get(question, "")
    record["main_model_reasoning"] = context_dict1.get(question, "")
    # Use ARR model's answer as Initial Answer when available (open-ended: critique model output, not gold)
    if question in arr_answer_dict:
        record["Initial_Answer_From_Model"] = arr_answer_dict[question]
    return record


Critique_system_prompt = """
You are a legal critique expert. A question has been answered based on the document provided in the retrieved_context.

Your task: Critically assess the reasoning and answer in light of the retrieved_context. Identify any logical gaps, potential misinterpretations of the law, or areas where the reasoning could be challenged or improved.

Additionally, evaluate each of the provided answer options individually. Where appropriate, suggest a better answer with well-reasoned arguments grounded in law.

Base 90% of your critique strictly on the retrieved_context, and use up to 10% general legal knowledge to support your evaluation.
"""


def generate_critique_content(data_entry):
    opts = (data_entry.get('A') or data_entry.get('B') or data_entry.get('C') or data_entry.get('D'))
    if opts:
        options_block = f"""Options:
A: {data_entry.get('A', '')}
B: {data_entry.get('B', '')}
C: {data_entry.get('C', '')}
D: {data_entry.get('D', '')}

"""
    else:
        options_block = "(Open-ended question; no multiple-choice options.)\n\n"
    return f"""Question:
{data_entry.get('Question', '')}

{options_block}Retrieved Context (Laws/Regulations):
{data_entry.get('retrieved_context', '')}

Initial Answer:
{data_entry.get('Initial_Answer_From_Model', data_entry.get('Correct Answer', ''))}

Initial Reasoning:
{data_entry.get('main_model_reasoning', '')}

👉 Your task: Critically evaluate the reasoning and answer above. Identify any logical gaps, potential misinterpretations of the law, or areas where the reasoning may be challenged or require clarification.
Focus your critique 90% on the retrieved context and 10% on general legal knowledge.
"""


# HF login optional
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(os.environ["HF_TOKEN"])

# Use 3B for lower VRAM (fits ~6GB); set CRITIQUE_8B=1 to use Llama-3.1-8B
model_id = "meta-llama/Llama-3.2-3B-Instruct" if os.environ.get("CRITIQUE_8B") != "1" else "meta-llama/Llama-3.1-8B-Instruct"
if "3B" in model_id:
    pipe = pipeline("text-generation", model=model_id, torch_dtype=torch.bfloat16, device_map="auto")
else:
    try:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        pipe = pipeline("text-generation", model=model_id, model_kwargs={"quantization_config": quant_config}, device_map="auto")
    except Exception:
        pipe = pipeline("text-generation", model=model_id, dtype=torch.bfloat16, device_map="auto")


def critique_model(data_entry):
    content = generate_critique_content(data_entry)
    messages = [
        {"role": "system", "content": Critique_system_prompt},
        {"role": "user", "content": content},
    ]
    gen_config = GenerationConfig(max_new_tokens=1500)
    outputs = pipe(messages, generation_config=gen_config)
    # Chat pipeline returns list of generated turn; take last content
    out = outputs[0]
    if isinstance(out.get("generated_text"), list) and len(out["generated_text"]) > 0:
        critique = out["generated_text"][-1].get("content", "")
    else:
        critique = out.get("generated_text", str(out))
    return critique


# Main loop
result_log_path = os.path.join(_script_dir, "Critique.jsonl")
if os.path.isfile(result_log_path):
    with open(result_log_path, "r", encoding="utf-8") as f:
        output_log = [json.loads(line) for line in f]
else:
    output_log = []

processed_questions = set(entry.get("Question") for entry in output_log)
print(f"Resuming: {len(processed_questions)} already in Critique.jsonl.")

for i, datapoint in enumerate(data_list):
    question = datapoint.get("Question")
    if not question or question in processed_questions:
        continue
    try:
        compact_data = get_data_point_with_context_list(datapoint)
        critique = critique_model(compact_data)
        temp_data = {"Question": question, "Critique": critique}
        output_log.append(temp_data)
        with open(result_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(temp_data, ensure_ascii=False) + "\n")
        print(f"Data {i} saved: {question[:60]}...")
    except Exception as e:
        print(f"Error processing: {question[:50]}... -> {e}")

print("Critique processing complete.", result_log_path)
