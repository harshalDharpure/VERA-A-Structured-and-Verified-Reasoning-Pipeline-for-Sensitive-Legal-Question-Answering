"""
Defense step: load ARR + Critique + RAG context, run Defender model (Qwen2.5-7B), write Defense.jsonl.
Run from QA_CHild_legal: HF_TOKEN=<token> python defence.py
"""
import json
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# Script dir and config
_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_script_dir)

with open(os.path.join(_script_dir, "config.yaml"), "r", encoding="utf-8") as file:
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

# ARR output (Question -> main_model_reasoning and model's answer for defense)
arr_file = config.get("save_file_main_model", "result_qwen7b_main_architecture05032025.jsonl")
arr_path = os.path.join(_script_dir, arr_file)
context_dict1 = {}
arr_answer_dict = {}
with open(arr_path, "r", encoding="utf-8") as cf:
    for line in cf:
        entry = json.loads(line)
        q = entry.get("Question")
        if q is not None:
            context_dict1[q] = entry.get("main_model_reasoning") or ""
            arr_answer_dict[q] = entry.get("Correct Answer", "")

# Critique output (Question -> Critique)
critique_path = os.path.join(_script_dir, "Critique.jsonl")
context_dict2 = {}
with open(critique_path, "r", encoding="utf-8") as cf:
    for line in cf:
        entry = json.loads(line)
        q = entry.get("Question")
        if q is not None:
            context_dict2[q] = entry.get("Critique") or ""

# Data list
data_file = config.get("actual_data", "train_data_100.jsonl")
data_path = os.path.join(_script_dir, data_file)
data_list = []
with open(data_path, "r", encoding="utf-8") as df:
    for line in df:
        line = line.strip()
        if line:
            data_list.append(json.loads(line))

print(f"Loaded {len(data_list)} data points, {len(context_dict1)} ARR, {len(context_dict2)} Critique, {len(context_dict)} RAG.")


def get_data_point_with_context_list(data_entry):
    record = data_entry.copy()
    question = record.get("Question")
    record["retrieved_context"] = context_dict.get(question, "")
    record["main_model_reasoning"] = context_dict1.get(question, "")
    record["Critique"] = context_dict2.get(question, "")
    if question in arr_answer_dict:
        record["Initial_Answer_From_Model"] = arr_answer_dict[question]
    return record


defense_system_prompt = """
You are a legal defense expert. The following answer and its reasoning have been critiqued.
Your task: Defend the original answer based on the retrieved_context. Respond to each critique point, reinforce why the answer remains correct under Indian law, and clarify any misunderstandings.
Base 90% of your defense strictly on the retrieved_context, and use up to 10% general legal knowledge to strengthen your argument where relevant.
"""


def generate_defense_content(data_entry):
    question = data_entry.get("Question", "")
    A = data_entry.get("A", "")
    B = data_entry.get("B", "")
    C = data_entry.get("C", "")
    D = data_entry.get("D", "")
    context = data_entry.get("retrieved_context", "")
    answer = data_entry.get("Initial_Answer_From_Model", data_entry.get("Correct Answer", ""))
    reasoning = data_entry.get("main_model_reasoning", "")
    critique = data_entry.get("Critique", "")
    if A or B or C or D:
        options_block = f"""
Options:
A: {A}
B: {B}
C: {C}
D: {D}

"""
    else:
        options_block = "\n(Open-ended question; no multiple-choice options.)\n\n"
    return f"""
Question:
{question}
{options_block}Initial Answer:
{answer}

Initial Reasoning:
{reasoning}

Challenger's Critique:
{critique}

Retrieved Context (Laws/Regulations):
{context}

Defend the answer by addressing the critique and clarifying reasoning.
Respond to each critique point, reinforce why the answer remains correct under Indian law, and clarify any misunderstandings.
Base your defense strictly on the retrieved_context. Focus your defense 90% on the retrieved context and 10% on general legal knowledge.
"""


# HF login optional (use HF_TOKEN when running)
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(os.environ["HF_TOKEN"])

model_name = "Qwen/Qwen2.5-7B-Instruct"
print("Loading model:", model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def defender_model(data_entry):
    defense_content = generate_defense_content(data_entry)
    messages = [
        {"role": "system", "content": defense_system_prompt},
        {"role": "user", "content": defense_content},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    defense = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return defense


# Main loop
result_log_path = os.path.join(_script_dir, "Defense.jsonl")
if os.path.isfile(result_log_path):
    with open(result_log_path, "r", encoding="utf-8") as f:
        output_log = [json.loads(line) for line in f]
else:
    output_log = []

processed_questions = set(entry.get("Question") for entry in output_log)
print(f"Resuming: {len(processed_questions)} already in Defense.jsonl.")

for i, datapoint in enumerate(data_list):
    question = datapoint.get("Question")
    if not question or question in processed_questions:
        continue
    try:
        compact_data = get_data_point_with_context_list(datapoint)
        defense = defender_model(compact_data)
        temp_data = {"Question": question, "Defense": defense}
        output_log.append(temp_data)
        with open(result_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(temp_data, ensure_ascii=False) + "\n")
        print(f"Data {i} saved: {question[:60]}...")
    except Exception as e:
        print(f"Error processing: {question[:50]}... -> {e}")

print("Defense processing complete.", result_log_path)
