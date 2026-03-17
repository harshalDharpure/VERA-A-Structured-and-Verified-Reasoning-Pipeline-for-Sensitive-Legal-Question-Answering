from transformers import  AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset
import pandas as pd
import numpy as np
import warnings
import logging
import random
import torch
import json
import yaml
import os
import json
import torch
from transformers import pipeline
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

warnings.filterwarnings("ignore")

_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_script_dir)

with open(os.path.join(_script_dir, 'config.yaml'), 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)


contexts_file1 = os.path.join(_script_dir, config.get('save_file_main_model', 'result_qwen7b_main_architecture05032025.jsonl'))
context_dict1 = {}
arr_answer_dict = {}  # Question -> model's answer (for open-ended: use model answer as Initial Answer, not gold)
with open(contexts_file1, 'r', encoding='utf-8') as cf:
    for line in cf:
        entry = json.loads(line)
        question = entry.get('Question')
        if question is not None:
            context_dict1[question] = entry.get('main_model_reasoning') or ''
            arr_answer_dict[question] = entry.get('Correct Answer', '')

contexts_file2 = os.path.join(_script_dir, "Critique.jsonl")
context_dict2 = {}
with open(contexts_file2, 'r', encoding='utf-8') as cf:
    for line in cf:
        entry = json.loads(line)
        question = entry.get('Question')
        if question is not None:
            context_dict2[question] = entry.get('Critique') or ''

contexts_file = os.path.join(_script_dir, "Defense.jsonl")
context_dict = {}
with open(contexts_file, 'r', encoding='utf-8') as cf:
    for line in cf:
        entry = json.loads(line)
        question = entry.get('Question')
        if question is not None:
            context_dict[question] = entry.get('Defense') or ''

data_file = os.path.join(_script_dir, config['actual_data'])
data_list = []
with open(data_file, 'r', encoding='utf-8') as df:
    for line in df:
        line = line.strip()
        if line:
            data_list.append(json.loads(line))

def get_data_point_with_context_list(data_entry):
    record = data_entry.copy()
    question = record.get('Question')
    record['main_model_reasoning'] = context_dict1.get(question, '')
    record['Critique'] = context_dict2.get(question, '')
    record['Defense'] = context_dict.get(question, '')
    # Use ARR model's answer as Initial Answer when available (critical for open-ended: do not feed gold to Judge)
    if question in arr_answer_dict:
        record['Correct Answer'] = arr_answer_dict[question]
    return record

judge_system_prompt = '''
You are a neutral legal expert who evaluates debates between a Challenger and a Defender.
Judge who made the stronger case. Clearly state the winner ('Challenger' or 'Defender') and explain your reasoning.
Provide the correct answer and sound reasoning for the case.
Your job is to:
1️⃣ **Assess both arguments** based on legal correctness, logical strength, and clarity.
2️⃣ **Declare the winner** by deciding who presented the stronger case: 'Challenger' or 'Defender'.
3️⃣ **Provide the correct legal answer** to the question (even if both parties made mistakes).
4️⃣ **Explain your final reasoning** concisely and precisely.Your job is to:
Your response must be in strict JSON format as shown below:
```json

{
    "Question": "<Insert the question here>",
    "A": "<Insert Option A text>",
    "B": "<Insert Option B text>",
    "C": "<Insert Option C text>",
    "D": "<Insert Option D text>",
    "Winner": "<Challenger or Defender>",
    "Correct Answer": "<Write the full correct answer, not just the option letter>",
    "Judgement": "<Provide detailed reasoning explaining why the winner’s argument was stronger and why the correct answer is correct>",
    "final_reasoning" : <Provide overall final reasoning behind the answer. Write full reasoning behind the answer Don't mention which option it is directly You can mention content of option>
}

 Output should be in strictly in json format
'''


def generate_judge_content(data_entry):
    question = data_entry.get('Question', '')
    A = data_entry.get('A', '')
    B = data_entry.get('B', '')
    C = data_entry.get('C', '')
    D = data_entry.get('D', '')
    answer = data_entry.get('Correct Answer', '')
    reasoning = data_entry.get('main_model_reasoning', '')
    critique = data_entry.get('Critique', '')
    defense= data_entry.get('Defense', '')
    if A or B or C or D:
        options_block = f'''
            Options:
            A: {A}
            B: {B}
            C: {C}
            D: {D}

            '''
    else:
        options_block = "\n            (Open-ended question; no multiple-choice options.)\n            \n"
    judge_content = f'''
            Question:
            {question}
            {options_block}
            Initial Answer:
            {answer}

            Initial Reasoning:
            {reasoning}

            Challenger’s Critique:
            {critique}

            Defender's defense:
            {defense}

Do not mention which option you are choosing. Provide only the content of the option. 
                '''

    return judge_content

def extract_nested_braces(s):
    # s = s.replace('"','')
    s += "}"
    stack = []
    start = -1
    result = []
    # print(s)
    for i, char in enumerate(s):
        if char == '{':
            if not stack:
                start = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:
                    result.append(s[start:i+1])
    return result[0] if result else None

def string_to_dict(json_string):
    try:
        # Parse the JSON string into a dictionary
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")


from huggingface_hub import login
if os.environ.get("HF_TOKEN"):
    login(os.environ["HF_TOKEN"])


model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


def judge_model(data_entry):
    judge_content = generate_judge_content(data_entry)
    messages = [
        {"role": "system", "content": judge_system_prompt},
        {"role": "user", "content": judge_content},
    ]
    outputs = pipe(messages, max_new_tokens=1500)
    out = outputs[0]
    if isinstance(out.get("generated_text"), list) and len(out["generated_text"]) > 0:
        json_output = out["generated_text"][-1].get("content", "")
    else:
        json_output = out.get("generated_text", str(out))
    raw = extract_nested_braces(json_output)
    if not raw:
        raise ValueError("No JSON object found in model output")
    result = string_to_dict(raw)
    return result


result_log_path = os.path.join(_script_dir, 'judge2.jsonl')
MAX_JUDGE_RETRIES = 2  # retry up to 2 times on JSON errors

# Load existing results if available
if os.path.isfile(result_log_path):
    with open(result_log_path, 'r', encoding='utf-8') as file:
        output_log = [json.loads(line) for line in file]
else:
    output_log = []

processed_questions = set(entry.get("Question") for entry in output_log)
print(f"Resuming: {len(processed_questions)} already in judge2.jsonl.")

for i, datapoint in enumerate(data_list):
    question = datapoint.get('Question')
    if not question or question in processed_questions:
        continue

    compact_data = get_data_point_with_context_list(datapoint)
    judge_result = None
    last_error = None
    for attempt in range(MAX_JUDGE_RETRIES + 1):
        try:
            judge_result = judge_model(compact_data)
            break
        except Exception as e:
            last_error = e
            if attempt < MAX_JUDGE_RETRIES:
                print(f"Retry {attempt + 1}/{MAX_JUDGE_RETRIES} for: {question[:50]}...")
            continue

    if judge_result is not None:
        temp_data = judge_result.copy()
        temp_data.pop("Question", None)
        temp_data["Question"] = datapoint["Question"]
        # Ensure final_reasoning present for vera_merge / evaluator (fallback to Judgement)
        if "final_reasoning" not in temp_data or not temp_data.get("final_reasoning"):
            temp_data["final_reasoning"] = temp_data.get("Judgement", "") or ""
        output_log.append(temp_data)
        with open(result_log_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(temp_data, ensure_ascii=False) + '\n')
        print(f"Data {i} saved: {question[:60]}...")
    else:
        # Save minimal record for failure after all retries
        fail_data = {
            "Question": question,
            "A": datapoint.get("A", ""),
            "B": datapoint.get("B", ""),
            "C": datapoint.get("C", ""),
            "D": datapoint.get("D", ""),
            "Winner": "Unknown",
            "Correct Answer": "",
            "Judgement": f"Failed after {MAX_JUDGE_RETRIES + 1} attempts: {last_error!s}" if last_error else "Failed (no response)",
            "final_reasoning": "",
        }
        output_log.append(fail_data)
        with open(result_log_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(fail_data, ensure_ascii=False) + '\n')
        print(f"Data {i} saved (failure record): {question[:60]}...")

print("Processing complete.", result_log_path)


