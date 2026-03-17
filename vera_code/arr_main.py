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

# Load config from script directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_script_dir)

with open(os.path.join(_script_dir, 'config.yaml'), 'r') as file:
    config = yaml.safe_load(file)
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)
system_prompt_file = config['arr_prompt']

with open(os.path.join(_script_dir, system_prompt_file), 'r') as file:
    system_prompt = file.read()

print(system_prompt)


data_file = config['actual_data']
contexts_file = config['rag_retrieved_context']

# Load all data into a list
data_list = []
with open(os.path.join(_script_dir, data_file), 'r', encoding='utf-8') as df:
    for line in df:
        line = line.strip()
        if line:
            data_list.append(json.loads(line))

# Load contexts into a dictionary
context_dict = {}
with open(os.path.join(_script_dir, contexts_file), 'r', encoding='utf-8') as cf:
    for line in cf:
        entry = json.loads(line)
        question = entry.get('Question')
        retrieved_context = entry.get('retrieved_context')
        if question is not None:
            context_dict[question] = retrieved_context if retrieved_context else ''


def generate_content(data_entry):
    Question = data_entry.get('Question', '')
    OptionA = data_entry.get('A', '')
    OptionB = data_entry.get('B', '')
    OptionC = data_entry.get('C', '')
    OptionD = data_entry.get('D', '')
    context = data_entry.get('retrieved_context', '')
    text = '''\n        Question: ''' + Question + '''\n            "A":''' + OptionA + '''\n            "B":''' + OptionB + '''\n            "C":''' + OptionC + '''\n            "D":''' + OptionD + '''\n "retrieved_context":''' + context + '''\n   \n    '''
    return text



# ✅ Function to fetch a data point + its retrieved context as a list
def get_data_point_with_context_list(data_entry):

    record = data_entry
    question = record.get('Question')
    retrieved_context = context_dict.get(question)

    record_with_context = record.copy()
    record_with_context['retrieved_context'] = retrieved_context

    # Convert to list of [key, value]


    return record_with_context


import os
from huggingface_hub import login
if os.environ.get('HF_TOKEN'):
    login(os.environ['HF_TOKEN'])


model_name = "Qwen/Qwen2.5-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


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
    return result[-1] if result else None

def string_to_dict(json_string):
    try:
        # Parse the JSON string into a dictionary
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")


def model_output(content, system_prompt_1):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    try:
        raw = extract_nested_braces(response)
        if not raw:
            return None, None
        result = string_to_dict(raw)
    except (ValueError, json.JSONDecodeError, TypeError):
        return None, None

    answer = (
        result.get("Answer") or
        result.get("answer") or
        result.get("correct_answer")
    )
    reasoning = (
        
        result.get("Reasoning") or
        result.get("reasoning")
    )

    return answer, reasoning


embedding_model = SentenceTransformer("paraphrase-mpnet-base-v2")

def get_best_option(data, predicted):
    if predicted is None or (isinstance(predicted, str) and not predicted.strip()):
        return "A"
    # Extract only the options (A, B, C, D)
    options = {key: data.get(key, '') for key in ['A', 'B', 'C', 'D']}
    if not any(options.values()):
        return "A"
    # Generate embeddings
    predicted_embedding = embedding_model.encode(predicted, convert_to_numpy=True)
    option_embeddings = {key: embedding_model.encode(val, convert_to_numpy=True) for key, val in options.items()}
    # Compute cosine similarities
    similarities = {key: cosine_similarity([predicted_embedding], [emb])[0][0] for key, emb in option_embeddings.items()}
    best_option = max(similarities, key=similarities.get)
    return best_option

# Define model name and output path (Judge expects this file in cwd)
save_file_main_model = config['save_file_main_model']

base_path = _script_dir
prompt_log = os.path.join(base_path, "prompt", os.path.splitext(save_file_main_model)[0])
result_log = base_path
accuracy_log = os.path.join(base_path, "accuracy", os.path.splitext(save_file_main_model)[0])

os.makedirs(prompt_log, exist_ok=True)
os.makedirs(accuracy_log, exist_ok=True)

file_name = os.path.basename(data_file)
print(f"Processing file: {file_name}")

data = data_list

try:
    # Generate system prompt and save it
    system_prompt_1 = system_prompt
    prompt_log_path = os.path.join(prompt_log, f'prompt_{os.path.splitext(save_file_main_model)[0]}_main_archi.txt')
    with open(prompt_log_path, "w") as file:
        file.write(system_prompt_1)

    test_data = data # Use the same slicing logic
    predictions = []
    true_labels = []
    result_log_path = os.path.join(result_log, save_file_main_model)

    # Load existing output_log if it exists
    if os.path.isfile(result_log_path):
        with open(result_log_path, 'r', encoding="utf-8") as file:
            output_log = [json.loads(line) for line in file]
    else:
        output_log = []

    # Create a set of already processed questions
    processed_questions = set(entry["Question"] for entry in output_log)

    # Process test data
    for i, data1 in enumerate(test_data):
        if data1["Question"] in processed_questions:
            continue  # Skip already processed data points

        try:
            content_retrieved = get_data_point_with_context_list(data1)
            final_content=generate_content(content_retrieved)

            predict, model_reasoning = model_output(final_content, system_prompt_1)
            if predict is None or model_reasoning is None:
                predict, model_reasoning = "A", "Parse error (model output invalid JSON)"

            print(predict)
            print(model_reasoning)
            print("HII")
            predict=get_best_option(data1, predict)
            actual=get_best_option(data1, data1.get("Correct Answer", ""))
            predictions.append(predict)
            true_labels.append(actual)
            print("HII")

            # Update the log with new data

            temp_dict = {"Question": data1["Question"]}
            print(temp_dict)
            temp_dict["Correct Answer"]=actual
            temp_dict["Predicted"] = predict
            temp_dict["main_model_reasoning"] = model_reasoning
            output_log.append(temp_dict)

            print(f"Data {i} saved")

            # Save updated output_log in JSONL format
            with open(result_log_path, 'w', encoding="utf-8") as file:
                for entry in output_log:
                    file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"An error occurred while processing data: {e}")
            continue  # Skip any errors and proceed to the next item

    # Calculate accuracy
    correct_predictions = [1 if pred == true else 0 for pred, true in zip(predictions, true_labels)]
    accuracy = sum(correct_predictions) / len(correct_predictions) * 100 if true_labels else 0

    # Log accuracy
    accuracy_log_path = os.path.join(accuracy_log, os.path.basename(save_file_main_model))
    with open(accuracy_log_path, "a", encoding="utf-8") as file:
        log_entry = f"ModelId: {save_file_main_model}\nFilename: {file_name.replace('.jsonl', '')}\nAccuracy: {accuracy:.2f}%\n\n"
        file.write(log_entry)

    print(f"Accuracy for {file_name}: {accuracy:.2f}%")

except Exception as e:
    print(f"An error occurred while processing {file_name}: {e}")