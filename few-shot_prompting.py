from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import random
import numpy as np
from itertools import combinations

load_dotenv()

# Initialize OpenAI
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL = "gpt-4-turbo"
NUM_FEWSHOT = 3

# Load dataset
with open("nyt_dataset.json", "r") as f:
    dataset = json.load(f)


def extract_groups(text):
    groups = []
    for segment in text.split(';'):
        words = segment.split(':')[-1] if ':' in segment else segment
        group = [word.strip().upper() for word in words.split(',')]
        if len(group) == 4:
            groups.append(sorted(group))
    return groups

def format_groups_output(text):
    groups = extract_groups(text)
    return '\n'.join([f"{i+1}. [{', '.join(group)}]" for i, group in enumerate(groups)])

def parse_model_output(text):
    lines = text.strip().split('\n')
    groups = []
    for line in lines:
        if '.' in line and '[' in line:
            content = line.split('.', 1)[-1]
            content = content.strip().strip('[]')
            words = [word.strip().upper() for word in content.split(',')]
            if len(words) == 4:
                groups.append(sorted(words))
    return groups

def groups_match(pred_groups, true_groups):
    matched = 0
    used = set()
    for pg in pred_groups:
        for i, tg in enumerate(true_groups):
            if i not in used and pg == tg:
                matched += 1
                used.add(i)
                break
    return matched

def group_to_pairs(group_sets):
    pairs = set()
    for group in group_sets:
        for a, b in combinations(sorted(group), 2):
            pairs.add(frozenset([a, b]))
    return pairs

def compute_f1(predicted_sets, gold_sets):
    pred_pairs = group_to_pairs(predicted_sets)
    gold_pairs = group_to_pairs(gold_sets)

    if not pred_pairs and not gold_pairs:
        return 0.0, 0.0, 0.0

    tp = len(pred_pairs & gold_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1

def build_messages_with_fewshot(few_shot, test_input):
    messages = []

    for example in few_shot:
        words = example["input"].split("Group the following words into 4 meaningful categories:")[-1].strip()
        formatted_groups = format_groups_output(example["output"])
        user_prompt = f"""Group the following 16 words into 4 categories of 4 words each based on common themes:

Words: {words}

Only return the list of groups. Do not include any explanations or labels.

Output format:
1. [group of 4 words]
2. [group of 4 words]
3. [group of 4 words]
4. [group of 4 words]"""
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": formatted_groups})

    test_words = test_input.split("Group the following words into 4 meaningful categories:")[-1].strip()
    test_prompt = f"""Group the following 16 words into 4 categories of 4 words each based on common themes:

Words: {test_words}

Only return the list of groups. Do not include any explanations or labels.

Output format:
1. [group of 4 words]
2. [group of 4 words]
3. [group of 4 words]
4. [group of 4 words]"""

    messages.append({"role": "user", "content": test_prompt})
    return messages

correct_puzzles = 0
correct_groups_total = 0
total_puzzles = len(dataset)
f1_scores = []

for i, test_item in enumerate(dataset):
    few_shot = random.sample([ex for ex in dataset if ex != test_item], NUM_FEWSHOT)
    messages = build_messages_with_fewshot(few_shot, test_item["input"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3
    )

    pred_output = response.choices[0].message.content
    pred_groups = parse_model_output(pred_output)
    true_groups = extract_groups(test_item["output"])

    correct_groups = groups_match(pred_groups, true_groups)
    correct_groups_total += correct_groups
    if correct_groups == 4:
        correct_puzzles += 1

    precision, recall, f1 = compute_f1(pred_groups, true_groups)
    f1_scores.append(f1)

    print(f"\nPuzzle #{i+1}")
    print("Words:", test_item["input"].split(":")[-1].strip())
    print("Model Output:\n", pred_output)
    print("Matched groups:", correct_groups, "/ 4")
    print(f"F1: {f1:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f}")


# Print final scores
accuracy_puzzle = (correct_puzzles / total_puzzles)
accuracy_groups = (correct_groups_total / (total_puzzles * 4))

print(f"\nPuzzle-Level Accuracy: {accuracy_puzzle:.2%}")
print(f"Group-Level Accuracy: {accuracy_groups:.2%}")
print(f"Average F1 Score: {np.mean(f1_scores):.2f}")