import numpy as np
from openai import OpenAI
import json
import re
import time

# Load dataset
with open("nyt_dataset.json", "r") as f:
    data = json.load(f)

# Initialize OpenAI
client = OpenAI(
    api_key="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"
)

from itertools import combinations

def group_to_pairs(group_sets):
    pairs = set()
    for group in group_sets:
        for a, b in combinations(sorted(group), 2):
            pairs.add(frozenset([a, b]))
    return pairs

def compute_f1(predicted_sets, gold_sets):
    pred_pairs = group_to_pairs(predicted_sets)
    gold_pairs = group_to_pairs(gold_sets)

    tp = len(pred_pairs & gold_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1




def extract_groups(output_text):
    def normalize(word):
        return word.strip().upper().strip("[](){}.,;:-_\"'")

    groups = []

    lines = output_text.strip().split("\n")
    for line in lines:
        match = re.match(r"\d+\.\s*(.*)", line)
        if match:
            words = [normalize(w) for w in match.group(1).split(",") if w.strip()]
            if len(words) == 4:
                groups.append(set(words))


    if not groups and ";" in output_text:
        parts = output_text.split(";")
        for part in parts:
            match = re.match(r".*?:\s*(.*)", part)
            if match:
                words = [normalize(w) for w in match.group(1).split(",") if w.strip()]
                if len(words) == 4:
                    groups.append(set(words))

    return groups


def evaluate_groups(predicted, expected):
    matched = 0
    used = set()
    for pg in predicted:
        for i, eg in enumerate(expected):
            if i in used:
                continue
            if pg == eg:
                matched += 1
                used.add(i)
                break
    return matched

if __name__ == "__main__":
    # Accuracy counters
    puzzle_correct = 0
    puzzle_total = 0
    group_match_total = 0
    group_possible_total = 0
    f1_scores = []

    # Evaluate N puzzles
    N = 20

    for i, item in enumerate(data[:N]):
        words = item["input"].split(":")[1].strip()
        expected_groups = extract_groups(item["output"])

        prompt = f"""Group the following 16 words into 4 categories of 4 words each based on common themes:

    Words: {words}

    Only return the list of groups. Do not include any explanations or labels.

    Output format:
    1. [group of 4 words]
    2. [group of 4 words]
    3. [group of 4 words]
    4. [group of 4 words]"""

        try:
            response = client.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            model_output = response.choices[0].message.content
            predicted_groups = extract_groups(model_output)

            matched_groups = evaluate_groups(predicted_groups, expected_groups)
            precision, recall, f1 = compute_f1(predicted_groups, expected_groups)
            print(f"F1: {f1:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f}")
            f1_scores.append(f1)

            group_match_total += matched_groups
            group_possible_total += len(expected_groups)

            if expected_groups and matched_groups == len(expected_groups):
                puzzle_correct += 1
            puzzle_total += 1

            print(f"\n Puzzle #{i+1}")
            print(f"Words: {words}")
            print(f"Model Output:\n{model_output}")
            print(f"Matched groups: {matched_groups}/{len(expected_groups)}")

            time.sleep(1)

        except Exception as e:
            print(f"Error on puzzle #{i+1}: {e}")

    # Print final scores
    accuracy_puzzle = puzzle_correct / puzzle_total if puzzle_total else 0
    accuracy_groups = group_match_total / group_possible_total if group_possible_total else 0

    print(f"\n Puzzle-Level Accuracy: {accuracy_puzzle:.2%}")
    print(f"Group-Level Accuracy: {accuracy_groups:.2%}")
    print(f"Average F1 Score: {np.mean(f1_scores):.2f}")
