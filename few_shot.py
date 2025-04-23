import json
from openai import OpenAI

# Initialize OpenAI
client = OpenAI(api_key="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")
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

# Evaluate N puzzles
N = 20
correct_puzzles = 0
correct_groups_total = 0
test_items = dataset[:N]

for i, test_item in enumerate(test_items):
    few_shot = dataset[i+1:i+1+NUM_FEWSHOT] if i+1+NUM_FEWSHOT <= len(dataset) else dataset[-NUM_FEWSHOT:]
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

    print(f"\nPuzzle #{i+1}")
    print("Words:", test_item["input"].split(":")[-1].strip())
    print("Model Output:\n", pred_output)
    print("Matched groups:", correct_groups, "/ 4")


# Print final scores
accuracy_puzzle = (correct_puzzles / N)
accuracy_groups = (correct_groups_total / (N * 4))

print(f"\nPuzzle-Level Accuracy: {accuracy_puzzle:.2%}")
print(f"Group-Level Accuracy: {accuracy_groups:.2%}")

