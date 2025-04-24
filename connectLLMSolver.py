import json
from api import evaluate_groups, client


with open("nyt_dataset.json", "r") as f:
    data = json.load(f)


# function to generate N candidate groups given a puzzle 
def generate_candidate_groups_baseline(puzzle, num_groups=40):
    
    prompt = f"""Generate {num_groups} candidate groups of 4 words each from the given puzzle. Return the response in the following JSON format:
    {{
        "groups": [
            ["WORD1", "WORD2", "WORD3", "WORD4"],
            ["WORD1", "WORD2", "WORD3", "WORD4"],
            ...
        ]
    }}
    Make sure all words are in UPPERCASE.
    
    Puzzle: {puzzle}"""
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={ "type": "json_object" }
    )

    try:
        # Parse the JSON response
        model_output = json.loads(response.choices[0].message.content)
        # Convert lists to sets
        groups = [set(group) for group in model_output["groups"]]
        return groups
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error parsing LLM response: {e}")
        return []


def generate_candidate_groups_few_shot(puzzle, num_groups=40):
    # Create few-shot examples from the dataset with explicit JSON format
    few_shot_prompt = f"""Given a set of 16 words, generate exactly {num_groups} groups of 4 related words each. Each group should contain exactly 4 words from the puzzle, and words can be reused across different groups.

Here's an example input and its expected output format:

Input: OLIVE, ESPRESSO, LATTE, WALNUT, AMERICANO, CAPPUCCINO, BEAN, KELLY, FOREST, CASHEW, PECAN, EMERALD, FOX, ALMOND, PEANUT, CLEAN

The output should be EXACTLY in this JSON format, with {num_groups} arrays of 4 words each:
{{
    "groups": [
        ["AMERICANO", "CAPPUCCINO", "ESPRESSO", "LATTE"],
        ["ALMOND", "CASHEW", "PECAN", "WALNUT"],
        ["EMERALD", "FOREST", "KELLY", "OLIVE"],
        ["BEAN", "CLEAN", "FOX", "PEANUT"],
        ["AMERICANO", "ALMOND", "BEAN", "OLIVE"],
        ["FOREST", "FOX", "PECAN", "WALNUT"],
    ]
}}

Important rules:
1. Return EXACTLY {num_groups} groups
2. Each group MUST contain EXACTLY 4 words
3. Use ONLY words from the input puzzle
4. All words MUST be in UPPERCASE
5. Each group must have a logical connection (like drinks, colors, animals, etc.)
6. The first few groups should be the strongest/most obvious connections
7. Later groups can be more creative but must still make logical sense
8. Format the output EXACTLY like the JSON example above
9. Do not include any numbers, only arrays of words

Now, create {num_groups} groups for this new puzzle:
{puzzle}

Return ONLY a valid JSON object with your answer."""

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": few_shot_prompt}],
        temperature=0.8,
        response_format={ "type": "json_object" },
        max_tokens=2000  # Increased to ensure we get all groups
    )

    try:
        # Parse the JSON response
        model_output = json.loads(response.choices[0].message.content)
        # Validate the response format
        if not isinstance(model_output, dict) or "groups" not in model_output:
            raise ValueError("Invalid response format: missing 'groups' key")
        if not isinstance(model_output["groups"], list):
            raise ValueError("Invalid response format: 'groups' is not a list")
        for group in model_output["groups"]:
            if not isinstance(group, list) or len(group) != 4:
                raise ValueError("Invalid group format: each group must be a list of exactly 4 strings")
            if not all(isinstance(word, str) for word in group):
                raise ValueError("Invalid group format: all elements must be strings")
        
        # Convert lists to sets
        groups = [set(group) for group in model_output["groups"]]
        return groups
    except (json.JSONDecoder.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"Error parsing LLM response: {e}")
        return []


if __name__ == "__main__":
    # load the first puzzle from the dataset
    puzzle = data[-1]["input"]


    groups = generate_candidate_groups_baseline(puzzle)
    print(groups)
    print(len(groups))

    groups = generate_candidate_groups_few_shot(puzzle)
    print(puzzle)
    print(groups)
    print(len(groups))
