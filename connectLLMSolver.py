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


if __name__ == "__main__":
    # load the first puzzle from the dataset
    puzzle = data[0]["input"]
    groups = generate_candidate_groups_baseline(puzzle)
    print(groups)
    print(len(groups))
