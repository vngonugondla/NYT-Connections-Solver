from itertools import combinations
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import re
import requests

from api import client

from connectLLMSolver import generate_candidate_groups_baseline, generate_candidate_groups_few_shot

def clean_input(input_text):
    return [word.strip().upper() for word in input_text.split(":")[-1].split(",")]

def extract_answer_groups(output_text):
    group_texts = output_text.strip().split(";")
    gold_groups = []
    for group in group_texts:
        try:
            label, members = group.split(":")
            words = [w.strip().upper() for w in members.split(",")]
            gold_groups.append(set(words))
        except ValueError:
            print("Malformed group:", group)
    return gold_groups

def score_group(group, word_to_embedding):
    # Convert set to list if it's a set
    group_list = list(group) if isinstance(group, set) else group
    return np.mean([
        util.cos_sim(word_to_embedding[group_list[i]], word_to_embedding[group_list[j]]).item()
        for i in range(len(group_list)) for j in range(i + 1, len(group_list))
    ])

def gpt4_score_group(group, client):
    # Convert set to list if it's a set
    group_list = list(group) if isinstance(group, set) else group
    
    prompt = f"""Rate how well these 4 words form a coherent thematic group on a scale of 0.0 to 1.0. 
    Only return the numerical score, no explanation.
    Words: {', '.join(group_list)}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        score = float(response.choices[0].message.content.strip())
        return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
    except Exception as e:
        print(f"Error getting GPT-4 score: {e}")
        return 0.0

def score_groups_with_gpt4(all_groups, client, cache={}):
    """Score all possible groups using GPT-4, with caching"""
    scored_groups = []
    
    for group in all_groups:
        # Convert group to frozenset for hashable cache key
        group_key = frozenset(group)
        
        if group_key in cache:
            score = cache[group_key]
        else:
            score = gpt4_score_group(group, client)
            cache[group_key] = score
            
        scored_groups.append((group, score))
    
    # Sort by score in descending order
    scored_groups.sort(key=lambda x: x[1], reverse=True)
    return scored_groups

def hybrid_score_group(group, word_to_embedding, client, embedding_weight=0.3, gpt_weight=0.7):
    """Combine embedding-based and GPT-4 scoring"""
    embedding_score = score_group(group, word_to_embedding)
    gpt_score = gpt4_score_group(group, client)
    return (embedding_score * embedding_weight) + (gpt_score * gpt_weight)

def ollama_score_group(group):
    """Score a group using Ollama's model"""
    # Convert set to list if it's a set
    group_list = list(group) if isinstance(group, set) else group
    
    system_message = "You are a scoring assistant that ONLY outputs a single number between 0.0 and 1.0. No other text or explanation."
    prompt = f"Rate how coherent these words are as a thematic group: {', '.join(group_list)}. Output only a number between 0.0 and 1.0."

    try:
        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   "model": "llama3.2",
                                   "system": system_message,
                                   "prompt": prompt,
                                   "stream": False,
                                   "temperature": 0.1,  # Very low temperature for more deterministic output
                                   "top_p": 0.1,       # Lower top_p for more focused sampling
                                   "num_predict": 10    # Limit the number of tokens to predict
                               })
        if response.status_code == 200:
            result = response.json()
            score_text = result['response'].strip()
            # Remove any non-numeric characters except decimal point
            score_text = ''.join(c for c in score_text if c.isdigit() or c == '.')
            try:
                score = float(score_text)
                return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
            except ValueError:
                print(f"Could not parse score from response: {score_text}")
                return 0.5
        else:
            print(f"Error from Ollama API: {response.status_code}")
            return 0.5
    except Exception as e:
        print(f"Error getting Ollama score: {e}")
        return 0.5

def score_groups_with_ollama(all_groups, cache={}):
    """Score all possible groups using Ollama, with caching"""
    scored_groups = []
    
    for group in all_groups:
        # Convert group to frozenset for hashable cache key
        group_key = frozenset(group)
        
        if group_key in cache:
            score = cache[group_key]
        else:
            score = ollama_score_group(group)
            cache[group_key] = score
            
        scored_groups.append((group, score))
    
    # Sort by score in descending order
    scored_groups.sort(key=lambda x: x[1], reverse=True)
    return scored_groups

def run_ollama_test(puzzle_input): 
    # Initialize scoring cache
    ollama_score_cache = {}

    # Generate all possible groups
    words = clean_input(puzzle_input)
    all_groups = list(combinations(words, 4))

    # Score groups using Ollama
    scored_groups = score_groups_with_ollama(all_groups, ollama_score_cache)

    # Run beam search with Ollama scores
    solution = beam_search_solver(scored_groups, beam_width=3)

    return solution

# Example usage with beam search:

def run_gpt4_embedding_test(client, puzzle_input): 
    
    # Initialize scoring cache
    gpt4_score_cache = {}

    # Generate all possible groups
    words = clean_input(puzzle_input)

    
    all_groups = list(combinations(words, 4))


    # Score groups using GPT-4
    scored_groups = score_groups_with_gpt4(all_groups, client, gpt4_score_cache)

    # Run beam search with GPT-4 scores
    solution = beam_search_solver(scored_groups, beam_width=3)

    return solution
    


def solve(scored_groups, used_words=set(), selected=[], depth=0):
    if len(selected) == 4:
        return selected
    for i, (group, score) in enumerate(scored_groups):
        group_set = set(group)
        if group_set & used_words:
            continue
        result = solve(scored_groups[i + 1:], used_words | group_set, selected + [(group, score)], depth + 1)
        if result:
            return result
    return None

def beam_search_solver(scored_groups, beam_width=3):
    beams = [([], set(), 0.0)]

    for step in range(4):
        new_beams = []
        for selected, used_words, score_so_far in beams:
            for group, score in scored_groups:
                group_set = set(group)
                if group_set & used_words:
                    continue
                new_selected = selected + [(group, score)]
                new_used = used_words | group_set
                new_score = score_so_far + score
                new_beams.append((new_selected, new_used, new_score))
        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = new_beams[:beam_width]

    return beams[0][0] if beams else None

def group_to_pairs(group_sets):
    """Convert a list of groups into a set of unordered word pairs."""
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


with open("nyt_dataset.json", "r") as f:
    puzzles = json.load(f)

def run_test(num_puzzles=None):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # If num_puzzles is specified, limit the dataset
    if num_puzzles is not None:
        puzzles_to_test = puzzles[:num_puzzles]
    else:
        puzzles_to_test = puzzles


    correct_count = 0
    llm_correct_count = 0
    few_shot_correct_count = 0
    correct_groups_total = 0
    llm_correct_groups_total = 0
    few_shot_correct_groups_total = 0

f1_scores = []

for idx, puzzle in enumerate(puzzles):
    words = clean_input(puzzle["input"])
    gold_sets = extract_answer_groups(puzzle["output"])


    total = len(puzzles_to_test)

    for idx, puzzle in enumerate(puzzles_to_test):
        words = clean_input(puzzle["input"])
        gold_sets = extract_answer_groups(puzzle["output"])

        embeddings = model.encode(words)
        word_to_embedding = {word: emb for word, emb in zip(words, embeddings)}

        all_groups = list(combinations(words, 4))

        llm_generated_groups = generate_candidate_groups_baseline(puzzle["input"])
        few_shot_generated_groups = generate_candidate_groups_few_shot(puzzle["input"])
        
        # Convert sets to lists for scoring
        llm_generated_groups = [list(group) for group in llm_generated_groups]
        few_shot_generated_groups = [list(group) for group in few_shot_generated_groups]

        scored_groups = [(group, score_group(group, word_to_embedding)) for group in all_groups]
        scored_groups.sort(key=lambda x: x[1], reverse=True)

        llm_scored_groups = [(group, score_group(group, word_to_embedding)) for group in llm_generated_groups]
        llm_scored_groups.sort(key=lambda x: x[1], reverse=True)

        few_shot_scored_groups = [(group, score_group(group, word_to_embedding)) for group in few_shot_generated_groups]
        few_shot_scored_groups.sort(key=lambda x: x[1], reverse=True)

        solution = beam_search_solver(scored_groups)
        llm_solution = beam_search_solver(llm_scored_groups)
        few_shot_solution = beam_search_solver(few_shot_scored_groups)

        gold_sets = extract_answer_groups(puzzle["output"])

        print(f"\nEvaluating Puzzle #{idx + 1}")
        print("-" * 40)

        # Evaluate Brute Force solution
        if solution:
            predicted_sets = [set(word.upper().strip() for word in group) for group, _ in solution]
            gold_used = set()
            num_correct_groups = 0

            for pred in predicted_sets:
                for i, gold in enumerate(gold_sets):
                    if i not in gold_used and pred == gold:
                        num_correct_groups += 1
                        gold_used.add(i)
                        break

            correct_groups_total += num_correct_groups
            pred_fs = set(frozenset(g) for g in predicted_sets)
            gold_fs = set(frozenset(g) for g in gold_sets)

            if pred_fs == gold_fs:
                print(f"✅ Brute Force: Fully Correct")
                correct_count += 1
            else:
                print(f"❌ Brute Force: Mismatch")
                print("Predicted:")
                for g in predicted_sets:
                    print(" ", sorted(g))
        else:
            print("⚠️ Brute Force: No solution found")

        # Evaluate LLM Baseline solution
        if llm_solution:
            llm_predicted_sets = [set(word.upper().strip() for word in group) for group, _ in llm_solution]
            gold_used = set()
            llm_num_correct_groups = 0

            for pred in llm_predicted_sets:
                for i, gold in enumerate(gold_sets):
                    if i not in gold_used and pred == gold:
                        llm_num_correct_groups += 1
                        gold_used.add(i)
                        break

            llm_correct_groups_total += llm_num_correct_groups
            llm_pred_fs = set(frozenset(g) for g in llm_predicted_sets)

            if llm_pred_fs == gold_fs:
                print(f"✅ LLM Baseline: Fully Correct")
                llm_correct_count += 1
            else:
                print(f"❌ LLM Baseline: Mismatch")
                print("LLM Predicted:")
                for g in llm_predicted_sets:
                    print(" ", sorted(g))
        else:
            print("⚠️ LLM Baseline: No solution found")

        # Evaluate Few-Shot solution
        if few_shot_solution:
            few_shot_predicted_sets = [set(word.upper().strip() for word in group) for group, _ in few_shot_solution]
            gold_used = set()
            few_shot_num_correct_groups = 0

            for pred in few_shot_predicted_sets:
                for i, gold in enumerate(gold_sets):
                    if i not in gold_used and pred == gold:
                        few_shot_num_correct_groups += 1
                        gold_used.add(i)
                        break

            few_shot_correct_groups_total += few_shot_num_correct_groups
            few_shot_pred_fs = set(frozenset(g) for g in few_shot_predicted_sets)

            if few_shot_pred_fs == gold_fs:
                print(f"✅ Few-Shot: Fully Correct")
                few_shot_correct_count += 1
            else:
                print(f"❌ Few-Shot: Mismatch")
                print("Few-Shot Predicted:")
                for g in few_shot_predicted_sets:
                    print(" ", sorted(g))
        else:
            print("⚠️ Few-Shot: No solution found")

        print("\nGold Standard Groups:")
        for g in gold_sets:
            print(" ", sorted(g))
        print("-" * 40)


    print(f"\nResults for {total} puzzles:")
    print(f"Considering All Groups fully correct puzzle accuracy: {correct_count} / {total} = {correct_count / total:.2%}")
    print(f"Considering All Groups average correct groups per puzzle: {correct_groups_total / total:.2f}")
    print(f"LLM Baseline fully correct puzzle accuracy: {llm_correct_count} / {total} = {llm_correct_count / total:.2%}")
    print(f"LLM Baseline average correct groups per puzzle: {llm_correct_groups_total / total:.2f}")
    print(f"Few-Shot fully correct puzzle accuracy: {few_shot_correct_count} / {total} = {few_shot_correct_count / total:.2%}")
    print(f"Few-Shot average correct groups per puzzle: {few_shot_correct_groups_total / total:.2f}")
if __name__ == "__main__":
    # Run test on first 5 puzzles by default
    # run_test(num_puzzles=50)
    print("Testing with GPT-4:")
    # print(run_gpt4_embedding_test(client, puzzles[0]["input"]))
    print("\nTesting with Ollama:")
    print(run_ollama_test(puzzles[0]["input"]))

