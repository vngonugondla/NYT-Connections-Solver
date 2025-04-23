from itertools import combinations
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import re

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
    return np.mean([
        util.cos_sim(word_to_embedding[group[i]], word_to_embedding[group[j]]).item()
        for i in range(4) for j in range(i + 1, 4)
    ])

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

model = SentenceTransformer("all-MiniLM-L6-v2")

correct_count = 0
correct_groups_total = 0
total = len(puzzles)

f1_scores = []

for idx, puzzle in enumerate(puzzles):
    words = clean_input(puzzle["input"])
    gold_sets = extract_answer_groups(puzzle["output"])

    embeddings = model.encode(words)
    word_to_embedding = {word: emb for word, emb in zip(words, embeddings)}

    all_groups = list(combinations(words, 4))
    scored_groups = [(group, score_group(group, word_to_embedding)) for group in all_groups]
    scored_groups.sort(key=lambda x: x[1], reverse=True)

    solution = beam_search_solver(scored_groups)

    if solution:
        predicted_sets = [set(word.upper().strip() for word in group) for group, _ in solution]
        gold_sets = extract_answer_groups(puzzle["output"])
        gold_used = set()
        num_correct_groups = 0

        for pred in predicted_sets:
            for i, gold in enumerate(gold_sets):
                if i not in gold_used and pred == gold:
                    num_correct_groups += 1
                    gold_used.add(i)
                    break

        correct_groups_total += num_correct_groups
        predicted_sets = [set(word.upper().strip() for word in group) for group, _ in solution]

        precision, recall, f1 = compute_f1(predicted_sets, gold_sets)
        f1_scores.append(f1)
        print(f"Puzzle #{idx + 1} — Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        pred_fs = set(frozenset(g) for g in predicted_sets)
        gold_fs = set(frozenset(g) for g in gold_sets)

        if pred_fs == gold_fs:
            correct_count += 1
        else:
            print(f"Puzzle #{idx + 1} mismatch")
            print("Predicted:")
            for g in predicted_sets:
                print(" ", sorted(g))
            print("Gold:")
            for g in gold_sets:
                print(" ", sorted(g))
            print("-" * 40)
    else:
        print(f"Puzzle #{idx + 1} — no solution found by solver.")
        print("Gold:")
        for g in gold_sets:
            print(" ", sorted(g))
        print("-" * 40)

print(f"\nFully correct puzzle accuracy: {correct_count} / {total} = {correct_count / total:.2%}")
print(f"Average number of correct groups per puzzle: {correct_groups_total / total:.2f}")
print(f"\nAverage F1 Score across puzzles: {np.mean(f1_scores):.2f}")