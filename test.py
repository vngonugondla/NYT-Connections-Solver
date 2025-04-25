from itertools import combinations
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import re
import requests

from baseline import client

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
    group_list = list(group) if isinstance(group, set) else group
    return np.mean([
        util.cos_sim(word_to_embedding[group_list[i]], word_to_embedding[group_list[j]]).item()
        for i in range(len(group_list)) for j in range(i + 1, len(group_list))
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

def run_test(num_puzzles=None):
    model = SentenceTransformer("all-MiniLM-L6-v2")

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
    llm_f1_scores = []
    few_shot_f1_scores = []

    greedy_embed_correct_count = 0
    greedy_embed_f1_scores = []
    greedy_llm_correct_count = 0
    greedy_llm_f1_scores = []
    greedy_fs_correct_count = 0
    greedy_fs_f1_scores = []
    greedy_embed_groups_total = 0
    greedy_llm_groups_total = 0
    greedy_fs_groups_total = 0

    total = len(puzzles_to_test)

    for idx, puzzle in enumerate(puzzles_to_test):
        words = clean_input(puzzle["input"])
        gold_sets = extract_answer_groups(puzzle["output"])

        embeddings = model.encode(words)
        word_to_embedding = {word: emb for word, emb in zip(words, embeddings)}

        all_groups = list(combinations(words, 4))

        llm_generated_groups = generate_candidate_groups_baseline(puzzle["input"])
        few_shot_generated_groups = generate_candidate_groups_few_shot(puzzle["input"])
        
        llm_generated_groups = [list(group) for group in llm_generated_groups]
        few_shot_generated_groups = [list(group) for group in few_shot_generated_groups]

        llm_generated_groups = [
            [word for word in group if word in word_to_embedding]
            for group in llm_generated_groups
        ]
        llm_generated_groups = [group for group in llm_generated_groups if len(group) == 4]

        few_shot_generated_groups = [
            [word for word in group if word in word_to_embedding]
            for group in few_shot_generated_groups
        ]
        few_shot_generated_groups = [group for group in few_shot_generated_groups if len(group) == 4]

        llm_scored_groups = [(group, score_group(group, word_to_embedding)) for group in llm_generated_groups]
        llm_scored_groups.sort(key=lambda x: x[1], reverse=True)

        few_shot_scored_groups = [(group, score_group(group, word_to_embedding)) for group in few_shot_generated_groups]
        few_shot_scored_groups.sort(key=lambda x: x[1], reverse=True)

        scored_groups = [(group, score_group(group, word_to_embedding)) for group in all_groups]
        scored_groups.sort(key=lambda x: x[1], reverse=True)

        solution = beam_search_solver(scored_groups)
        llm_solution = beam_search_solver(llm_scored_groups)
        few_shot_solution = beam_search_solver(few_shot_scored_groups)

        gold_sets = extract_answer_groups(puzzle["output"])

        print(f"\nEvaluating Puzzle #{idx + 1}")
        print("-" * 40)

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

            precision, recall, f1 = compute_f1(predicted_sets, gold_sets)
            f1_scores.append(f1)

            if pred_fs == gold_fs:
                print(f"✅ Brute Force: Fully Correct")
                correct_count += 1
            else:
                print(f"❌ Brute Force: Mismatch")
                print("Predicted:")
                for g in predicted_sets:
                    print(" ", sorted(g))
            print(f"F1: {f1:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f}")
        else:
            print("⚠️ Brute Force: No solution found")
            f1_scores.append(0.0)

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

            llm_precision, llm_recall, llm_f1 = compute_f1(llm_predicted_sets, gold_sets)
            llm_f1_scores.append(llm_f1)

            if llm_pred_fs == gold_fs:
                print(f"✅ LLM Baseline: Fully Correct")
                llm_correct_count += 1
            else:
                print(f"❌ LLM Baseline: Mismatch")
                print("LLM Predicted:")
                for g in llm_predicted_sets:
                    print(" ", sorted(g))
            print(f"F1: {llm_f1:.2f} | Precision: {llm_precision:.2f} | Recall: {llm_recall:.2f}")
        else:
            print("⚠️ LLM Baseline: No solution found")
            llm_f1_scores.append(0.0)

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

            few_shot_precision, few_shot_recall, few_shot_f1 = compute_f1(few_shot_predicted_sets, gold_sets)
            few_shot_f1_scores.append(few_shot_f1)

            if few_shot_pred_fs == gold_fs:
                print(f"✅ Few-Shot: Fully Correct")
                few_shot_correct_count += 1
            else:
                print(f"❌ Few-Shot: Mismatch")
                print("Few-Shot Predicted:")
                for g in few_shot_predicted_sets:
                    print(" ", sorted(g))
            print(f"F1: {few_shot_f1:.2f} | Precision: {few_shot_precision:.2f} | Recall: {few_shot_recall:.2f}")
        else:
            print("⚠️ Few-Shot: No solution found")
            few_shot_f1_scores.append(0.0)

        gold_fs = set(map(frozenset, gold_sets))
        embed_sol = solve(scored_groups)
        if embed_sol:
            ge_used = set()
            ge_match_count = 0
            for pred in [set(group) for group, _ in embed_sol]:
                for i, gold in enumerate(gold_sets):
                    if i not in ge_used and pred == gold:
                        ge_match_count += 1
                        ge_used.add(i)
                        break
            greedy_embed_groups_total += ge_match_count
            embed_pred_sets = [set(group) for group, _ in embed_sol]
            _, _, ge_f1 = compute_f1(embed_pred_sets, gold_sets)
            greedy_embed_f1_scores.append(ge_f1)
            if set(map(frozenset, embed_pred_sets)) == gold_fs:
                greedy_embed_correct_count += 1
        else:
            greedy_embed_f1_scores.append(0.0)

        llm_sol = solve(llm_scored_groups)
        if llm_sol:
            ll_used = set()
            ll_match_count = 0
            for pred in [set(group) for group, _ in llm_sol]:
                for i, gold in enumerate(gold_sets):
                    if i not in ll_used and pred == gold:
                        ll_match_count += 1
                        ll_used.add(i)
                        break
            greedy_llm_groups_total += ll_match_count
            llm_pred_sets = [set(group) for group, _ in llm_sol]
            _, _, gl_f1 = compute_f1(llm_pred_sets, gold_sets)
            greedy_llm_f1_scores.append(gl_f1)
            if set(map(frozenset, llm_pred_sets)) == gold_fs:
                greedy_llm_correct_count += 1
        else:
            greedy_llm_f1_scores.append(0.0)

        fs_sol = solve(few_shot_scored_groups)
        if fs_sol:
            fs_used = set()
            fs_match_count = 0
            for pred in [set(group) for group, _ in fs_sol]:
                for i, gold in enumerate(gold_sets):
                    if i not in fs_used and pred == gold:
                        fs_match_count += 1
                        fs_used.add(i)
                        break
            greedy_fs_groups_total += fs_match_count
            fs_pred_sets = [set(group) for group, _ in fs_sol]
            _, _, fs_f1 = compute_f1(fs_pred_sets, gold_sets)
            greedy_fs_f1_scores.append(fs_f1)
            if set(map(frozenset, fs_pred_sets)) == gold_fs:
                greedy_fs_correct_count += 1
        else:
            greedy_fs_f1_scores.append(0.0)

        print("\nGold Standard Groups:")
        for g in gold_sets:
            print(" ", sorted(g))
        print("-" * 40)


    print(f"\nResults for {total} puzzles:")
    print(f"Considering All Groups fully correct puzzle accuracy: {correct_count} / {total} = {correct_count / total:.2%}")
    print(f"Considering All Groups average correct groups per puzzle: {correct_groups_total / total:.2f}")
    print(f"Considering All Groups average F1 score: {np.mean(f1_scores):.2f}")
    
    print(f"LLM Baseline fully correct puzzle accuracy: {llm_correct_count} / {total} = {llm_correct_count / total:.2%}")
    print(f"LLM Baseline average correct groups per puzzle: {llm_correct_groups_total / total:.2f}")
    print(f"LLM Baseline average F1 score: {np.mean(llm_f1_scores):.2f}")
    
    print(f"Few-Shot fully correct puzzle accuracy: {few_shot_correct_count} / {total} = {few_shot_correct_count / total:.2%}")
    print(f"Few-Shot average correct groups per puzzle: {few_shot_correct_groups_total / total:.2f}")
    print(f"Few-Shot average F1 score: {np.mean(few_shot_f1_scores):.2f}")

    print(f"Greedy Embedding fully correct puzzle accuracy: {greedy_embed_correct_count} / {total} = {greedy_embed_correct_count/total:.2%}")
    print(f"Greedy Embedding average correct groups per puzzle: {greedy_embed_groups_total/total:.2f}")
    print(f"Greedy Embedding average F1 score: {np.mean(greedy_embed_f1_scores):.2f}")

    print(f"Greedy LLM fully correct puzzle accuracy: {greedy_llm_correct_count} / {total} = {greedy_llm_correct_count/total:.2%}")
    print(f"Greedy LLM average correct groups per puzzle: {greedy_llm_groups_total/total:.2f}")
    print(f"Greedy LLM average F1 score: {np.mean(greedy_llm_f1_scores):.2f}")

    print(f"Greedy Few-Shot fully correct puzzle accuracy: {greedy_fs_correct_count} / {total} = {greedy_fs_correct_count/total:.2%}")
    print(f"Greedy Few-Shot average correct groups per puzzle: {greedy_fs_groups_total/total:.2f}")
    print(f"Greedy Few-Shot average F1 score: {np.mean(greedy_fs_f1_scores):.2f}")


if __name__ == "__main__":
    run_test(num_puzzles=50)
