from itertools import combinations
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json

words = [
    "red", "yellow", "Honda", "Tesla",
    "banana", "orange", "bracelet", "ring",
    "green", "blue", "apple", "mango",
    "Ford", "Toyota", "necklace", "earring"
]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(words)
word_to_embedding = {word: emb for word, emb in zip(words, embeddings)}

def score_group(group):
    sims = []
    for i in range(4):
        for j in range(i + 1, 4):
            sim = util.cos_sim(word_to_embedding[group[i]], word_to_embedding[group[j]]).item()
            sims.append(sim)
    return np.mean(sims)

all_groups = list(combinations(words, 4))
scored_groups = [(group, score_group(group)) for group in all_groups]
scored_groups.sort(key=lambda x: x[1], reverse=True)

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

solution = solve(scored_groups)

result = []
if solution:
    for idx, (group, score) in enumerate(solution, 1):
        result.append({
            "group_number": idx,
            "words": list(group),
            "avg_similarity": round(score, 4)
        })

with open("final_connection_groups.json", "w") as f:
    json.dump(result, f, indent=2)
