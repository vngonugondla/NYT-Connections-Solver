import kagglehub
import os
import pandas as pd
import json

path = kagglehub.dataset_download("christophersinger/new-york-times-connections-archive")

df = pd.read_csv(os.path.join(path, "connections_clean.csv"))

print(df.head())
examples = []

for date, group_df in df.groupby("date"):
    all_words = []
    output_groups = []

    for _, row in group_df.iterrows():
        words = [str(w).strip() for w in [row["w1"], row["w2"], row["w3"], row["w4"]] if pd.notna(w)]
        if len(words) != 4:
            break

        all_words.extend(words)
        output_groups.append(f"{row['category']}: {', '.join(words)}")

    if len(output_groups) != 4 or len(all_words) != 16:
        continue

with open("nyt_dataset.json", "w") as f:
    json.dump(examples, f, indent=2)