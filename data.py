# The following code will only execute
# successfully when compression is complete

import kagglehub
import os
import pandas as pd
import json
import random

# Download latest version
path = kagglehub.dataset_download("christophersinger/new-york-times-connections-archive")

print("Path to dataset files:", path)

df = pd.read_csv(os.path.join(path, "connections_clean.csv"))

# Show first 5 rows
print(df.head())
examples = []

# Group by puzzle date
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

    # Augment: shuffle 3 times
    for _ in range(4):  # 1 original + 3 shuffles = 4 total
        shuffled_words = all_words.copy()
        random.shuffle(shuffled_words)

        input_text = "Group the following words into 4 meaningful categories: " + ", ".join(shuffled_words)
        output_text = "; ".join(output_groups)

        examples.append({
            "input": input_text,
            "output": output_text
        })

# Save to file
with open("nyt_dataset.json", "w") as f:
    json.dump(examples, f, indent=2)

print(f"✅ Saved {len(examples)} augmented training examples to nyt_dataset.json")