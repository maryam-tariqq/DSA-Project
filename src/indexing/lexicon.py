import json
import os

INPUT_FILE = "../../data/processed/preprocessing.json"
OUTPUT_FILE = "../../data/processed/lexicon.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    docs = json.load(f)


lexicon = {}
current_id = 1

for doc in docs:
    for token in doc["tokens"]:
        lexicon[token] = current_id
        current_id += 1


os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(lexicon, f, indent=2)


print(f"Lexicon created successfully! Total unique tokens: {len(lexicon)}")
