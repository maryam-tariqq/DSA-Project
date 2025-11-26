import json
import os

INPUT_FILE = "../../data/processed/preprocessing.json"
LEXICON_FILE = "../../data/processed/lexicon.json"

# Step 1: Load existing lexicon if it exists
if os.path.exists(LEXICON_FILE):
    with open(LEXICON_FILE, "r", encoding="utf-8") as f:
        lexicon = json.load(f)
    # Ensure keys are string and values are integers
    lexicon = {str(k): int(v) for k, v in lexicon.items()}
    current_id = max(lexicon.values()) + 1
else:
    lexicon = {}
    current_id = 1

# Step 2: Adding new tokens only
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    docs = json.load(f)

for doc in docs:
    for token in doc["tokens"]:
        if token not in lexicon:
            lexicon[token] = current_id
            current_id += 1

# Step 3: Saves updated lexicon
os.makedirs(os.path.dirname(LEXICON_FILE), exist_ok=True)
with open(LEXICON_FILE, "w", encoding="utf-8") as f:
    json.dump(lexicon, f, indent=2)

print(f"Lexicon updated successfully! Total unique tokens: {len(lexicon)}")
