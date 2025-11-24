#----FORWARD INDEX----
import json
import os
from collections import defaultdict

INPUT_FILE = "./data/processed/preprocessing.json"
LEXICON_FILE = "./data/processed/lexicon.json"
FORWARD_INDEX_FILE = "./data/processed/forward_index.json"

# ---------- Load Lexicon ----------
with open(LEXICON_FILE, "r", encoding="utf-8") as f:
    lexicon = {str(k): int(v) for k, v in json.load(f).items()}

# ---------- Load Existing Forward Index (if exists) ----------
if os.path.exists(FORWARD_INDEX_FILE):
    with open(FORWARD_INDEX_FILE, "r", encoding="utf-8") as f:
        forward_index = json.load(f)
else:
    forward_index = {}

# ---------- Load Preprocessed Documents ----------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    docs = json.load(f)

# ---------- Process New Documents Only ----------
added_count = 0
for doc in docs:
    doc_id = str(doc["id"])  # ensure consistent key type

    # Skip if already indexed
    if doc_id in forward_index:
        continue

    token_positions = defaultdict(list)

    for position, token in enumerate(doc["tokens"]):
        if token in lexicon:
            word_id = lexicon[token]
            token_positions[str(word_id)].append(position)

    # Add to forward index
    forward_index[doc_id] = token_positions
    added_count += 1

# ---------- Save Updated Forward Index ----------
os.makedirs(os.path.dirname(FORWARD_INDEX_FILE), exist_ok=True)
with open(FORWARD_INDEX_FILE, "w", encoding="utf-8") as f:
    json.dump(forward_index, f, indent=2)

print(f"Forward index updated successfully!")
print(f"New documents indexed: {added_count}")
print(f"Total documents in forward index: {len(forward_index)}")
