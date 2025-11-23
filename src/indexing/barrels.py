

#----BARRELS-----
import json
import os
import string

# ---------- Configuration ----------

LEXICON_FILE = "./data/processed/lexicon.json"
FORWARD_INDEX_FILE = "./data/processed/forward_index.json"
BARRELS_FOLDER = "./data/processed/barrels/"

# ---------- Load Lexicon ----------
with open(LEXICON_FILE, "r", encoding="utf-8") as f:
    lexicon = json.load(f)  # word -> word_id

# ---------- Load Forward Index ----------
with open(FORWARD_INDEX_FILE, "r", encoding="utf-8") as f:
    forward_index = json.load(f)  # doc_id -> {word_id: positions}

# ---------- Ensure Barrels Folder Exists ----------
os.makedirs(BARRELS_FOLDER, exist_ok=True)

# ---------- Helper functions ----------
def load_barrel(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_barrel(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------- Map word_id -> actual word ----------
id_to_word = {str(v): k for k, v in lexicon.items()}

# ---------- Track updated barrels ----------
updated_barrels = {}

# ---------- Process each document ----------
for doc_id, token_positions in forward_index.items():
    for word_id in token_positions.keys():
        word = id_to_word[word_id]

        # Skip empty words
        if not word:
            continue

        # Use first 3 letters for multi-level hashing
        # pad with '#' if word < 3 letters
        w1 = word[0].lower() if word[0].isalpha() else '#'
        w2 = word[1].lower() if len(word) > 1 and word[1].isalpha() else '#'
        w3 = word[2].lower() if len(word) > 2 and word[2].isalpha() else '#'

        barrel_file = os.path.join(BARRELS_FOLDER, f"{w1}.json")

        # Load barrel once per first letter
        if w1 not in updated_barrels:
            updated_barrels[w1] = load_barrel(barrel_file)

        # Initialize second & third levels if not exist
        if w2 not in updated_barrels[w1]:
            updated_barrels[w1][w2] = {}
        if w3 not in updated_barrels[w1][w2]:
            updated_barrels[w1][w2][w3] = {}

        # Add doc_id to word entry
        if word not in updated_barrels[w1][w2][w3]:
            updated_barrels[w1][w2][w3][word] = []
        if doc_id not in updated_barrels[w1][w2][w3][word]:
            updated_barrels[w1][w2][w3][word].append(doc_id)


# ---------- Save updated barrels ----------
total_words = 0
for w1, data in updated_barrels.items():
    barrel_file = os.path.join(BARRELS_FOLDER, f"{w1}.json")
    save_barrel(barrel_file, data)

    # Count words in this barrel
    barrel_word_count = sum(len(w3_dict) for w2_dict in data.values() for w3_dict in w2_dict.values())
    total_words += barrel_word_count


print(f"Total barrels updated/created: {len(updated_barrels)}")
print(f"Total words across all barrels: {total_words}")