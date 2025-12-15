import json
import os


LEXICON_FILE = "../../data/processed/lexicon_5.json"
INVERTED_INDEX_FILE = "../../data/processed/inverted_index_5.json"
BARRELS_FOLDER = "../../data/processed/barrels_5"

LETTERS = "abcdefghijklmnopqrstuvwxyz#"

# ------------------- LOAD CORE DATA -------------------
with open(LEXICON_FILE, "r", encoding="utf-8") as f:
    lexicon = json.load(f)              # word -> word_id

with open(INVERTED_INDEX_FILE, "r", encoding="utf-8") as f:
    inverted_index = json.load(f)       # word_id -> postings

# Reverse lexicon ONLY for routing
id_to_word = {str(v): k for k, v in lexicon.items()}
del lexicon

#  LOAD EXISTING BARRELS 
barrels = {}

for ch in LETTERS:
    path = os.path.join(BARRELS_FOLDER, f"{ch}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            barrels[ch] = json.load(f)
    else:
        #  Do NOT create new barrel files later
        barrels[ch] = {}

print("Existing barrels loaded")

#  MERGE 
added = 0
skipped = 0

for word_id, postings in inverted_index.items():
    word = id_to_word.get(word_id)

    if not word or not word[0].isalpha():
        barrel_key = "#"
    else:
        barrel_key = word[0].lower()

    barrel = barrels.get(barrel_key)
    if barrel is None:
        continue  

    # KEY CHECK: do NOT overwrite existing entries
    if word_id in barrel:
        skipped += 1
        continue

    barrel[word_id] = postings
    added += 1

# SAVE BACK (SAME FILES) 
for ch, data in barrels.items():
    path = os.path.join(BARRELS_FOLDER, f"{ch}.json")
    if os.path.exists(path):  #  never create new files
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(',', ':'))

print("=" * 60)
print("Incremental barrel update complete")
print(f"New entries added   : {added:,}")
print(f"Existing skipped    : {skipped:,}")
print("Barrel structure    : letter → word_id → postings")
print("Words stored        : NO")
print("IDs stored          : YES")
