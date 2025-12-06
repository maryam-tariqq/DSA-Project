# barrels.py — CORRECTED & OPTIMIZED

import json
import os
from collections import defaultdict

# ------------------- CONFIG -------------------
LEXICON_FILE = "/content/drive/MyDrive/DSA-Project/data/processed/lexicon_5.json"
INVERTED_INDEX_FILE = "/content/drive/MyDrive/DSA-Project/data/processed/inverted_index_5.json"
BARRELS_FOLDER = "/content/drive/MyDrive/DSA-Project/data/processed/barrels_5/"

# ------------------- Load Data -------------------
print("Loading lexicon...")
with open(LEXICON_FILE, "r", encoding="utf-8") as f:
    lexicon = json.load(f)

print("Loading inverted index...")  # ← FIXED
with open(INVERTED_INDEX_FILE, "r", encoding="utf-8") as f:
    inverted_index = json.load(f)

os.makedirs(BARRELS_FOLDER, exist_ok=True)

# ------------------- Reverse Lexicon -------------------
print("Building reverse lexicon...")
id_to_word = {str(word_id): word for word, word_id in lexicon.items()}
del lexicon
print(f"Reverse lexicon ready: {len(id_to_word):,} terms")

# ------------------- Pre-load Existing Barrels -------------------
print("Pre-loading existing barrels...")
existing_barrels = {}
for letter in "abcdefghijklmnopqrstuvwxyz#":
    path = os.path.join(BARRELS_FOLDER, f"{letter}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing_barrels[letter] = json.load(f)
    else:
        existing_barrels[letter] = {}

# ------------------- Build Barrels (1-level: first char only) -------------------
print("Building barrels in memory...")
barrels = defaultdict(dict)  # Simple: w1 → {word: postings}

# Copy existing data
for w1, words in existing_barrels.items():
    barrels[w1] = words.copy()

# Merge inverted index data
processed = 0
for word_id, postings in inverted_index.items():
    word = id_to_word.get(word_id)

    if not word or not word[0].isalpha():
        w1 = '#'
    else:
        w1 = word[0].lower()

    # Store COMPLETE posting information (not just doc IDs!)
    barrels[w1][word] = postings  # ← FIXED: includes [freq, positions, field_counts]

    processed += 1
    if processed % 10000 == 0:
        print(f"   → Processed {processed:,} terms...")

# ------------------- Save Barrels -------------------
print("Saving barrels to disk...")
saved_count = 0
total_terms = 0

for w1, words in barrels.items():
    if words:
        path = os.path.join(BARRELS_FOLDER, f"{w1}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(words, f, separators=(',', ':'))

        saved_count += 1
        total_terms += len(words)
        print(f"   ✓ Saved {w1}.json ({len(words):,} terms)")

# ------------------- Final Report -------------------
print(f"   Barrel files saved      : {saved_count}")
print(f"   Unique terms indexed    : {total_terms:,}")
print(f"   Structure               : first_char → word → postings")
print(f"   Postings include        : freq, positions, field_counts")
print(f"   Ready for search        : YES")
print("="*60)
