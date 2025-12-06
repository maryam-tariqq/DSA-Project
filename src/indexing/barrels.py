# barrels.py — CORRECTED & OPTIMIZED
import json
import os
from collections import defaultdict

# CONFIG
# File locations for barrel files, the lexicon, and the inverted index.
LEXICON_FILE = "../../data/processed/lexicon.json"
INVERTED_INDEX_FILE = "../../data/processed/inverted_index.json"
BARRELS_FOLDER = "../../data/processed/barrels/"

#Load the necessary data
print("Loading lexicon...")
with open(LEXICON_FILE, "r", encoding="utf-8") as f:
    lexicon = json.load(f)

print("Loading inverted index...") 
with open(INVERTED_INDEX_FILE, "r", encoding="utf-8") as f:
    inverted_index = json.load(f)

# Verify the existence of the output folder
os.makedirs(BARRELS_FOLDER, exist_ok=True)

# Reverse Lexicon
# To access words using IDs, convert {word → id} to {id → word}.
print("Building reverse lexicon...")
id_to_word = {str(word_id): word for word, word_id in lexicon.items()}
del lexicon
print(f"Reverse lexicon ready: {len(id_to_word):,} terms")

# Load Existing Barrels (if any)
# To prevent new data from being overwritten, pre-load previously saved barrels.
print("Pre-loading existing barrels...")
existing_barrels = {}
for letter in "abcdefghijklmnopqrstuvwxyz#":
    path = os.path.join(BARRELS_FOLDER, f"{letter}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing_barrels[letter] = json.load(f)
    else:
        existing_barrels[letter] = {}

# Build Barrels in Memory
# Barrels are arranged according to each word's initial character.
# For instance, "apple" → barrel 'a', "network" → barrel 'n', and numbers/symbols → '#'
print("Building barrels in memory...")
barrels = defaultdict(dict)  # Simple: w1 → {word: postings}

# Start by loading the current content.
for w1, words in existing_barrels.items():
    barrels[w1] = words.copy()

# Merge new inverted index entries into the barrels
processed = 0
for word_id, postings in inverted_index.items():
    word = id_to_word.get(word_id)
# The '#' barrel is used for words that don't begin with A–Z.
    if not word or not word[0].isalpha():
        w1 = '#'
    else:
        w1 = word[0].lower()

    # # Save the complete posting list, including field counts, positions, and frequency.
    barrels[w1][word] = postings 

    processed += 1
    if processed % 10000 == 0:
        print(f"   → Processed {processed:,} terms...")

# Save Barrels 
# To enable quicker search access in the future, write each barrel to a different JSON file.
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

#  Final Report
print(f"   Barrel files saved      : {saved_count}")
print(f"   Unique terms indexed    : {total_terms:,}")
print(f"   Structure               : first_char → word → postings")
print(f"   Postings include        : freq, positions, field_counts")
print(f"   Ready for search        : YES")
print("="*60)
