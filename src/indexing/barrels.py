import json
import os


# File paths for lexicon, inverted index, and barrels
LEXICON_FILE = "../../data/processed/lexicon_5.json"
INVERTED_INDEX_FILE = "../../data/processed/inverted_index_5.json"
BARRELS_FOLDER = "../../data/processed/barrels_5"

# Characters used to separate barrels
LETTERS = "abcdefghijklmnopqrstuvwxyz#"

# Load the lexicon and inverted index files

with open(LEXICON_FILE, "r", encoding="utf-8") as f:
    lexicon = json.load(f)              # maps words to word ids

with open(INVERTED_INDEX_FILE, "r", encoding="utf-8") as f:
    inverted_index = json.load(f)       # maps word ids to postings

# Create a reverse mapping only to find the starting letter of each word
id_to_word = {str(v): k for k, v in lexicon.items()}

# Remove lexicon from memory since it is no longer needed
del lexicon

# Load all existing barrel files into memory

barrels = {}

for ch in LETTERS:
    path = os.path.join(BARRELS_FOLDER, f"{ch}.json")
    if os.path.exists(path):
        # Load the barrel if the file already exists
        with open(path, "r", encoding="utf-8") as f:
            barrels[ch] = json.load(f)
    else:
        # Keep an empty dictionary but do not create a new file later
        barrels[ch] = {}

print("Existing barrels loaded")

# Merge the inverted index into the appropriate barrels

added = 0
skipped = 0

for word_id, postings in inverted_index.items():
    # Get the actual word using the word id
    word = id_to_word.get(word_id)

    # Decide which barrel the word belongs to
    if not word or not word[0].isalpha():
        barrel_key = "#"
    else:
        barrel_key = word[0].lower()

    barrel = barrels.get(barrel_key)
    if barrel is None:
        # Skip if the barrel does not exist
        continue  

    # Skip if this word id already exists to avoid overwriting data
    if word_id in barrel:
        skipped += 1
        continue

    # Add the new word id and its postings to the barrel
    barrel[word_id] = postings
    added += 1

# Save the updated barrels back to their original files

for ch, data in barrels.items():
    path = os.path.join(BARRELS_FOLDER, f"{ch}.json")
    if os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(',', ':'))

print("=" * 60)
print("Incremental barrel update complete")
print(f"New entries added   : {added:,}")
print(f"Existing skipped    : {skipped:,}")
print("Barrel structure    : letter to word id to postings")
print("Words stored        : NO")
print("IDs stored          : YES")
