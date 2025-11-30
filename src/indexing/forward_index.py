# forward_index.py
import json
import os
from collections import defaultdict
from array import array

INPUT_PRE = "../../data/processed/preprocessing.json"
LEXICON_FILE = "../../data/processed/lexicon.json"
OUTPUT_INDEX = "../../data/processed/forward_index.json"

# Field name to index mapping (lookup in O(1))
FIELD_MAP = {
    "title": 2,
    "authors": 3,
    "categories": 4,
    "report_no": 5,
    "journal": 6,
    "abstract": 7,
    "update_date": 8
}

print("Loading lexicon...")
with open(LEXICON_FILE) as f:
    lexicon = json.load(f)

print("Loading documents...")
with open(INPUT_PRE) as f:
    docs = json.load(f)

print(f"Processing {len(docs)} documents...")

forward_index = {}
processed = 0

for doc in docs:
    doc_id = str(doc["id"])

    # Format: [total_freq, [positions], title_freq, authors_freq, ...]
    data = defaultdict(lambda: [0, [], 0, 0, 0, 0, 0, 0, 0])

    tokens = doc.get("tokens", [])

    for t in tokens:
        token = t["token"]

        # Here skip the tokens not present in the lexicon
        if token not in lexicon:
            continue

        wid = str(lexicon[token])
        entry = data[wid]

        # Updating the total frequency
        entry[0] += 1

        # Here we are storing theposition
        entry[1].append(t["global_pos"])

        # Field-specific frequency update (O(1) lookup)
        field = t.get("field")
        if field in FIELD_MAP:
            entry[FIELD_MAP[field]] += 1

   # Convert to a standard dictionary (no copying)
    forward_index[doc_id] = dict(data)

    processed += 1
    if processed % 10000 == 0:
        print(f"Processed {processed}/{len(docs)} documents...")

print("Writing index to disk...")
os.makedirs(os.path.dirname(OUTPUT_INDEX), exist_ok=True)

# To reduce file size, write with as little whitespace as possible.
with open(OUTPUT_INDEX, "w", encoding="utf-8") as f:
    json.dump(forward_index, f, separators=(',', ':'))

print(f"✓ Forward index built successfully!")
print(f"✓ Total documents: {len(forward_index):,}")
print(f"✓ Format: token_id -> [total_freq, [positions], title, authors, categories, report_no, journal, abstract, update_date]")
print(f"✓ Saved to {OUTPUT_INDEX}")

total_postings = sum(len(doc_data) for doc_data in forward_index.values())
avg_terms = total_postings / len(forward_index) if forward_index else 0
print(f"✓ Average unique terms per document: {avg_terms:.1f}")
