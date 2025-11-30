# inverted_index.py
import json
import os
from collections import defaultdict

INPUT_PRE = "../../data/processed/preprocessing.json"
LEXICON_FILE = "../../data/processed/lexicon.json"
OUTPUT_INDEX = "../../data/processed/inverted_index.json"

# Quick lookup map - way faster than if-elif chains
# Maps field names to their position in our compact array
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

# Check if we've already built an index before
# If yes, we'll only add new documents (incremental indexing)
if os.path.exists(OUTPUT_INDEX):
    print("Loading existing inverted index...")
    with open(OUTPUT_INDEX) as f:
        inverted_index = json.load(f)
    
    # Figure out which documents we've already processed
    processed_ids = set()
    for token_data in inverted_index.values():
        for doc_id in token_data.keys():
            processed_ids.add(doc_id)
    print(f"Existing: {len(inverted_index)} tokens, {len(processed_ids)} docs")
else:
    # Starting fresh - no existing index
    inverted_index = {}
    processed_ids = set()

new_docs_count = 0
processed = 0

# Process each document
for doc in docs:
    doc_id = str(doc["id"])
    
    # Skip documents we've already indexed
    if doc_id in processed_ids:
        continue
    
    new_docs_count += 1

    # Inverted index structure: token_id -> doc_id -> [stats]
    # This lets us quickly find "which documents contain token X?"
    tokens = doc.get("tokens", [])

    for t in tokens:
        token = t["token"]

        # Skip tokens that aren't in our lexicon
        if token not in lexicon:
            continue

        wid = str(lexicon[token])
        
        # Create an entry for this token if it's the first time we see it
        if wid not in inverted_index:
            inverted_index[wid] = {}
        
        # Create stats array for this doc under this token
        # [total_freq, [positions], title_freq, authors_freq, ...]
        if doc_id not in inverted_index[wid]:
            inverted_index[wid][doc_id] = [0, [], 0, 0, 0, 0, 0, 0, 0]
        
        entry = inverted_index[wid][doc_id]

        # Bump up the total count
        entry[0] += 1

        # Remember where this token appeared
        entry[1].append(t["global_pos"])

        # Track which field this token came from
        # Title matches are worth more than abstract matches for ranking
        field = t.get("field")
        if field in FIELD_MAP:
            entry[FIELD_MAP[field]] += 1

    processed += 1
    # Show progress every 10k docs so we know it's not frozen
    if processed % 10000 == 0:
        print(f"Processed {processed}/{len(docs)} documents...")

# Save our work
if new_docs_count == 0:
    print("No new documents to index.")
else:
    print(f"Indexed {new_docs_count} new documents")
    print("Writing index to disk...")
    os.makedirs(os.path.dirname(OUTPUT_INDEX), exist_ok=True)

    # Compact JSON format - no unnecessary spaces, smaller file
    with open(OUTPUT_INDEX, "w", encoding="utf-8") as f:
        json.dump(inverted_index, f, separators=(',', ':'))

    print(f"Inverted index built successfully!")
    print(f"Total unique tokens: {len(inverted_index):,}")
    print(f"Format: token_id -> doc_id -> [total_freq, [positions], title, authors, categories, report_no, journal, abstract, update_date]")
    print(f"Saved to {OUTPUT_INDEX}")

    total_postings = sum(len(doc_dict) for doc_dict in inverted_index.values())
    avg_docs = total_postings / len(inverted_index) if inverted_index else 0
    print(f"Total postings: {total_postings:,}")
    print(f"Average documents per token: {avg_docs:.1f}")
