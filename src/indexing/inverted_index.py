# inverted_index.py
import json
import os
from collections import defaultdict

INPUT_PRE = "../../data/processed/preprocessing.json"
LEXICON_FILE = "../../data/processed/lexicon.json"
OUTPUT_INDEX = "../../data/processed/inverted_index.json"

# Fast lookup map—much quicker than if-elif chains
 # Associates field names with their locations within our small array
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

# Verify whether we have previously created an index; if so, we will only add new documents (incremental indexing).
if os.path.exists(OUTPUT_INDEX):
    print("Loading existing inverted index...")
    with open(OUTPUT_INDEX) as f:
        inverted_index = json.load(f)
    
    # Determine which documents have already been processed.
    processed_ids = set()
    for token_data in inverted_index.values():
        for doc_id in token_data.keys():
            processed_ids.add(doc_id)
    print(f"Existing: {len(inverted_index)} tokens, {len(processed_ids)} docs")
else:
    #Starting from scratch with no preexisting index
    inverted_index = {}
    processed_ids = set()

new_docs_count = 0
processed = 0

# here we Processing each document
for doc in docs:
    doc_id = str(doc["id"])
    
    # Ignore documents that we have already indexed.
    if doc_id in processed_ids:
        continue
    
    new_docs_count += 1

    # Index structure inverted: token_id -> doc_id -> [stats]
     # We can quickly determine "which documents contain token X?" thanks to this.
    tokens = doc.get("tokens", [])

    for t in tokens:
        token = t["token"]

        # Skipping the  tokens that aren't in our lexicon
        if token not in lexicon:
            continue

        wid = str(lexicon[token])
        
       # If this token appears for the first time, create an entry for it.
        if wid not in inverted_index:
            inverted_index[wid] = {}
        
        # Under this token, create a stats array for this document: [total_freq, [positions], title_freq, authors_freq,...]
        if doc_id not in inverted_index[wid]:
            inverted_index[wid][doc_id] = [0, [], 0, 0, 0, 0, 0, 0, 0]
        
        entry = inverted_index[wid][doc_id]

        # Increase the overall count
        entry[0] += 1

        # Recall the location of this token's appearance
        entry[1].append(t["global_pos"])

       # Keep track of the field from which this token originated. # For ranking purposes, title matches are more valuable than abstract matches.
        field = t.get("field")
        if field in FIELD_MAP:
            entry[FIELD_MAP[field]] += 1

    processed += 1
    # To ensure it's not frozen, display progress every 10,000 documents.
    if processed % 10000 == 0:
        print(f"Processed {processed}/{len(docs)} documents...")

# Saving our work
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
