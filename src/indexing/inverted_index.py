# inverted_index.py 

import json
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import os

nltk.download('stopwords', quiet=True)

INPUT_FILE = "/content/drive/MyDrive/DSA-Project/data/processed/preprocessing_5.json"   # ← NEW
OUTPUT_FILE = "/content/drive/MyDrive/DSA-Project/data/processed/inverted_index_5.json"
LEXICON_FILE = "/content/drive/MyDrive/DSA-Project/data/processed/lexicon.json"

def load_lexicon(file_path):
    if not os.path.exists(file_path):
        print("Error: Lexicon file not found.")
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        lexicon = json.load(f)
        return {str(k): int(v) for k, v in lexicon.items()}

def load_existing_index(file_path):
    if not os.path.exists(file_path):
        return {}, set()
    with open(file_path, "r", encoding="utf-8") as f:
        inverted_index = json.load(f)
    processed_ids = set()
    for doc_dict in inverted_index.values():
        processed_ids.update(doc_dict.keys())
    return inverted_index, processed_ids

def load_preprocessed_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("Error loading preprocessing:", e)
        return []

def build_inverted_index(docs, existing_index, processed_ids, token_to_id):
    inverted_index = defaultdict(lambda: defaultdict(list), existing_index)
    new_docs = 0

    for doc in docs:
        doc_id = str(doc["id"])
        if doc_id in processed_ids:
            continue

        new_docs += 1
        tokens_with_meta = doc.get("tokens", [])

        for entry in tokens_with_meta:
            token = entry["token"]                    # ← THIS IS THE ONLY CHANGE
            global_pos = entry["global_pos"]          # ← Use global position
            token_id = token_to_id.get(token)
            if token_id is not None:
                inverted_index[str(token_id)][doc_id].append(global_pos)

    # Convert to plain dict
    result = {tid: dict(docs) for tid, docs in inverted_index.items()}
    return result, new_docs

def save_inverted_index(index, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"Inverted index saved: {len(index)} terms")

# ===================== MAIN =====================
def main():
    token_to_id = load_lexicon(LEXICON_FILE)
    print(f"Lexicon loaded: {len(token_to_id):,} terms")

    existing_index, processed = load_existing_index(OUTPUT_FILE)
    print(f"Existing index: {len(existing_index)} terms, {len(processed)} docs processed")

    docs = load_preprocessed_data(INPUT_FILE)
    print(f"Loaded {len(docs)} documents from preprocessing_final.json")

    if not docs:
        print("No documents. Run preprocessing_final.py first!")
        return

    new_index, new_count = build_inverted_index(docs, existing_index, processed, token_to_id)

    if new_count == 0:
        print("No new documents to index.")
        return

    save_inverted_index(new_index, OUTPUT_FILE)
    total_postings = sum(len(d) for d in new_index.values())
    print(f"Added {new_count} new documents")
    print(f"Final inverted index: {len(new_index)} terms, {total_postings} postings")

if __name__ == "__main__":
    main()
