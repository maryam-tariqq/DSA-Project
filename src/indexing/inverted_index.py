import json
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import os

# Load stopwords quietly so the script runs without interruptions
nltk.download('stopwords', quiet=True)

INPUT_FILE = "../../data/processed/preprocessing.json"
OUTPUT_FILE = "../../data/processed/inverted_index.json"
LEXICON_FILE = "../../data/processed/lexicon.json"

stop_words = set(stopwords.words('english'))


def load_preprocessed_data(file_path):
    # Reads the preprocessed JSON file and safely returns an empty list if anything goes wrong
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        print("Error: Could not load preprocessed data.")
        return []


def load_lexicon(file_path):
    # Loads the word-to-ID lexicon and ensures all keys are stored as strings
    if not os.path.exists(file_path):
        print("Error: Lexicon file not found.")
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lexicon = json.load(f)
            return {str(k): int(v) for k, v in lexicon.items()}
    except:
        print("Error: Could not load lexicon.")
        return {}


def load_existing_index(file_path):
    # Loads an already saved inverted index and collects all document IDs already processed
    if not os.path.exists(file_path):
        return {}, set()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            inverted_index = json.load(f)

        processed_ids = set()
        for token_id, doc_dict in inverted_index.items():
            processed_ids.update(doc_dict.keys())

        return inverted_index, processed_ids
    except:
        return {}, set()


def build_inverted_index(docs, existing_index, processed_ids, token_to_id):
    # Builds or updates the inverted index mapping each token ID to its documents and positions
    inverted_index = defaultdict(lambda: defaultdict(list), existing_index)
    new_docs_count = 0

    for doc in docs:
        doc_id = str(doc.get("id"))

        # Skip documents already indexed earlier
        if doc_id in processed_ids:
            continue

        new_docs_count += 1
        token_words = doc.get("tokens", [])

        # Add each token’s position into the index
        for position, token_word in enumerate(token_words):
            token_id = token_to_id.get(token_word)
            if token_id is not None:
                token_id_str = str(token_id)
                inverted_index[token_id_str][doc_id].append(position)

    # Convert nested structures into plain dicts for JSON saving
    result = {}
    for token_id, doc_dict in inverted_index.items():
        result[token_id] = dict(doc_dict)

    return result, new_docs_count


def save_inverted_index(inverted_index, output_file):
    # Saves the inverted index into a JSON file and creates folders automatically
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(inverted_index, f, indent=2, ensure_ascii=False)
        print("Inverted index saved successfully.")
    except Exception as e:
        print("Error while saving inverted index:", e)


def main():
    # Start by loading the lexicon since token IDs are needed for indexing
    token_to_id = load_lexicon(LEXICON_FILE)
    print(f"Loaded lexicon with {len(token_to_id)} tokens.")

    if not token_to_id:
        print("Cannot proceed without lexicon.")
        return

    # Load previous index so we only process new documents
    existing_index, processed_ids = load_existing_index(OUTPUT_FILE)
    print(f"Existing index has {len(existing_index)} token IDs and {len(processed_ids)} processed documents.")

    # Load the preprocessed tokenized documents
    docs = load_preprocessed_data(INPUT_FILE)
    print(f"Loaded {len(docs)} documents from preprocessing file.")

    if not docs:
        print("No documents found. Run preprocessing first.")
        return

    # Build the updated index using only new documents
    inverted_index, new_docs_count = build_inverted_index(
        docs, existing_index, processed_ids, token_to_id
    )

    if new_docs_count == 0:
        print("No new documents to add to the index.")
        return

    print(f"Indexed {new_docs_count} new documents.")
    print(f"Final index has {len(inverted_index)} unique token IDs.")

    # Count how many token-document pairs exist in the index
    total_postings = sum(len(doc_dict) for doc_dict in inverted_index.values())
    print(f"Total postings (token-document pairs): {total_postings}")

    # Save the updated index
    save_inverted_index(inverted_index, OUTPUT_FILE)


if __name__ == "__main__":
    main()
