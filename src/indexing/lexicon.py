import json
import os


INPUT_FILE = "/content/drive/MyDrive/DSA-Project/data/processed/preprocessing.json"
LEXICON_FILE = "/content/drive/MyDrive/DSA-Project/data/processed/lexicon.json"

# If the lexicon is available, load it; if not, return an empty and initial ID.
def load_existing_lexicon():
    if os.path.exists(LEXICON_FILE):
        with open(LEXICON_FILE, "r", encoding="utf-8") as f:
            lexicon = json.load(f)

        # Ensuring the consistency of ID
        lexicon = {str(k): int(v) for k, v in lexicon.items()}
        next_id = max(lexicon.values()) + 1
    else:
        lexicon = {}
        next_id = 1

    return lexicon, next_id

# Here we are loading the  preprocessed tokens from the file
def load_documents():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input preprocessing file not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# Here we are adding the  new tokens to the lexicon
def update_lexicon(lexicon, next_id, docs):
    for doc in docs:
        for token in doc.get("tokens", []):
            token = token.strip()
            if token and token not in lexicon:
                lexicon[token] = next_id
                next_id += 1

    return lexicon, next_id


def save_lexicon(lexicon):

    os.makedirs(os.path.dirname(LEXICON_FILE), exist_ok=True)
    with open(LEXICON_FILE, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, indent=2)


def main():
    lexicon, next_id = load_existing_lexicon()
    docs = load_documents()
    lexicon, next_id = update_lexicon(lexicon, next_id, docs)
    save_lexicon(lexicon)

    print(f"Lexicon updated successfully! Total unique tokens: {len(lexicon)}")


if __name__ == "__main__":
    main()
