# preprocessing.py

import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import os

nltk.download('stopwords', quiet=True)

INPUT_FILE = "/content/drive/MyDrive/DSA-Project/data/raw/arxiv_100k.json"
OUTPUT_FILE = "/content/drive/MyDrive/DSA-Project/data/processed/preprocessing.json"

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    docs = json.load(f)

processed_data = []

for doc in docs:
    arxiv_id = doc["id"]

    # List of (stemmed_token, field, global_position)
    enriched_tokens = []
    global_pos = 0

    # === 1. TITLE + ABSTRACT (stemmed) ===
    title = (doc.get("title") or "").lower()
    abstract = (doc.get("abstract") or "").lower()
    combined = f"{title} {abstract}"
    combined = re.sub(r'[^a-z0-9\s]', ' ', combined)
    words = combined.split()

    for word in words:
        if len(word) > 1 and word not in stop_words:
            stemmed = stemmer.stem(word)
            field = "title" if word in title.split() else "abstract"
            enriched_tokens.append({
                "token": stemmed,
                "field": field,
                "global_pos": global_pos
            })
            global_pos += 1

    # === 2. AUTHORS (unstemmed + merged surnames) ===
    authors = doc.get("authors")
    if isinstance(authors, list):
        authors_text = " ".join(authors)
    else:
        authors_text = authors or ""
    authors_text = authors_text.lower()
    authors_text = re.sub(r'[^a-z\s]', '', authors_text)
    author_parts = authors_text.split()

    i = 0
    while i < len(author_parts):
        token = author_parts[i]
        if len(token) <= 1:
            i += 1
            continue

        # Add single token
        enriched_tokens.append({
            "token": token,
            "field": "author",
            "global_pos": global_pos
        })
        global_pos += 1

        # Handle van/von/de/etc.
        if token in ['van', 'von', 'de', 'der', 'den', 'la', 'le', 'di'] and i + 1 < len(author_parts):
            merged = token + author_parts[i+1]
            enriched_tokens.append({
                "token": merged,
                "field": "author",
                "global_pos": global_pos
            })
            global_pos += 1
            i += 2
        else:
            i += 1

    # === 3. CATEGORIES (split + combined) ===
    categories = doc.get("categories", "").lower()
    cats = [c.strip() for c in categories.split() if c.strip()]
    for cat in cats:
        # combined: cs.ai
        enriched_tokens.append({"token": cat.replace(".", ""), "field": "category", "global_pos": global_pos})
        global_pos += 1
        # split: cs, ai
        for part in cat.split('.'):
            if len(part) > 1:
                enriched_tokens.append({"token": part, "field": "category", "global_pos": global_pos})
                global_pos += 1

    # === 4. JOURNAL-REF & REPORT_NO ===
    for field in [doc.get("journal-ref", ""), doc.get("report_no", "")]:
        if not field:
            continue
        text = field.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        for word in text.split():
            if len(word) > 1:
                enriched_tokens.append({"token": word, "field": "journal", "global_pos": global_pos})
                global_pos += 1

    processed_data.append({
        "id": arxiv_id,
        "tokens": enriched_tokens  # Rich field-aware tokens
    })

# Save
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=2)

print(f"Enhanced preprocessing complete! {len(processed_data)} documents saved to {OUTPUT_FILE}")
