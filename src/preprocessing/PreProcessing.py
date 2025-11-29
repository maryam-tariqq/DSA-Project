import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import os

nltk.download('stopwords')

INPUT_FILE = "../../data/raw/arxiv_100k.json"
OUTPUT_FILE = "../../data/processed/preprocessing.json"

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    docs = json.load(f)

processed_data = []

for doc in docs:
    # TITLE & ABSTRACT (stemmed)
    title = doc.get("title") or ""
    abstract = doc.get("abstract") or ""
    combined_text = f"{title} {abstract}".lower()
    # Normalize punctuation
    combined_text = re.sub(r'[^a-z0-9\s]', ' ', combined_text)
    # Tokenize
    title_abstract_tokens = combined_text.split()
    # Stem & remove stopwords
    title_abstract_tokens = [
        stemmer.stem(w) for w in title_abstract_tokens
        if len(w) > 1 and w not in stop_words
    ]


    # AUTHORS (un-stemmed)
    authors = doc.get("authors")
    if isinstance(authors, list):
        authors_text = " ".join(authors)
    else:
        authors_text = authors or ""
    authors_text = authors_text.lower()
    # Keep letters & spaces only, remove punctuation
    authors_text = re.sub(r'[^a-z\s]', '', authors_text)
    authors_tokens = authors_text.split()

    # Merge multi-word surnames - store BOTH original AND merged tokens
    merged_authors_tokens = []
    skip_next = 0
    for i, token in enumerate(authors_tokens):
        if skip_next:
            skip_next -= 1
            continue
        
        # Check if it's a surname prefix
        if token in ['van', 'von', 'de', 'der', 'den', 'la', 'le']:
            merged_authors_tokens.append(token)  # Store original prefix
            if i+1 < len(authors_tokens):
                merged_token = token + authors_tokens[i+1]  # e.g., "vander"
                merged_authors_tokens.append(merged_token)  # Store merged
                merged_authors_tokens.append(authors_tokens[i+1])  # Store next part
                skip_next = 1
            # If prefix is at end, just append it (already done above)
        else:
            merged_authors_tokens.append(token)


    # CATEGORIES (un-stemmed)
    categories = doc.get("categories") or ""
    categories_text = categories.lower()
    
    # Keep letters & dots for combined token
    categories_text = re.sub(r'[^a-z0-9\.]', ' ', categories_text)
    categories_tokens = categories_text.split()
    
    # Store both combined and split versions
    split_categories_tokens = []
    for cat in categories_tokens:
        split_categories_tokens.append(cat)  # combined token (e.g., "cs.ai")
        parts = cat.split('.')
        split_categories_tokens.extend(parts)  # split tokens (e.g., "cs", "ai")


    # JOURNAL REF & REPORT NO (un-stemmed)
    journal = doc.get("journal-ref") or ""
    report_no = doc.get("report_no") or ""
    journal_report_tokens = []
    for t in [journal, report_no]:
        t = t.lower()
        t = re.sub(r'[^a-z0-9\s]', ' ', t)
        journal_report_tokens.extend(t.split())


    # COMBINE ALL TOKENS
    all_tokens = (
        title_abstract_tokens +
        merged_authors_tokens +
        split_categories_tokens +
        journal_report_tokens
    )
    
    # REMOVE DUPLICATES while preserving order
    # Use this for inverted index / boolean search
    all_tokens = list(dict.fromkeys(all_tokens))

    processed_data.append({
        "id": doc["id"],
        "tokens": all_tokens
    })


# SAVE OUTPUT
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=2)

print(f"Preprocessing complete! Processed {len(processed_data)} documents.")
print(f"Saved to {OUTPUT_FILE}")
