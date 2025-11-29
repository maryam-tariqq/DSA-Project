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
    # Combine all searchable text fields
    text_parts = []
    
    # Title and abstract (main content)
    text_parts.append(doc.get("title") or "")
    text_parts.append(doc.get("abstract") or "")
    
    # Authors (people search for papers by author names)
    text_parts.append(doc.get("authors") or "")
    
    # Categories (e.g., "cs.AI", "math.CO" - useful for filtering)
    text_parts.append(doc.get("categories") or "")
    
    # Journal reference (e.g., "Nature Physics" - prestigious journals)
    text_parts.append(doc.get("journal-ref") or "")
    
    # Report number (sometimes used in citations)
    text_parts.append(doc.get("report_no") or "")
    
    # Combine all text
    text = " ".join(text_parts)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep letters, numbers, and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Stem and filter stopwords
    words = [
        stemmer.stem(w)
        for w in words
        if len(w) > 1 and w not in stop_words  # Keep words longer than 1 char
    ]

    processed_data.append({
        "id": doc["id"],
        "tokens": words
    })


os.makedirs("../../data/processed", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=2)


print("Preprocessing complete! Saved to", OUTPUT_FILE)
