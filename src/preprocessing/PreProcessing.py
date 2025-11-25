import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
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
    text = (doc.get("title") or "") + " " + (doc.get("abstract") or "")

    words = text.lower().split()

    words = [
        stemmer.stem(w.strip(".,!?()[]{}\"'"))
        for w in words
        if w.isalpha() and w not in stop_words
    ]

    processed_data.append({
        "id": doc["id"],
        "tokens": words
    })


os.makedirs("data/processed", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=2)


print("Preprocessing complete! Saved to", OUTPUT_FILE)

