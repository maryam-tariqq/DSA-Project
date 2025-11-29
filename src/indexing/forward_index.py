# forward_index.py 

import json, os
from collections import defaultdict

INPUT_PRE = "/content/drive/MyDrive/DSA-Project/data/processed/preprocessing.json"
LEXICON_FILE = "/content/drive/MyDrive/DSA-Project/data/processed/lexicon.json"
OUTPUT_INDEX = "/content/drive/MyDrive/DSA-Project/data/processed/forward_index.json"

with open(LEXICON_FILE) as f:
    lexicon = json.load(f)

with open(INPUT_PRE) as f:
    docs = json.load(f)

forward_index = {}

for doc in docs:
    doc_id = str(doc["id"])
    data = defaultdict(lambda: {
        "frequency": 0,
        "positions": [],
        "fields": defaultdict(int),
        "in_title": False,
        "in_author": False,
        "in_category": False
    })

    for t in doc["tokens"]:
        token = t["token"]
        if token not in lexicon: continue
        wid = str(lexicon[token])
        d = data[wid]
        d["frequency"] += 1
        d["positions"].append(t["global_pos"])
        d["fields"][t["field"]] += 1
        if t["field"] == "title": d["in_title"] = True
        if t["field"] == "author": d["in_author"] = True
        if t["field"] == "category": d["in_category"] = True

    forward_index[doc_id] = {k: dict(v) for k, v in data.items()}

os.makedirs(os.path.dirname(OUTPUT_INDEX), exist_ok=True)
with open(OUTPUT_INDEX, "w", encoding="utf-8") as f:
    json.dump(forward_index, f, indent=2)

print(f"Forward index built with 'frequency' and field boosts!")
print(f"Saved to {OUTPUT_INDEX}")
