import json

INPUT_FILE = "../../data/raw/arxiv.json"
OUTPUT_FILE = "../../data/raw/arxiv_5.json"


def make_sample(n=5):
    output = []
    count = 0
    skipped = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if count >= n:
                break

            try:
                doc = json.loads(line)

                if "id" in doc:
                    output.append({
                        "id": doc.get("id"),
                        "title": doc.get("title"),
                        "abstract": doc.get("abstract")
                    })
                    count += 1
            except json.JSONDecodeError:
                skipped += 1
                continue

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(output, out, indent=2)

    print(f"Saved {len(output)} documents.")
    print(f"Skipped {skipped} invalid lines")


make_sample()
