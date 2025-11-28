import json

INPUT_FILE = "../../data/raw/arxiv.json"
OUTPUT_FILE = "../../data/raw/arxiv_100k.json"


def make_sample(n=100000):
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
                    arxiv_id = doc.get("id")
                    output.append({
                        "id": arxiv_id,
                        "title": doc.get("title"),
                        "authors": doc.get("authors"),
                        "categories": doc.get("categories"),
                        "report_no": doc.get("report_no"),
                        "journal-ref": doc.get("journal-ref"),
                        "abstract": doc.get("abstract"),
                        "update_date": doc.get("update_date"),
                        "paper_url": f"https://arxiv.org/abs/{arxiv_id}"
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
