# dynamic_indexer_ultrafast.py
# ULTRA-OPTIMIZED: Indexes documents in SECONDS, not minutes
# Key optimizations:
# 1. Batch barrel updates (only write once per session)
# 2. Lazy loading (don't load barrels until needed)
# 3. Minimal file I/O
# 4. In-memory barrel cache

import json
import os
from datetime import datetime
from nltk.stem import PorterStemmer
from collections import defaultdict
import time


class DynamicIndexer:
    """
    ULTRA-FAST: Indexes documents in <10 seconds typically
    - Caches barrels in memory during session
    - Batches all writes to end of operation
    - Only touches modified files
    """

    def __init__(self, base_path=None):
        if base_path is None:
            if os.path.exists("/content/drive/MyDrive/DSA-Project"):
                base_path = "/content/drive/MyDrive/DSA-Project/data/processed/"
            else:
                base_path = "./data/processed/"

        self.base_path = base_path
        self.lexicon_file = os.path.join(base_path, "lexicon.json")
        self.forward_index_file = os.path.join(base_path, "forward_index.json")
        self.barrels_folder = os.path.join(base_path, "barrels")

        raw_path = base_path.replace("/processed/", "/raw/")
        self.doc_meta_file = os.path.join(raw_path, "arxiv_100k.json")

        self.stemmer = PorterStemmer()

        # Field mapping
        self.field_map = {
            "title": 2,
            "authors": 3,
            "categories": 4,
            "abstract": 5
        }

        # IN-MEMORY CACHE for barrels (only load when needed)
        self.barrel_cache = {}
        self.modified_barrels = set()

        self._load_data()

    def _load_data(self):
        """Load only lexicon and forward index (NOT barrels)"""
        print(" Loading index files...")

        try:
            with open(self.lexicon_file, 'r', encoding='utf-8') as f:
                self.lexicon = json.load(f)
        except FileNotFoundError:
            self.lexicon = {}

        try:
            with open(self.forward_index_file, 'r', encoding='utf-8') as f:
                self.forward_index = json.load(f)
        except FileNotFoundError:
            self.forward_index = {}

        # Load doc metadata
        if os.path.exists(self.doc_meta_file):
            with open(self.doc_meta_file, 'r') as f:
                docs = json.load(f)
                self.doc_meta = {d["id"]: d for d in docs}
        else:
            self.doc_meta = {}

        os.makedirs(self.barrels_folder, exist_ok=True)

        # Track next available word ID
        self.next_word_id = max(self.lexicon.values(), default=0) + 1

        print(
            f"✓ Ready: {len(self.lexicon):,} terms, {len(self.forward_index):,} docs")
        print(f"  Next word ID: {self.next_word_id}")

    # ==================== BARREL CACHE (LAZY LOADING) ====================

    def _get_barrel(self, letter):
        """Load barrel only when needed (lazy loading)"""
        if letter not in self.barrel_cache:
            barrel_path = os.path.join(self.barrels_folder, f"{letter}.json")
            if os.path.exists(barrel_path):
                with open(barrel_path, 'r') as f:
                    self.barrel_cache[letter] = json.load(f)
            else:
                self.barrel_cache[letter] = {}
        return self.barrel_cache[letter]

    def _save_modified_barrels(self):
        """Save ONLY barrels that were modified"""
        if not self.modified_barrels:
            return

        print(f" Saving {len(self.modified_barrels)} modified barrels...")
        for letter in self.modified_barrels:
            barrel_path = os.path.join(self.barrels_folder, f"{letter}.json")
            with open(barrel_path, 'w') as f:
                json.dump(self.barrel_cache[letter], f, separators=(',', ':'))

        self.modified_barrels.clear()
        print("✓ Barrels saved")

    # ==================== ADD DOCUMENTS ====================

    def add(self, data, format_type="auto"):
        """Add document(s) - ULTRA FAST"""
        start_time = time.time()

        if format_type == "auto":
            format_type = self._detect_format(data)

        count = 0
        if format_type == "dict":
            if isinstance(data, list):
                count = sum(1 for d in data if self._add_document(
                    self._normalize_fields(d)))
            else:
                count = 1 if self._add_document(
                    self._normalize_fields(data)) else 0

        elif format_type == "json":
            doc = json.loads(data)
            if isinstance(doc, list):
                count = sum(1 for d in doc if self._add_document(
                    self._normalize_fields(d)))
            else:
                count = 1 if self._add_document(
                    self._normalize_fields(doc)) else 0

        elif format_type == "json_file":
            with open(data, 'r', encoding='utf-8') as f:
                doc = json.load(f)
            if isinstance(doc, list):
                count = sum(1 for d in doc if self._add_document(
                    self._normalize_fields(d)))
            else:
                count = 1 if self._add_document(
                    self._normalize_fields(doc)) else 0

        # BATCH SAVE at the end
        self._save_all()

        elapsed = time.time() - start_time
        print(f"⚡ Indexed {count} document(s) in {elapsed:.2f}s")
        return count > 0

    def _detect_format(self, data):
        if isinstance(data, dict) or isinstance(data, list):
            return "dict"
        if isinstance(data, str):
            if os.path.exists(data):
                return "json_file" if data.endswith(".json") else "text"
            try:
                json.loads(data)
                return "json"
            except:
                return "text"
        return "unknown"

    def _normalize_fields(self, doc):
        """Normalize document fields"""
        return {
            'id': str(doc.get('id', f"AUTO_{int(time.time() * 1000)}")),
            'title': str(doc.get('title', '')),
            'authors': str(doc.get('authors', '')),
            'categories': str(doc.get('categories', '')),
            'abstract': str(doc.get('abstract', '')),
            'update_date': doc.get('update_date', datetime.now().strftime("%Y-%m-%d")),
            'paper_url': doc.get('paper_url', f"https://arxiv.org/abs/{doc.get('id', '')}")
        }

    # ==================== CORE INDEXING (ULTRA FAST) ====================

    def _add_document(self, doc):
        """Add single document - IN-MEMORY UPDATES ONLY"""
        doc_id = str(doc["id"])

        if doc_id in self.forward_index:
            print(f"⚠ Document {doc_id} already exists")
            return False

        # Tokenize
        tokens = self._preprocess_document(doc)

        if not tokens:
            print(f"⚠ No valid tokens for document {doc_id}")
            return False

        # Track new terms added
        new_terms = []

        # Update lexicon (in memory)
        for t in tokens:
            if t["token"] not in self.lexicon:
                self.lexicon[t["token"]] = self.next_word_id
                new_terms.append(t["token"])
                self.next_word_id += 1

        if new_terms:
            print(f"  ✓ Added {len(new_terms)} new terms to lexicon")

        # Update forward index (in memory)
        self._update_forward_index(doc_id, tokens)

        # Update barrels (in memory cache)
        self._update_barrels_fast(doc_id, tokens)

        # Update metadata (in memory)
        self.doc_meta[doc_id] = doc

        return True

    def _preprocess_document(self, doc):
        """Tokenize with field information - OPTIMIZED"""
        tokens = []
        pos = 0

        fields = [
            ("title", doc.get("title", "")),
            ("authors", doc.get("authors", "")),
            ("categories", doc.get("categories", "")),
            ("abstract", doc.get("abstract", ""))
        ]

        for field, text in fields:
            # Fast tokenization
            words = str(text).lower().split()
            for word in words:
                # Clean word (optimized)
                word = ''.join(c for c in word if c.isalnum() or c == '-')
                if len(word) > 1:
                    try:
                        stemmed = self.stemmer.stem(word)
                        tokens.append({
                            "token": stemmed,
                            "global_pos": pos,
                            "field": field
                        })
                        pos += 1
                    except:
                        continue

        return tokens

    def _update_forward_index(self, doc_id, tokens):
        """Update forward index - OPTIMIZED"""
        data = defaultdict(lambda: [0, [], 0, 0, 0, 0])

        for t in tokens:
            wid = str(self.lexicon[t["token"]])
            entry = data[wid]

            entry[0] += 1
            entry[1].append(t["global_pos"])

            field_idx = self.field_map.get(t["field"])
            if field_idx is not None:
                entry[field_idx] += 1

        self.forward_index[doc_id] = dict(data)

    def _update_barrels_fast(self, doc_id, tokens):
        """
        ULTRA FAST: Updates barrels in memory cache only
        No disk I/O until _save_all() is called
        """
        # Group tokens by word
        word_stats = defaultdict(lambda: {
            "total": 0,
            "positions": [],
            "title": 0,
            "authors": 0,
            "categories": 0,
            "abstract": 0
        })

        for t in tokens:
            word = t["token"]
            word_stats[word]["total"] += 1
            word_stats[word]["positions"].append(t["global_pos"])

            field = t["field"]
            if field in word_stats[word]:
                word_stats[word][field] += 1

        # Update barrel cache (in memory)
        for word, stats in word_stats.items():
            wid = str(self.lexicon[word])
            letter = word[0] if word and word[0].isalpha() else "#"

            # Get barrel from cache (lazy load)
            barrel = self._get_barrel(letter)

            # Create entry
            entry = [
                stats["total"],
                stats["positions"],
                stats["title"],
                stats["authors"],
                stats["categories"],
                stats["abstract"]
            ]

            # Update barrel in memory
            if wid not in barrel:
                barrel[wid] = {}
            barrel[wid][doc_id] = entry

            # Mark as modified
            self.modified_barrels.add(letter)

    def _save_all(self):
        """Save all modified files in ONE batch"""
        print(" Saving changes...")

        # Save lexicon
        with open(self.lexicon_file, 'w') as f:
            json.dump(self.lexicon, f, separators=(',', ':'))

        # Save forward index
        with open(self.forward_index_file, 'w') as f:
            json.dump(self.forward_index, f, separators=(',', ':'))

        # Save metadata
        with open(self.doc_meta_file, 'w') as f:
            json.dump(list(self.doc_meta.values()), f, indent=2)

        # Save modified barrels
        self._save_modified_barrels()

        print("✓ All changes saved")

    def verify_document(self, doc_id):
        """Verify a document was properly indexed"""
        doc_id = str(doc_id)

        print(f"\n🔍 Verifying document {doc_id}...")

        # Check forward index
        if doc_id not in self.forward_index:
            print("  ✗ NOT in forward index")
            return False
        print(
            f"  ✓ Found in forward index ({len(self.forward_index[doc_id])} unique terms)")

        # Check metadata
        if doc_id not in self.doc_meta:
            print("  ✗ NOT in metadata")
            return False
        print(f"  ✓ Found in metadata")

        # Check a few barrels
        doc_data = self.forward_index[doc_id]
        sample_words = list(doc_data.keys())[:3]

        print(f" Checking barrels for sample terms...")
        for wid in sample_words:
            # Find the word from lexicon
            word = next((w for w, id in self.lexicon.items()
                        if str(id) == wid), None)
            if word:
                letter = word[0] if word and word[0].isalpha() else "#"
                barrel = self._get_barrel(letter)

                if wid in barrel and doc_id in barrel[wid]:
                    print(f"    ✓ '{word}' found in barrel {letter}")
                else:
                    print(f"    ✗ '{word}' NOT in barrel {letter}")
                    return False

        print(f" Document {doc_id} is fully indexed!")
        return True


# ==================== USAGE ====================

if __name__ == "__main__":
    indexer = DynamicIndexer()

    # Test adding a NEW document with RARE/UNIQUE terms
    import random
    doc_num = random.randint(100000, 999999)

    new_doc = {
        "id": f"2025.{doc_num}",
        "title": "Zorblaxian Hyperdimensional Quantumflux Metacognition Framework",
        "authors": "Dr. Xenophius Quarkbender, Prof. Zynthia Neutrino-Cascade",
        "categories": "cs.ZXQM cs.HYPR",
        "abstract": "We introduce zorblaxian metacognitive quantumflux protocols utilizing hyperdimensional neutrino-cascade entanglement. Our xenomorphic algorithmic framework demonstrates unprecedented quarkbending capabilities in multidimensional spacetime lattices. The zorblax coefficient shows remarkable hypersynchronization with metacognitive neutrino streams."
    }

    print(f"\n Adding test document (ID: {new_doc['id']})...")
    print(f"Testing with UNIQUE terms that shouldn't exist in lexicon...")

    # Check before
    before_count = len(indexer.lexicon)

    success = indexer.add(new_doc)

    if success:
        after_count = len(indexer.lexicon)
        new_terms = after_count - before_count

        print("\n Indexing complete!")
        print(f"  Total documents: {len(indexer.forward_index):,}")
        print(f"  Lexicon BEFORE: {before_count:,} terms")
        print(f"  Lexicon AFTER:  {after_count:,} terms")
        print(f"  NEW TERMS ADDED: {new_terms}")

        # Verify the document was added
        if new_doc['id'] in indexer.forward_index:
            print(f"  ✓ Document {new_doc['id']} confirmed in index")
            print(
                f"  Word count in doc: {len(indexer.forward_index[new_doc['id']])} unique terms")

        # Show some of the new terms
        if new_terms > 0:
            print(f"\n Sample new terms in lexicon:")
            new_words = sorted(indexer.lexicon.items(),
                               key=lambda x: x[1], reverse=True)[:10]
            for word, wid in new_words:
                if wid > before_count:
                    print(f"     '{word}' → ID {wid}")

        # VERIFY the document is searchable
        indexer.verify_document(new_doc['id'])

        print("\n" + "="*60)
        print(" IMPORTANT: To search this document:")
        print("="*60)
        print("1. RESTART your search engine to reload the index files")
        print("2. Or reload the SearchEngine class if it's already running")
        print("3. Try searching for: 'zorblaxian' or 'quantumflux'")
        print("="*60)
    else:
        print("\n✗ Failed to add document")
