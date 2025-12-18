# search_engine_doc_only.py
# CLEANED: Unnecessary console output removed (no banners / verbose logs)
# Errors and critical warnings are preserved

import json
import os
import math
import time
import numpy as np
from collections import defaultdict
from nltk.stem import PorterStemmer

# ==================== CONFIG ====================

PROJECT_BASE = "/content/drive/MyDrive/DSA-Project"

LEXICON_FILE = f"{PROJECT_BASE}/data/processed/lexicon.json"
BARRELS_FOLDER = f"{PROJECT_BASE}/data/processed/barrels"
FORWARD_INDEX_FILE = f"{PROJECT_BASE}/data/processed/forward_index.json"
DOC_META_FILE = f"{PROJECT_BASE}/data/raw/arxiv_100k.json"
DOC_EMBEDDINGS_FILE = f"{PROJECT_BASE}/data/processed/doc_embeddings_100k.npz"

TOP_K = 30
MAX_PROXIMITY_DOCS = 50

FIELD_WEIGHTS = {
    "title": 3.0,
    "authors": 2.0,
    "categories": 1.5,
    "abstract": 1.0
}

FIELD_INDICES = {
    "total_freq": 0,
    "positions": 1,
    "title": 2,
    "authors": 3,
    "categories": 4,
    "abstract": 5
}

stemmer = PorterStemmer()

# ==================== LOAD CORE DATA ====================

with open(LEXICON_FILE, "r", encoding="utf-8") as f:
    lexicon = json.load(f)

with open(DOC_META_FILE, "r", encoding="utf-8") as f:
    raw_list = json.load(f)
    raw_docs = {d["id"]: d for d in raw_list}

with open(FORWARD_INDEX_FILE, "r", encoding="utf-8") as f:
    forward_index = json.load(f)

# ==================== LAZY DOC EMBEDDINGS ====================

class LazyDocEmbeddings:
    """Lazy loader for document embeddings"""
    def __init__(self, file_path):
        self.file_path = file_path
        self.loaded = False
        self.doc_vectors = None
        self.doc_ids_list = None
        self.doc_embedding_matrix = None
        self.doc_norms = None
        self.doc_id_to_idx = {}

    def load(self):
        if self.loaded:
            return self.doc_vectors is not None

        if not os.path.exists(self.file_path):
            self.loaded = True
            return False

        try:
            emb = np.load(self.file_path)
            self.doc_vectors = emb["matrix"]
            self.doc_ids_list = emb["doc_ids"].tolist()
            self.doc_norms = np.linalg.norm(self.doc_vectors, axis=1)
            self.doc_embedding_matrix = self.doc_vectors
            self.doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids_list)}
            self.loaded = True
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load embeddings: {e}")
            self.loaded = True
            return False

    def is_available(self):
        return os.path.exists(self.file_path)


doc_embeddings = LazyDocEmbeddings(DOC_EMBEDDINGS_FILE)

# ==================== TRIE ====================

class TrieNode:
    __slots__ = ['children', 'is_end', 'word', 'frequency']
    
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.word = None
        self.frequency = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, frequency=0):
        if len(word) < 3 or not word.isalpha():
            return
        
        node = self.root
        for c in word.lower():
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        
        node.is_end = True
        node.word = word
        node.frequency = frequency

    def autocomplete(self, prefix, limit=5):
        if not prefix or len(prefix) < 2:
            return []
        
        node = self.root
        for c in prefix.lower():
            if c not in node.children:
                return []
            node = node.children[c]
        
        results = []
        
        def collect(n):
            if len(results) >= limit * 2:
                return
            if n.is_end and n.word:
                results.append((n.word, n.frequency))
            for child in n.children.values():
                collect(child)
        
        collect(node)
        results.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in results[:limit]]

print("Building Trie...")
trie = Trie()

word_freq = defaultdict(int)
for doc in raw_docs.values():
    title = str(doc.get('title', '')).lower()
    abstract = str(doc.get('abstract', ''))[:200].lower()
    
    for word in (title + ' ' + abstract).split():
        clean = ''.join(c for c in word if c.isalpha())
        if len(clean) >= 3:
            word_freq[clean] += 1

count = 0
for word in lexicon:
    if len(word) >= 3 and word.isalpha() and word_freq.get(word, 0) > 0:
        trie.insert(word, word_freq[word])
        count += 1

print(f"✓ Trie built with {count:,} words")

# ==================== BARRELS ====================

barrel_cache = {}

def load_barrel(word):
    key = word[0] if word and word[0].isalpha() else '#'
    if key in barrel_cache:
        return barrel_cache[key]
    path = os.path.join(BARRELS_FOLDER, f"{key}.json")
    if not os.path.exists(path):
        barrel_cache[key] = {}
        return {}
    with open(path, "r", encoding="utf-8") as f:
        barrel_cache[key] = json.load(f)
    return barrel_cache[key]

# ==================== HELPERS ====================

def preprocess_query(q):
    return [stemmer.stem(w) for w in q.lower().split()]

def get_postings(word):
    barrel = load_barrel(word)
    if word in lexicon:
        return barrel.get(str(lexicon[word]), {})
    return barrel.get(word, {})

def score_doc(entry, idf):
    score = 0.0
    tf = entry[0]
    for field, weight in FIELD_WEIGHTS.items():
        idx = FIELD_INDICES.get(field)
        if idx is not None and idx < len(entry):
            score += entry[idx] * weight
    return score * (1 + math.log(1 + tf)) * idf

def normalize_scores(scores):
    if not scores:
        return {}
    mn, mx = min(scores.values()), max(scores.values())
    if mn == mx:
        return {k: 1.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}

# ==================== RANKING ====================

def calculate_term_proximity_bonus_fast(doc_id, query_words):
    doc_terms = forward_index.get(doc_id)
    if not doc_terms:
        return 0.0
    positions = [doc_terms[w][:10] for w in query_words if w in doc_terms]
    if len(positions) < 2:
        return 0.0
    min_dist = min(abs(p1 - p2) for a in positions for b in positions if a is not b for p1 in a for p2 in b)
    return math.exp(-min_dist / 10.0) if min_dist else 1.0

def calculate_field_coverage_fast(doc_id, query_words):
    doc = raw_docs.get(doc_id, {})
    title = str(doc.get('title', '')).lower()
    categories = ' '.join(doc.get('categories', [])).lower()
    hits = sum(w in title for w in query_words) * 3.0
    hits += sum(w in categories for w in query_words) * 1.5
    return hits / len(query_words) if query_words else 0.0

# ==================== KEYWORD SEARCH ====================

def multi_word_search(query, top_k=30):
    words = preprocess_query(query)
    if not words:
        return {}
    postings_map = {w: get_postings(w) for w in words if get_postings(w)}
    if not postings_map:
        return {}
    idf = {w: math.log(len(raw_docs) / (1 + len(p))) for w, p in postings_map.items()}
    scores = defaultdict(float)
    counts = defaultdict(int)
    for w, p in postings_map.items():
        for d, e in p.items():
            scores[d] += score_doc(e, idf[w])
            counts[d] += 1
    for d in scores:
        scores[d] *= math.pow(2, counts[d] / len(words) * 2)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return dict(ranked)

# ==================== SEMANTIC SEARCH ====================

def compute_query_embedding_from_docs(query, top_n_docs=30):
    words = preprocess_query(query)
    if not words or not doc_embeddings.load():
        return None
    ids = set()
    for w in words:
        ids.update(get_postings(w).keys())
    idxs = [doc_embeddings.doc_id_to_idx[d] for d in list(ids)[:top_n_docs] if d in doc_embeddings.doc_id_to_idx]
    if not idxs:
        return None
    return np.mean(doc_embeddings.doc_embedding_matrix[idxs], axis=0)

def semantic_search(query, top_k=30):
    qv = compute_query_embedding_from_docs(query)
    if qv is None:
        return multi_word_search(query, top_k)
    sims = np.dot(doc_embeddings.doc_embedding_matrix, qv) / (doc_embeddings.doc_norms * np.linalg.norm(qv) + 1e-10)
    idxs = np.argpartition(sims, -top_k)[-top_k:]
    idxs = idxs[np.argsort(sims[idxs])[::-1]]
    return {doc_embeddings.doc_ids_list[i]: float(sims[i]) for i in idxs}

# ==================== HYBRID SEARCH ====================

def hybrid_search(query, top_k=30, alpha=0.3):
    kw = multi_word_search(query, top_k * 2)
    se = semantic_search(query, top_k * 2)
    if not kw:
        return dict(list(se.items())[:top_k])
    if not se or not doc_embeddings.loaded:
        return kw
    nkw, nse = normalize_scores(kw), normalize_scores(se)
    combined = {d: (1-alpha)*nkw.get(d,0) + alpha*nse.get(d,0) for d in set(nkw)|set(nse)}
    return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k])

# ==================== SEARCH API ====================

def search(query, mode="keyword", top_k=30):
    if mode == "semantic":
        scores = semantic_search(query, top_k)
    elif mode == "hybrid":
        scores = hybrid_search(query, top_k)
    else:
        scores = multi_word_search(query, top_k)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "doc_id": d,
            "score": round(float(s), 5),
            "title": raw_docs.get(d, {}).get("title", "N/A"),
            "authors": raw_docs.get(d, {}).get("authors", []),
            "paper_url": raw_docs.get(d, {}).get("paper_url", f"https://arxiv.org/abs/{d}"),
            "abstract": raw_docs.get(d, {}).get("abstract", "")[:200] + "..."
        }
        for d, s in ranked
    ]
