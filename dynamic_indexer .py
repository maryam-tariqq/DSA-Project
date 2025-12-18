# dynamic_indexer.py — COMPLETE OPTIMIZED VERSION
import os
import json
import time
from datetime import datetime
from collections import defaultdict
from nltk.stem import PorterStemmer

class DynamicIndexer:

    def __init__(self, base_path=None):
        """Initialize indexer with optimized loading"""
        # -------------------- BASE PATH --------------------
        if base_path is None:
            if os.path.exists("/content/drive/MyDrive/DSA-Project"):
                base_path = "/content/drive/MyDrive/DSA-Project/data/processed/"
            else:
                base_path = "./data/processed/"

        self.base_path = base_path
        self.lexicon_file = os.path.join(base_path, "lexicon.json")
        self.forward_index_file = os.path.join(base_path, "forward_index.json")
        self.inverted_index_file = os.path.join(base_path, "inverted_index.json")
        self.barrels_folder = os.path.join(base_path, "barrels")

        raw_path = base_path.replace("/processed/", "/raw/")
        self.doc_meta_file = os.path.join(raw_path, "arxiv_100k.json")

        self.stemmer = PorterStemmer()
        # ONLY 4 FIELDS: title, authors, abstract (+ positions)
        self.field_map = {"title": 2, "authors": 3, "abstract": 4}
        
        # Batch processing optimization
        self.pending_commits = []
        self.batch_size = 10  # Commit every 10 docs for safety
        
        # Barrel buffer for batch operations
        self.barrel_buffer = {}
        
        self._load()

    # ================= OPTIMIZED LOAD =================
    def _load(self):
        """Load with error handling and performance tracking"""
        start = time.time()
        
        os.makedirs(self.barrels_folder, exist_ok=True)
        
        # Load core structures
        self.lexicon = self._load_json(self.lexicon_file, {})
        self.forward_index = self._load_json(self.forward_index_file, {})
        
        # Load inverted index and convert ALL values to sets
        inv_data = self._load_json(self.inverted_index_file, {})
        self.inverted_index = {}
        
        for wid, docs in inv_data.items():
            if isinstance(docs, list):
                self.inverted_index[wid] = set(docs)
            elif isinstance(docs, dict):
                # Handle dict case (shouldn't happen but just in case)
                self.inverted_index[wid] = set(docs.keys()) if docs else set()
            elif isinstance(docs, set):
                self.inverted_index[wid] = docs
            else:
                # Fallback: try to convert to set
                try:
                    self.inverted_index[wid] = set(docs)
                except:
                    print(f"⚠ Warning: Could not convert inverted index entry {wid} (type: {type(docs)})")
                    self.inverted_index[wid] = set()
        
        # Load metadata (can be slow, optimize)
        try:
            self.doc_meta = {d["id"]: d for d in self._load_json(self.doc_meta_file, [])}
        except Exception as e:
            print(f"⚠ Error loading metadata: {e}")
            self.doc_meta = {}
        
        self.next_term_id = max(map(int, self.lexicon.values()), default=0) + 1
        
        elapsed = time.time() - start
        print(f"✓ Loaded in {elapsed:.2f}s: {len(self.lexicon):,} terms, {len(self.forward_index):,} docs")

    def _load_json(self, path, default):
        """Load JSON with error handling"""
        if not os.path.exists(path):
            return default
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠ JSON error in {path}: {e}")
            return default
        except Exception as e:
            print(f"⚠ Error loading {path}: {e}")
            return default

    # ================= OPTIMIZED ADD =================
    def add(self, doc):
        """Add document with optimizations"""
        try:
            doc = self._normalize(doc)
            doc_id = doc["id"]

            # Skip if already indexed
            if doc_id in self.forward_index:
                return False

            # Tokenize
            tokens = self._tokenize(doc)
            if not tokens:
                return False

            # Update all structures
            self._update_lexicon(tokens)
            self._update_forward_index(doc_id, tokens)
            
            # CRITICAL: Check inverted index state before updating
            try:
                self._update_inverted_index(doc_id, tokens)
            except AttributeError as e:
                print(f"⚠ Inverted index error for {doc_id}: {e}")
                # Try to fix the inverted index on-the-fly
                self._fix_inverted_index()
                # Retry
                self._update_inverted_index(doc_id, tokens)
            
            self._buffer_barrel_updates(doc_id, tokens)

            # Update metadata
            self.doc_meta[doc_id] = doc
            
            # Add to pending commits
            self.pending_commits.append(doc_id)
            
            # Auto-commit in batches
            if len(self.pending_commits) >= self.batch_size:
                self._commit()
                self.pending_commits = []
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding {doc.get('id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ================= NORMALIZE & TOKENIZE =================
    def _normalize(self, doc):
        """Normalize document with validation - ONLY 4 FIELDS"""
        if not isinstance(doc, dict):
            raise ValueError("Document must be a dictionary")
        
        doc_id = str(doc.get("id", "")).strip()
        if not doc_id:
            raise ValueError("Document must have an ID")
        
        # ONLY 4 FIELDS: id, title, authors, abstract
        return {
            "id": doc_id,
            "title": str(doc.get("title", ""))[:500],
            "authors": str(doc.get("authors", ""))[:200],
            "abstract": str(doc.get("abstract", ""))[:1000]
        }

    def _tokenize(self, doc):
        """Optimized tokenization - ONLY 3 TEXT FIELDS"""
        tokens = []
        pos = 0
        stem = self.stemmer.stem
        
        # ONLY: title, authors, abstract
        for field in ("title", "authors", "abstract"):
            text = doc.get(field, "").lower()
            
            # Skip empty fields
            if not text:
                continue
            
            # Fast tokenization
            for word in text.split():
                # Clean word
                word = "".join(c for c in word if c.isalnum() or c == "-")
                
                # Skip short words
                if len(word) <= 1:
                    continue
                
                # Stem and add
                try:
                    stemmed = stem(word)
                    tokens.append((stemmed, pos, field))
                    pos += 1
                except Exception:
                    # Skip problematic words
                    continue
        
        return tokens

    # ================= LEXICON =================
    def _update_lexicon(self, tokens):
        """Fast lexicon update"""
        lex = self.lexicon
        nid = self.next_term_id
        
        for word, _, _ in tokens:
            if word not in lex:
                lex[word] = nid
                nid += 1
        
        self.next_term_id = nid

    # ================= FORWARD INDEX =================
    def _update_forward_index(self, doc_id, tokens):
        """Optimized forward index update"""
        data = {}
        lex = self.lexicon
        fmap = self.field_map
        
        for word, pos, field in tokens:
            wid = str(lex[word])
            
            if wid not in data:
                data[wid] = [0, [], 0, 0, 0, 0]
            
            entry = data[wid]
            entry[0] += 1  # total_freq
            entry[1].append(pos)  # positions
            entry[fmap[field]] += 1  # field count
        
        self.forward_index[doc_id] = data

    # ================= INVERTED INDEX =================
    def _update_inverted_index(self, doc_id, tokens):
        """Fast inverted index update - FIXED for dict/set conversion"""
        lex = self.lexicon
        inv = self.inverted_index
        
        # Use set to avoid duplicates
        words_in_doc = set(word for word, _, _ in tokens)
        
        for word in words_in_doc:
            wid = str(lex[word])
            
            if wid not in inv:
                inv[wid] = set()
            else:
                # CRITICAL FIX: Handle both list and dict from JSON
                if isinstance(inv[wid], list):
                    inv[wid] = set(inv[wid])
                elif isinstance(inv[wid], dict):
                    # Sometimes JSON can produce nested dicts
                    inv[wid] = set(inv[wid].keys()) if inv[wid] else set()
                elif not isinstance(inv[wid], set):
                    # Fallback: convert to set
                    try:
                        inv[wid] = set(inv[wid])
                    except:
                        inv[wid] = set()
            
            inv[wid].add(doc_id)

    # ================= OPTIMIZED BARRELS (BUFFERED) =================
    def _buffer_barrel_updates(self, doc_id, tokens):
        """Buffer barrel updates instead of writing immediately"""
        lex = self.lexicon
        fmap = self.field_map
        
        # Group tokens by letter
        for word, pos, field in tokens:
            letter = word[0] if word and word[0].isalpha() else "#"
            wid = str(lex[word])
            
            # Initialize buffer structure
            if letter not in self.barrel_buffer:
                self.barrel_buffer[letter] = {}
            
            if wid not in self.barrel_buffer[letter]:
                self.barrel_buffer[letter][wid] = {}
            
            if doc_id not in self.barrel_buffer[letter][wid]:
                self.barrel_buffer[letter][wid][doc_id] = [0, [], 0, 0, 0, 0]
            
            entry = self.barrel_buffer[letter][wid][doc_id]
            entry[0] += 1
            entry[1].append(pos)
            entry[fmap[field]] += 1

    def _flush_barrel_buffer(self):
        """Flush buffered barrel updates to disk"""
        for letter, words in self.barrel_buffer.items():
            self._update_barrel_file_batch(letter, words)
        
        # Clear buffer
        self.barrel_buffer = {}

    def _update_barrel_file_batch(self, letter, words_data):
        """Update barrel file with batched data"""
        path = os.path.join(self.barrels_folder, f"{letter}.json")
        
        # Load existing barrel
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    barrel = json.load(f)
            else:
                barrel = {}
        except Exception as e:
            print(f"⚠ Error loading barrel {letter}: {e}")
            barrel = {}
        
        # Merge buffered data
        for wid, doc_entries in words_data.items():
            if wid not in barrel:
                barrel[wid] = {}
            
            for doc_id, entry in doc_entries.items():
                barrel[wid][doc_id] = entry
        
        # Save barrel (compact format)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(barrel, f, separators=(",", ":"))
        except Exception as e:
            print(f"❌ Error saving barrel {letter}: {e}")

    # ================= OPTIMIZED SAVE =================
    def _commit(self):
        """
        Optimized commit:
        - Flush barrel buffer first
        - Save changed files
        - Compact JSON format
        """
        try:
            print(f"  📝 Committing changes...")
            
            # Flush barrel buffer
            print(f"    - Flushing barrel buffer ({len(self.barrel_buffer)} letters)...")
            self._flush_barrel_buffer()
            
            # Save lexicon (usually small)
            print(f"    - Saving lexicon ({len(self.lexicon)} terms)...")
            self._save_json(self.lexicon_file, self.lexicon)
            
            # Save forward index (can be large)
            print(f"    - Saving forward index ({len(self.forward_index)} docs)...")
            self._save_json(self.forward_index_file, self.forward_index)
            
            # Save inverted index (convert sets to lists)
            print(f"    - Saving inverted index ({len(self.inverted_index)} entries)...")
            inv_serializable = {k: list(v) for k, v in self.inverted_index.items()}
            self._save_json(self.inverted_index_file, inv_serializable)
            
            # Save metadata (compact format)
            print(f"    - Saving metadata ({len(self.doc_meta)} docs)...")
            self._save_json(
                self.doc_meta_file, 
                list(self.doc_meta.values()),
                indent=None  # No indentation for speed
            )
            
            print(f"  ✓ Commit complete!")
            
        except Exception as e:
            print(f"❌ Commit error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _save_json(self, path, data, indent=None):
        """Save JSON with error handling"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Write to temp file first (safer)
            temp_path = path + '.tmp'
            with open(temp_path, "w", encoding="utf-8") as f:
                if indent:
                    json.dump(data, f, indent=indent)
                else:
                    json.dump(data, f, separators=(",", ":"))
            
            # Rename to final path (atomic operation)
            os.replace(temp_path, path)
            
            print(f"      ✓ Saved: {os.path.basename(path)} ({os.path.getsize(path)} bytes)")
            
        except Exception as e:
            print(f"❌ Error saving {path}: {e}")
            import traceback
            traceback.print_exc()
            raise

    # ================= PUBLIC METHODS =================
    def batch_add(self, docs):
        """
        Add multiple documents efficiently
        Returns: (success_count, failed_count)
        """
        success = 0
        failed = 0
        
        start_time = time.time()
        
        for i, doc in enumerate(docs):
            if self.add(doc):
                success += 1
            else:
                failed += 1
            
            # Progress indicator for large batches
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Progress: {i+1}/{len(docs)} docs ({rate:.1f} docs/sec)")
        
        # CRITICAL: Always commit at end of batch
        print(f"  Committing {len(self.pending_commits)} pending + buffer...")
        self._commit()
        self.pending_commits = []
        
        total_time = time.time() - start_time
        print(f"✓ Batch complete: {success} added, {failed} failed in {total_time:.2f}s")
        
        return success, failed

    def _fix_inverted_index(self):
        """Fix inverted index by converting all values to sets"""
        print("  🔧 Fixing inverted index data types...")
        fixed = 0
        
        for wid, docs in self.inverted_index.items():
            if not isinstance(docs, set):
                if isinstance(docs, list):
                    self.inverted_index[wid] = set(docs)
                    fixed += 1
                elif isinstance(docs, dict):
                    self.inverted_index[wid] = set(docs.keys()) if docs else set()
                    fixed += 1
                else:
                    try:
                        self.inverted_index[wid] = set(docs)
                        fixed += 1
                    except:
                        self.inverted_index[wid] = set()
                        fixed += 1
        
        if fixed > 0:
            print(f"  ✓ Fixed {fixed} entries in inverted index")

    def remove(self, doc_id):
        """Remove a document from the index"""
        try:
            doc_id = str(doc_id)
            
            # Check if exists
            if doc_id not in self.forward_index:
                return False
            
            # Get terms in this document
            doc_terms = self.forward_index[doc_id]
            
            # Remove from inverted index
            for wid in doc_terms.keys():
                if wid in self.inverted_index:
                    self.inverted_index[wid].discard(doc_id)
                    if not self.inverted_index[wid]:
                        del self.inverted_index[wid]
            
            # Remove from forward index
            del self.forward_index[doc_id]
            
            # Remove from metadata
            if doc_id in self.doc_meta:
                del self.doc_meta[doc_id]
            
            # Note: Barrel cleanup is expensive, skip for now
            # Barrels will be cleaned up on next full rebuild
            
            return True
            
        except Exception as e:
            print(f"❌ Error removing {doc_id}: {e}")
            return False

    def get_document(self, doc_id):
        """Retrieve document metadata"""
        return self.doc_meta.get(str(doc_id))

    def search_term(self, term):
        """Search for documents containing a term"""
        try:
            # Stem the term
            stemmed = self.stemmer.stem(term.lower())
            
            # Get term ID
            if stemmed not in self.lexicon:
                return []
            
            wid = str(self.lexicon[stemmed])
            
            # Get documents
            if wid not in self.inverted_index:
                return []
            
            return list(self.inverted_index[wid])
            
        except Exception as e:
            print(f"❌ Search error for '{term}': {e}")
            return []

    def get_stats(self):
        """Get indexer statistics"""
        barrel_count = len([f for f in os.listdir(self.barrels_folder) 
                           if f.endswith('.json')]) if os.path.exists(self.barrels_folder) else 0
        
        return {
            "terms": len(self.lexicon),
            "documents": len(self.forward_index),
            "inverted_entries": len(self.inverted_index),
            "metadata_entries": len(self.doc_meta),
            "barrel_files": barrel_count,
            "pending_commits": len(self.pending_commits),
            "buffer_size": len(self.barrel_buffer)
        }

    def rebuild_barrels(self):
        """Rebuild all barrel files from forward index (maintenance operation)"""
        print("🔧 Rebuilding barrels from forward index...")
        start = time.time()
        
        # Clear existing barrels
        if os.path.exists(self.barrels_folder):
            for f in os.listdir(self.barrels_folder):
                if f.endswith('.json'):
                    os.remove(os.path.join(self.barrels_folder, f))
        
        # Rebuild from forward index
        barrels = defaultdict(lambda: defaultdict(dict))
        
        for doc_id, terms in self.forward_index.items():
            for wid, entry in terms.items():
                # Get term from lexicon
                term = None
                for word, tid in self.lexicon.items():
                    if str(tid) == wid:
                        term = word
                        break
                
                if term:
                    letter = term[0] if term and term[0].isalpha() else "#"
                    barrels[letter][wid][doc_id] = entry
        
        # Save barrels
        for letter, words in barrels.items():
            path = os.path.join(self.barrels_folder, f"{letter}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(words, f, separators=(",", ":"))
        
        elapsed = time.time() - start
        print(f"✓ Rebuilt {len(barrels)} barrel files in {elapsed:.2f}s")

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure commit on exit"""
        if self.pending_commits or self.barrel_buffer:
            self._commit()
        return False

# ================= USAGE EXAMPLE =================
if __name__ == "__main__":
    # Example usage
    indexer = DynamicIndexer()
    
    # Add single document
    doc = {
        "id": "test_001",
        "title": "Machine Learning Basics",
        "authors": "John Doe, Jane Smith",
        "abstract": "An introduction to machine learning algorithms and techniques."
    }
    
    if indexer.add(doc):
        print("✓ Document added successfully")
    
    # Batch add
    docs = [
        {
            "id": f"test_{i:03d}",
            "title": f"Paper {i}",
            "authors": "Author Name",
            "abstract": f"This is paper number {i} about research topic."
        }
        for i in range(2, 10)
    ]
    
    success, failed = indexer.batch_add(docs)
    print(f"Added {success} documents, {failed} failed")
    
    # Get statistics
    stats = indexer.get_stats()
    print(f"\nIndex Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Search for term
    results = indexer.search_term("machine")
    print(f"\nDocuments containing 'machine': {results}")