# app.py - COLAB VERSION WITH GOOGLE DRIVE PATHS
from flask import Flask, render_template, request, jsonify
import sys
import os
import json
import gc
import time

# ==================== COLAB PATHS ====================
# Google Drive base path
BASE_PROJECT_PATH = "/content/drive/MyDrive/DSA-Project"

# Add to Python path
sys.path.insert(0, BASE_PROJECT_PATH)
sys.path.insert(0, os.path.join(BASE_PROJECT_PATH, "Indexing"))

# Import search engine and indexer
try:
    import search_engine
except:
    print("⚠ search_engine not found, creating dummy")
    class DummySearchEngine:
        raw_docs = {}
    search_engine = DummySearchEngine()

app = Flask(__name__)

# ==================== LAZY INDEXER ====================
_indexer_instance = None

def get_indexer():
    """Lazy-load indexer when needed"""
    global _indexer_instance
    if _indexer_instance is None:
        print("\n⚡ Loading Dynamic Indexer...")
        from dynamic_indexer import DynamicIndexer
        
        # Use Google Drive path
        base_path = os.path.join(BASE_PROJECT_PATH, "data", "processed")
        
        print(f"  📂 Base path: {base_path}")
        _indexer_instance = DynamicIndexer(base_path)
        print(f"  📊 Loaded: {len(_indexer_instance.lexicon)} terms, {len(_indexer_instance.forward_index)} docs")
        print("✓ Indexer ready\n")
    
    return _indexer_instance

def unload_indexer():
    """Unload indexer to free RAM"""
    global _indexer_instance
    if _indexer_instance:
        del _indexer_instance
        _indexer_instance = None
        gc.collect()
        print("\n✓ Indexer unloaded from memory\n")

# ==================== HELPERS ====================
def validate_doc(doc):
    """Validate and clean document - ONLY 4 FIELDS"""
    if not isinstance(doc, dict):
        return None
    
    doc_id = str(doc.get("id", "")).strip()
    if not doc_id:
        content = doc.get("title", "") or doc.get("abstract", "")
        if content:
            doc_id = f"DOC_{abs(hash(content[:100])) % 1000000}"
        else:
            return None
    
    title = str(doc.get("title", "")).strip()
    abstract = str(doc.get("abstract", "")).strip()
    
    if not title and not abstract:
        return None
    
    return {
        "id": doc_id,
        "title": title[:500] if title else "Untitled",
        "authors": str(doc.get("authors", ""))[:200],
        "abstract": abstract[:1000]
    }

def parse_json_safe(content):
    """Safely parse JSON with error handling"""
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            result = []
            for item in data:
                if isinstance(item, dict):
                    result.append(item)
                elif isinstance(item, list):
                    result.extend([x for x in item if isinstance(x, dict)])
            return result
        return []
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return []

def parse_csv_safe(content):
    """Parse CSV with robust error handling"""
    try:
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        if len(lines) < 2:
            return []
        
        header = lines[0]
        delimiter = ','
        for delim in [',', '\t', ';']:
            headers = [h.strip().strip('"').strip("'") for h in header.split(delim)]
            if len(headers) > 1:
                delimiter = delim
                break
        
        if len(headers) <= 1:
            return []
        
        docs = []
        for line in lines[1:]:
            try:
                values = [v.strip().strip('"').strip("'") for v in line.split(delimiter)]
                if len(values) == len(headers):
                    doc = dict(zip(headers, values))
                    docs.append(doc)
            except Exception:
                continue
        
        return docs
    except Exception as e:
        print(f"CSV parse error: {e}")
        return []

# ==================== ROUTES ====================

@app.route('/')
def home():
    """Home page"""
    try:
        return render_template('index.html')
    except:
        return """
        <html>
        <head><title>ArXiv Search Engine</title></head>
        <body style="font-family: Arial; margin: 50px;">
            <h1>🔍 ArXiv Search Engine - Colab Edition</h1>
            <hr>
            <h2>📤 Upload Documents</h2>
            <form action="/api/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".json,.csv,.txt" style="margin: 10px 0;">
                <button type="submit" style="padding: 10px 20px;">Upload</button>
            </form>
            <hr>
            <h2>🔧 Debug</h2>
            <a href="/api/debug/index_status">Index Status</a> | 
            <a href="/api/debug/paths">Check Paths</a>
        </body>
        </html>
        """

@app.route('/add')
def add_document_page():
    """Add document page"""
    try:
        return render_template('add.html')
    except:
        return home()

@app.route('/search')
def search():
    """Search page with results"""
    query = request.args.get('q', '').strip()
    mode = request.args.get('mode', 'keyword')
    
    if not query:
        return home()
    
    try:
        results, metrics = search_engine.search(
            query,
            mode=mode,
            top_k=30,
            return_metrics=True
        )
        
        try:
            return render_template(
                'results.html',
                query=query,
                results=results,
                mode=mode,
                result_count=len(results),
                metrics=metrics
            )
        except:
            # Fallback HTML
            results_html = "<br>".join([
                f"<div style='margin: 20px; padding: 10px; border: 1px solid #ccc;'>"
                f"<h3>{r.get('title', 'No title')}</h3>"
                f"<p><b>Authors:</b> {r.get('authors', 'N/A')}</p>"
                f"<p>{r.get('abstract', 'No abstract')[:200]}...</p>"
                f"</div>"
                for r in results[:10]
            ])
            return f"<html><body><h1>Results for: {query}</h1>{results_html}</body></html>"
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return f"<html><body><h1>Search Error</h1><p>{str(e)}</p></body></html>"

# ==================== API ENDPOINTS ====================

@app.route('/api/search')
def api_search():
    """API search endpoint"""
    query = request.args.get('q', '').strip()
    mode = request.args.get('mode', 'keyword')
    
    if not query:
        return jsonify({'query': '', 'results': [], 'count': 0, 'error': 'No query provided'})
    
    try:
        results, metrics = search_engine.search(query, mode=mode, top_k=30, return_metrics=True)
        return jsonify({
            'query': query,
            'mode': mode,
            'results': results,
            'count': len(results),
            'time_ms': metrics.get('time_ms', 0),
            'memory_mb': metrics.get('memory_used_mb', 0)
        })
    except Exception as e:
        return jsonify({'query': query, 'results': [], 'count': 0, 'error': str(e)}), 500

@app.route('/api/autocomplete')
def api_autocomplete():
    """Autocomplete suggestions"""
    query = request.args.get('q', '').strip()
    limit = max(3, min(int(request.args.get('limit', 5)), 10))
    
    if not query or len(query) < 2:
        return jsonify([])
    
    try:
        return jsonify(search_engine.autocomplete(query, limit=limit))
    except Exception as e:
        print(f"❌ Autocomplete error: {e}")
        return jsonify([])

# ------------------ ADD SINGLE DOCUMENT ------------------
@app.route('/api/add_document', methods=['POST'])
def add_document():
    """Add a single document"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        doc = validate_doc(data)
        if not doc:
            return jsonify({'error': 'Invalid document format'}), 400
        
        indexer = get_indexer()
        
        if doc['id'] in indexer.forward_index:
            return jsonify({'error': f"Document '{doc['id']}' already exists"}), 400
        
        print(f"\n📄 Adding document: {doc['id']}")
        
        if indexer.add(doc):
            print("  Committing...")
            indexer._commit()
            indexer.pending_commits = []
            
            search_engine.raw_docs[doc['id']] = doc
            
            elapsed = time.time() - start_time
            print(f"✓ Added in {elapsed:.2f}s\n")
            
            if elapsed > 10:
                unload_indexer()
            
            return jsonify({
                'success': True,
                'doc_id': doc['id'],
                'message': f"Document '{doc['id']}' indexed successfully",
                'time_s': round(elapsed, 2)
            })
        else:
            return jsonify({'error': 'Failed to index document'}), 500
            
    except Exception as e:
        print(f"❌ Add document error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ------------------ UPLOAD MULTIPLE DOCUMENTS ------------------
@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and index multiple documents from file"""
    start_time = time.time()
    
    try:
        print("\n" + "="*70)
        print("🚀 UPLOAD STARTED")
        print("="*70)
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        print(f"📄 File: {file.filename} (extension: {file_ext})")
        
        if file_ext not in {'.json', '.csv', '.txt'}:
            return jsonify({'error': f'Invalid file type. Allowed: .json, .csv, .txt'}), 400
        
        # Read file
        try:
            content = file.read().decode('utf-8')
            print(f"✓ Read {len(content)} bytes")
        except UnicodeDecodeError:
            file.seek(0)
            content = file.read().decode('latin-1')
            print(f"✓ Read {len(content)} bytes (latin-1)")
        
        # Parse based on extension
        docs_to_add = []
        
        if file_ext == '.json':
            print("  Parsing as JSON...")
            docs_to_add = parse_json_safe(content)
        elif file_ext == '.csv':
            print("  Parsing as CSV...")
            docs_to_add = parse_csv_safe(content)
        elif file_ext == '.txt':
            print("  Trying JSON first...")
            docs_to_add = parse_json_safe(content)
            if not docs_to_add:
                print("  Not JSON, treating as plain text...")
                docs_to_add = [{
                    'id': f"TXT_{abs(hash(content[:100])) % 1000000}",
                    'title': file.filename.replace('.txt', ''),
                    'authors': 'Unknown',
                    'abstract': content[:1000]
                }]
        
        print(f"✓ Parsed {len(docs_to_add)} documents")
        
        if not docs_to_add:
            return jsonify({'error': 'No valid documents found'}), 400
        
        if len(docs_to_add) > 100:
            return jsonify({'error': f'Too many documents. Max 100, found {len(docs_to_add)}'}), 400
        
        # Load indexer
        indexer = get_indexer()
        print(f"📊 Index before: {len(indexer.forward_index)} docs")
        
        # Validate and filter
        valid_docs = []
        count_invalid = 0
        count_duplicate = 0
        
        for raw_doc in docs_to_add:
            doc = validate_doc(raw_doc)
            if not doc:
                count_invalid += 1
                continue
            if doc['id'] in indexer.forward_index:
                count_duplicate += 1
                continue
            valid_docs.append(doc)
        
        print(f"✓ Valid: {len(valid_docs)}, Invalid: {count_invalid}, Duplicates: {count_duplicate}")
        
        if not valid_docs:
            return jsonify({
                'error': 'No valid documents to add',
                'count_invalid': count_invalid,
                'count_duplicate': count_duplicate
            }), 400
        
        # Batch add
        print(f"⚡ Indexing {len(valid_docs)} documents...")
        success, failed = indexer.batch_add(valid_docs)
        
        # Update search engine
        for doc in valid_docs:
            if doc['id'] in indexer.forward_index:
                search_engine.raw_docs[doc['id']] = doc
        
        print(f"📊 Index after: {len(indexer.forward_index)} docs")
        print(f"✓ Success: {success}, Failed: {failed}")
        
        elapsed = time.time() - start_time
        print(f"⏱️  Total time: {elapsed:.2f}s")
        print("="*70 + "\n")
        
        # Unload indexer
        unload_indexer()
        
        return jsonify({
            'success': True,
            'message': f'Indexed {success} documents in {elapsed:.1f}s',
            'count_added': success,
            'count_failed': failed,
            'count_skipped': count_duplicate,
            'count_invalid': count_invalid,
            'time_s': round(elapsed, 2)
        })
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        unload_indexer()
        return jsonify({'error': str(e)}), 500

# ------------------ INDEXER STATUS ------------------
@app.route('/api/indexer/status')
def indexer_status():
    """Get indexer status"""
    global _indexer_instance
    return jsonify({
        'loaded': _indexer_instance is not None,
        'status': 'loaded' if _indexer_instance else 'unloaded',
        'message': 'Indexer is in memory' if _indexer_instance else 'Indexer loads on-demand'
    })

@app.route('/api/indexer/unload', methods=['POST'])
def force_unload_indexer():
    """Force unload indexer from memory"""
    unload_indexer()
    return jsonify({'success': True, 'message': 'Indexer unloaded from memory'})

# ------------------ DEBUG ENDPOINTS ------------------
@app.route('/api/debug/index_status')
def debug_index_status():
    """Debug endpoint to check index state"""
    try:
        indexer = get_indexer()
        stats = indexer.get_stats()
        
        files_status = {
            'lexicon': os.path.exists(indexer.lexicon_file),
            'forward_index': os.path.exists(indexer.forward_index_file),
            'inverted_index': os.path.exists(indexer.inverted_index_file),
            'barrels_folder': os.path.exists(indexer.barrels_folder)
        }
        
        file_sizes = {}
        for name, path in [
            ('lexicon', indexer.lexicon_file),
            ('forward_index', indexer.forward_index_file),
            ('inverted_index', indexer.inverted_index_file)
        ]:
            if os.path.exists(path):
                size_bytes = os.path.getsize(path)
                file_sizes[name] = f"{size_bytes / 1024:.2f} KB"
            else:
                file_sizes[name] = "N/A"
        
        barrel_files = []
        if os.path.exists(indexer.barrels_folder):
            barrel_files = sorted([f for f in os.listdir(indexer.barrels_folder) 
                                  if f.endswith('.json')])
        
        sample_docs = list(indexer.forward_index.keys())[:5]
        
        return jsonify({
            'stats': stats,
            'files_exist': files_status,
            'file_sizes': file_sizes,
            'barrel_files': barrel_files,
            'sample_doc_ids': sample_docs,
            'base_path': indexer.base_path
        })
        
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/debug/paths')
def debug_paths():
    """Show all paths being used"""
    indexer = get_indexer()
    
    paths = {
        'base_path': indexer.base_path,
        'lexicon': indexer.lexicon_file,
        'forward_index': indexer.forward_index_file,
        'inverted_index': indexer.inverted_index_file,
        'barrels': indexer.barrels_folder,
        'metadata': indexer.doc_meta_file
    }
    
    result = {}
    for name, path in paths.items():
        if os.path.isfile(path):
            result[name] = {
                'path': path,
                'exists': os.path.exists(path),
                'size': os.path.getsize(path) if os.path.exists(path) else 0
            }
        else:
            result[name] = {
                'path': path,
                'exists': os.path.exists(path)
            }
    
    return jsonify(result)

@app.route('/api/debug/document/<doc_id>')
def debug_get_document(doc_id):
    """Get detailed info about a specific document"""
    try:
        indexer = get_indexer()
        
        forward_entry = indexer.forward_index.get(doc_id)
        metadata = indexer.doc_meta.get(doc_id)
        search_doc = search_engine.raw_docs.get(doc_id)
        
        return jsonify({
            'doc_id': doc_id,
            'exists_in_forward_index': forward_entry is not None,
            'exists_in_metadata': metadata is not None,
            'exists_in_search_engine': search_doc is not None,
            'forward_index_data': forward_entry,
            'metadata': metadata,
            'search_engine_data': search_doc
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== MAIN ====================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("ARXIV SEARCH ENGINE - COLAB EDITION")
    print("="*70)
    print(f"📂 Project path: {BASE_PROJECT_PATH}")
    print("✓ Server: http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, port=5000, threaded=True)