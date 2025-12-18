# 🔍 ArXiv Research Paper Search Engine

A **high-performance semantic search engine** for ArXiv research papers, built with advanced data structures and algorithms. This project implements a complete information retrieval system with features like inverted indexing, semantic search, autocomplete, and dynamic document addition.

### **Data Structures & Algorithms**
- **Trie**: Prefix-tree for autocomplete (O(k) lookup, k = query length)
- **Inverted Index**: Term-to-documents mapping in barrel files
- **Forward Index**: Document-to-term positions for proximity scoring
- **Hash Maps**: Lexicon and posting list storage

### **Frontend**
- **HTML & CSS**: Responsive web interface

### **Backend**
- **Flask**


## 💻 Installation

### **Prerequisites**
- Python 3.8 or higher
- pip (Python package manager)
- Git

### **Step 1: Clone the Repository**
```bash
git clone _<repo-link>_
cd DSA-Project
```

### **Step 2: Install Dependencies**
install necessary dependencies:
```bash
pip install flask nltk numpy
```

### **Step 3: Download NLTK Data**
Run Python and execute:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```


### **Step 4: Prepare the Dataset**
1. Place your ArXiv dataset JSON file in `data/raw/arxiv_100k.json`
2. Run preprocessing:
```bash
cd src/preprocessing
python PreProcessing.py
```


### **Step 5: Build the Index**
```bash
cd ../indexing
python lexicon.py
python forward_index.py
python inverted_index.py
python barrels.py
```

---

## 🚀 Usage

### **Running the Search Engine**

1. **Update the base path** in `search_engine.py`:
   ```python
   PROJECT_BASE = "/path/to/DSA-Project"  # Line 15
   ```

2. **Start the Flask server**:
   ```bash
   python search_engine.py
   ```

3. **Access the application**:
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

**Now you can search multiple research papers and add documents too if you want!**
