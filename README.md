# Semantic Similarity Explorer
AI PBL (5.0CE107C01H) - Project 16. Uses WordNet (NLTK) for synonyms, antonyms,
related words and similarity scores. Optional GloVe comparison.

## Run in VS Code
1. Install Python 3.9+ and the **Python** extension in VS Code.
2. File > Open Folder... > select `semantic_similarity_explorer`.
3. Open a terminal (Ctrl + `) and create a virtual environment:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. `pip install -r requirements.txt`
5. Press F5 (uses `.vscode/launch.json`) or run `python main.py`.
6. Optional GloVe: `pip install -r requirements-optional.txt`

WordNet data downloads automatically on the first run (internet needed once).

## Files
- `main.py` - menu-driven interface
- `wordnet_tools.py` - lookup, synonyms/antonyms, related words, similarity
- `embeddings.py` - optional GloVe comparison
