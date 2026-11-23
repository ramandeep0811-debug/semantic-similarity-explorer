"""Optional: GloVe word vectors (via gensim) to compare with WordNet scores."""
_model = None


def available():
    try:
        import gensim  # noqa: F401
        return True
    except ImportError:
        return False


def load(name="glove-wiki-gigaword-50"):
    """Load vectors once (about 65 MB download on first use)."""
    global _model
    if _model is None:
        import gensim.downloader as api
        print("Loading GloVe vectors (first time downloads ~65 MB)...")
        _model = api.load(name)
    return _model


def cosine(w1, w2):
    m = load()
    a, b = w1.strip().lower(), w2.strip().lower()
    if a not in m.key_to_index or b not in m.key_to_index:
        return None
    return float(m.similarity(a, b))


def nearest(word, n=5):
    m = load()
    w = word.strip().lower()
    if w not in m.key_to_index:
        return []
    return m.most_similar(w, topn=n)
