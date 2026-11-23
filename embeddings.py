"""Optional: GloVe word vectors, loaded with NumPy only (no gensim needed).

First use downloads the 50-dimension GloVe vectors (~65 MB) into ./data.
Manual fallback: put any GloVe-style text file (word v1 v2 ... per line)
at data/glove.txt and it will be used instead of downloading.
"""
import gzip
import os
import urllib.request

URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
       "glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GZ_PATH = os.path.join(DATA_DIR, "glove-wiki-gigaword-50.gz")
TXT_PATH = os.path.join(DATA_DIR, "glove.txt")
MAX_WORDS = 50000          # file is frequency-ordered, so this keeps common words

_vocab = {}
_words = []
_matrix = None


def available():
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def _open_source():
    if os.path.exists(TXT_PATH):
        return open(TXT_PATH, encoding="utf-8")
    if not os.path.exists(GZ_PATH):
        os.makedirs(DATA_DIR, exist_ok=True)
        print("Downloading GloVe vectors (~65 MB, one time)...")
        tmp = GZ_PATH + ".part"
        urllib.request.urlretrieve(URL, tmp)
        os.replace(tmp, GZ_PATH)
    return gzip.open(GZ_PATH, "rt", encoding="utf-8")


def load():
    global _matrix, _vocab, _words
    if _matrix is not None:
        return
    import numpy as np
    words, vecs = [], []
    with _open_source() as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) <= 2:          # skips a "count dim" header line
                continue
            words.append(parts[0])
            vecs.append(np.array(parts[1:], dtype=np.float32))
            if len(words) >= MAX_WORDS:
                break
    m = np.vstack(vecs)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1
    _matrix = m / norms
    _words = words
    _vocab = {w: i for i, w in enumerate(words)}


def cosine(w1, w2):
    load()
    a, b = _vocab.get(w1.strip().lower()), _vocab.get(w2.strip().lower())
    if a is None or b is None:
        return None
    return float(_matrix[a] @ _matrix[b])


def nearest(word, n=5):
    load()
    i = _vocab.get(word.strip().lower())
    if i is None:
        return []
    sims = _matrix @ _matrix[i]
    order = sims.argsort()[::-1]
    return [(_words[j], float(sims[j])) for j in order if j != i][:n]
