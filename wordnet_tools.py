"""WordNet helpers: lookup, synonyms/antonyms, related words, similarity scores."""
import nltk
from nltk.corpus import wordnet as wn

SAMPLE_PAIRS = [
    ("dog", "cat"), ("car", "automobile"), ("car", "bus"),
    ("happy", "sad"), ("doctor", "nurse"), ("doctor", "hospital"),
    ("king", "queen"), ("computer", "laptop"), ("tree", "flower"),
]


def ensure_wordnet():
    """Download WordNet data on first run (needs internet once)."""
    try:
        wn.synsets("test")
    except LookupError:
        print("Downloading WordNet data (one time)...")
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


def clean(word):
    return word.strip().lower().replace(" ", "_")


def word_info(word):
    """Every sense (synset) of a word with its definition and an example."""
    result = []
    for s in wn.synsets(clean(word)):
        result.append({
            "name": s.name(),
            "pos": s.pos(),
            "definition": s.definition(),
            "example": s.examples()[0] if s.examples() else "",
        })
    return result


def synonyms_antonyms(word):
    """Return (synonyms, antonyms) as sorted lists across all senses."""
    w = clean(word)
    syn, ant = set(), set()
    for s in wn.synsets(w):
        for lemma in s.lemmas():
            if lemma.name().lower() != w:
                syn.add(lemma.name().replace("_", " "))
            for a in lemma.antonyms():   # antonyms live on lemmas, not synsets
                ant.add(a.name().replace("_", " "))
    return sorted(syn), sorted(ant)


def related_words(word):
    """Broader (hypernyms), narrower (hyponyms), whole-of (holonyms), part-of (meronyms)."""
    out = {"broader": set(), "narrower": set(), "part of": set(), "has parts": set()}
    for s in wn.synsets(clean(word)):
        for h in s.hypernyms():
            out["broader"].update(l.name().replace("_", " ") for l in h.lemmas())
        for h in s.hyponyms():
            out["narrower"].update(l.name().replace("_", " ") for l in h.lemmas())
        for h in s.part_holonyms() + s.member_holonyms():
            out["part of"].update(l.name().replace("_", " ") for l in h.lemmas())
        for h in s.part_meronyms() + s.member_meronyms():
            out["has parts"].update(l.name().replace("_", " ") for l in h.lemmas())
    return {k: sorted(v) for k, v in out.items()}


def similarity(w1, w2):
    """Best path / Wu-Palmer / LCH score over all sense pairs of the two words.

    Returns None if either word is not in WordNet.
    """
    s1, s2 = wn.synsets(clean(w1)), wn.synsets(clean(w2))
    if not s1 or not s2:
        return None
    best = {"path": 0.0, "wup": 0.0, "lch": 0.0, "pair": None}
    for a in s1:
        for b in s2:
            p = a.path_similarity(b) or 0.0
            w = a.wup_similarity(b) or 0.0
            try:
                l = a.lch_similarity(b) or 0.0   # only defined for same part of speech
            except Exception:
                l = 0.0
            best["path"] = max(best["path"], p)
            best["lch"] = max(best["lch"], l)
            if w > best["wup"] or best["pair"] is None:
                best["wup"] = max(best["wup"], w)
                best["pair"] = (a, b)
    return best


def label(wup):
    if wup >= 0.85:
        return "very similar"
    if wup >= 0.60:
        return "similar"
    if wup >= 0.40:
        return "somewhat related"
    return "weakly related"
