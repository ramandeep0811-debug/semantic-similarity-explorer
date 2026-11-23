"""Semantic Similarity Explorer - AI PBL (Project 16)
Run in VS Code terminal:  python main.py
"""
import wordnet_tools as wt
import embeddings as emb


def rule():
    print("-" * 64)


def do_lookup():
    word = input("Word: ")
    info = wt.word_info(word)
    if not info:
        print("Not found in WordNet.")
        return
    rule()
    for i, s in enumerate(info, 1):
        print(f"{i}. [{s['pos']}] {s['name']}: {s['definition']}")
        if s["example"]:
            print(f"     e.g. \"{s['example']}\"")


def do_syn_ant():
    word = input("Word: ")
    syn, ant = wt.synonyms_antonyms(word)
    rule()
    print("Synonyms:", ", ".join(syn) if syn else "none found")
    print("Antonyms:", ", ".join(ant) if ant else "none found")


def do_related():
    word = input("Word: ")
    rel = wt.related_words(word)
    rule()
    for kind, words in rel.items():
        shown = ", ".join(words[:12]) + (" ..." if len(words) > 12 else "")
        print(f"{kind.title():10}: {shown if words else 'none'}")


def do_similarity():
    w1, w2 = input("First word: "), input("Second word: ")
    r = wt.similarity(w1, w2)
    rule()
    if r is None:
        print("One of the words is not in WordNet.")
        return
    a, b = r["pair"]
    print(f"Best matching senses: {a.name()}  <->  {b.name()}")
    print(f"  {a.definition()}\n  {b.definition()}")
    print(f"Path similarity : {r['path']:.3f}")
    print(f"Wu-Palmer       : {r['wup']:.3f}  ({wt.label(r['wup'])})")
    print(f"LCH             : {r['lch']:.3f}  (raw score, higher = closer)")


def do_compare():
    if not emb.available():
        print("NumPy not installed. Run: python -m pip install -r requirements-optional.txt")
        return
    w1, w2 = input("First word: "), input("Second word: ")
    r = wt.similarity(w1, w2)
    g = emb.cosine(w1, w2)
    rule()
    print(f"WordNet Wu-Palmer : {r['wup']:.3f}" if r else "WordNet Wu-Palmer : n/a")
    print(f"GloVe cosine      : {g:.3f}" if g is not None else "GloVe cosine      : n/a")
    print("Nearest to", w1, "in GloVe:", [w for w, _ in emb.nearest(w1)])


def do_table():
    use_glove = emb.available() and input("Include GloVe column? (y/n): ").lower() == "y"
    rule()
    head = f"{'Word pair':26}{'Path':>8}{'Wu-P':>8}{'LCH':>8}"
    print(head + (f"{'GloVe':>8}" if use_glove else "") + "  Verdict")
    for a, b in wt.SAMPLE_PAIRS:
        r = wt.similarity(a, b)
        if r is None:
            continue
        row = f"{a + ' / ' + b:26}{r['path']:>8.3f}{r['wup']:>8.3f}{r['lch']:>8.3f}"
        if use_glove:
            g = emb.cosine(a, b)
            row += f"{g:>8.3f}" if g is not None else f"{'n/a':>8}"
        print(row + "  " + wt.label(r["wup"]))


MENU = {
    "1": ("Word lookup (senses & definitions)", do_lookup),
    "2": ("Synonyms and antonyms", do_syn_ant),
    "3": ("Related words (broader/narrower/part-of)", do_related),
    "4": ("Similarity score between two words", do_similarity),
    "5": ("Compare WordNet vs GloVe (optional)", do_compare),
    "6": ("Sample word-pair table", do_table),
}


def main():
    wt.ensure_wordnet()
    while True:
        print("\n=== Semantic Similarity Explorer ===")
        for k, (title, _) in MENU.items():
            print(f" {k}. {title}")
        print(" 0. Exit")
        choice = input("Choose: ").strip()
        if choice == "0":
            print("Bye!")
            break
        if choice in MENU:
            MENU[choice][1]()
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
