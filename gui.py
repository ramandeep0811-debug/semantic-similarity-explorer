"""Semantic Similarity Explorer - desktop GUI (Tkinter, no extra install).
Run:  python gui.py
Uses the same wordnet_tools.py and embeddings.py as the console menu (main.py).
"""
import queue
import threading
import tkinter as tk
from tkinter import ttk

import embeddings as emb
import wordnet_tools as wt


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Semantic Similarity Explorer")
        self.geometry("920x640")
        self.minsize(780, 540)

        self.q = queue.Queue()          # results from background threads
        self.status = tk.StringVar(value="Ready")

        ttk.Label(self, text="Semantic Similarity Explorer",
                  font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=12, pady=(10, 0))
        ttk.Label(self, text="AI PBL Project 16  |  WordNet (NLTK) with optional GloVe comparison"
                  ).pack(anchor="w", padx=12, pady=(0, 6))

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=4)
        ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken"
                  ).pack(fill="x", side="bottom")

        self.word_tab(nb, "Lookup", "bank", self.lookup_text)
        self.word_tab(nb, "Synonyms & Antonyms", "happy", self.syn_text)
        self.word_tab(nb, "Related Words", "dog", self.related_text)
        self.build_similarity(nb)
        self.build_compare(nb)
        self.build_table(nb)

        self.after(100, self.poll)

    # ------------------------------------------------------------ helpers
    def poll(self):
        """Run callbacks for finished background jobs (keeps the window responsive)."""
        try:
            while True:
                done, result, err = self.q.get_nowait()
                done(result, err)
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def background(self, work, done, message):
        self.status.set(message)

        def target():
            try:
                self.q.put((done, work(), None))
            except Exception as e:      # network / file problems
                self.q.put((done, None, e))

        threading.Thread(target=target, daemon=True).start()

    @staticmethod
    def output_box(parent):
        frame = ttk.Frame(parent)
        text = tk.Text(frame, wrap="word", font=("Consolas", 10), state="disabled",
                       padx=8, pady=6)
        bar = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        return frame, text

    @staticmethod
    def show(text_widget, content):
        text_widget.configure(state="normal")
        text_widget.delete("1.0", "end")
        text_widget.insert("1.0", content)
        text_widget.configure(state="disabled")

    # ------------------------------------------------------ one-word tabs
    def word_tab(self, nb, title, default, func):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text=title)
        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Label(row, text="Word:").pack(side="left")
        entry = ttk.Entry(row, width=28)
        entry.insert(0, default)
        entry.pack(side="left", padx=6)
        frame, box = self.output_box(tab)

        def go(event=None):
            word = entry.get().strip()
            if not word:
                self.show(box, "Please type a word.")
                return
            self.show(box, func(word))
            self.status.set(f"Done: {title} for '{word}'")

        ttk.Button(row, text="Search", command=go).pack(side="left")
        entry.bind("<Return>", go)
        frame.pack(fill="both", expand=True, pady=(10, 0))
        go()

    @staticmethod
    def lookup_text(word):
        senses = wt.word_info(word)
        if not senses:
            return "Not found in WordNet."
        out = [f"{len(senses)} senses found for '{word}'\n"]
        for i, s in enumerate(senses, 1):
            out.append(f"{i}. [{s['pos']}] {s['name']}: {s['definition']}")
            if s["example"]:
                out.append(f'      e.g. "{s["example"]}"')
        return "\n".join(out)

    @staticmethod
    def syn_text(word):
        syn, ant = wt.synonyms_antonyms(word)
        if not wt.word_info(word):
            return "Not found in WordNet."
        return ("Synonyms:\n  " + (", ".join(syn) if syn else "none found") +
                "\n\nAntonyms:\n  " + (", ".join(ant) if ant else "none found"))

    @staticmethod
    def related_text(word):
        if not wt.word_info(word):
            return "Not found in WordNet."
        out = []
        for kind, words in wt.related_words(word).items():
            shown = ", ".join(words[:25]) + (" ..." if len(words) > 25 else "")
            out.append(f"{kind.title()}:\n  {shown if words else 'none'}\n")
        return "\n".join(out)

    # ------------------------------------------------------- similarity tab
    def pair_inputs(self, tab, d1, d2):
        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Label(row, text="First word:").pack(side="left")
        e1 = ttk.Entry(row, width=18)
        e1.insert(0, d1)
        e1.pack(side="left", padx=6)
        ttk.Label(row, text="Second word:").pack(side="left")
        e2 = ttk.Entry(row, width=18)
        e2.insert(0, d2)
        e2.pack(side="left", padx=6)
        return row, e1, e2

    def build_similarity(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Similarity")
        row, e1, e2 = self.pair_inputs(tab, "dog", "cat")
        self.sim_bar = ttk.Progressbar(tab, maximum=1.0, length=300)
        self.sim_label = ttk.Label(tab, text="", font=("Segoe UI", 11, "bold"))
        frame, box = self.output_box(tab)

        def go(event=None):
            w1, w2 = e1.get().strip(), e2.get().strip()
            if not w1 or not w2:
                self.show(box, "Please type both words.")
                return
            r = wt.similarity(w1, w2)
            if r is None:
                self.sim_bar["value"] = 0
                self.sim_label.config(text="")
                self.show(box, "One of the words is not in WordNet.")
                return
            a, b = r["pair"]
            self.sim_bar["value"] = min(r["wup"], 1.0)
            self.sim_label.config(text=f"Wu-Palmer {r['wup']:.3f}  -  {wt.label(r['wup'])}")
            self.show(box,
                      f"Path similarity : {r['path']:.3f}\n"
                      f"Wu-Palmer       : {r['wup']:.3f}  ({wt.label(r['wup'])})\n"
                      f"LCH             : {r['lch']:.3f}  (raw score, higher = closer)\n\n"
                      f"Best matching senses:\n"
                      f"  {a.name()}: {a.definition()}\n"
                      f"  {b.name()}: {b.definition()}")
            self.status.set(f"Done: similarity of '{w1}' and '{w2}'")

        ttk.Button(row, text="Compare", command=go).pack(side="left")
        e1.bind("<Return>", go)
        e2.bind("<Return>", go)
        self.sim_label.pack(anchor="w", pady=(10, 2))
        self.sim_bar.pack(anchor="w")
        frame.pack(fill="both", expand=True, pady=(10, 0))
        go()

    # ---------------------------------------------------------- GloVe tab
    def build_compare(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="WordNet vs GloVe")
        row, e1, e2 = self.pair_inputs(tab, "doctor", "hospital")
        ttk.Label(tab, text="First use downloads about 65 MB of GloVe vectors (one time).",
                  foreground="#555555").pack(anchor="w", pady=(6, 0))
        frame, box = self.output_box(tab)
        btn = ttk.Button(row, text="Compare")
        btn.pack(side="left")

        def go(event=None):
            w1, w2 = e1.get().strip(), e2.get().strip()
            if not w1 or not w2:
                self.show(box, "Please type both words.")
                return
            if not emb.available():
                self.show(box, "NumPy is not installed.\nRun: python -m pip install -r requirements.txt")
                return
            btn.config(state="disabled")
            self.show(box, "Loading GloVe vectors... (the first time this downloads about 65 MB)")

            def work():
                return emb.cosine(w1, w2), emb.nearest(w1), wt.similarity(w1, w2)

            def done(result, err):
                btn.config(state="normal")
                if err:
                    self.show(box, f"Could not load GloVe:\n{err}\n\n"
                                   "Check your internet connection, or put a GloVe text file at data/glove.txt")
                    self.status.set("GloVe failed")
                    return
                g, near, r = result
                wup = f"{r['wup']:.3f}" if r else "n/a"
                gl = f"{g:.3f}" if g is not None else "n/a (word not in the 50,000-word GloVe vocabulary)"
                lines = [f"WordNet Wu-Palmer : {wup}",
                         f"GloVe cosine      : {gl}", ""]
                if near:
                    lines.append(f"Nearest words to '{w1}' in GloVe:")
                    lines += [f"  {w:<14} {s:.3f}" for w, s in near]
                self.show(box, "\n".join(lines))
                self.status.set(f"Done: WordNet vs GloVe for '{w1}' and '{w2}'")

            self.background(work, done, "Working...")

        btn.config(command=go)
        e1.bind("<Return>", go)
        e2.bind("<Return>", go)
        frame.pack(fill="both", expand=True, pady=(10, 0))

    # --------------------------------------------------------- table tab
    def build_table(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="Sample Table")
        row = ttk.Frame(tab)
        row.pack(fill="x")
        use_glove = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Include GloVe column (downloads ~65 MB first time)",
                        variable=use_glove).pack(side="left")
        btn = ttk.Button(row, text="Build table")
        btn.pack(side="left", padx=10)

        cols = ("pair", "path", "wup", "lch", "glove", "verdict")
        heads = ("Word pair", "Path", "Wu-Palmer", "LCH", "GloVe", "Verdict")
        widths = (190, 80, 90, 80, 80, 150)
        tree = ttk.Treeview(tab, columns=cols, show="headings", height=12)
        for c, h, w in zip(cols, heads, widths):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor="w" if c in ("pair", "verdict") else "center")
        tree.pack(fill="both", expand=True, pady=(10, 0))

        def fill(glove_on):
            tree.delete(*tree.get_children())
            for a, b in wt.SAMPLE_PAIRS:
                r = wt.similarity(a, b)
                if r is None:
                    continue
                g = "-"
                if glove_on:
                    v = emb.cosine(a, b)
                    g = f"{v:.3f}" if v is not None else "n/a"
                tree.insert("", "end", values=(f"{a} / {b}", f"{r['path']:.3f}", f"{r['wup']:.3f}",
                                               f"{r['lch']:.3f}", g, wt.label(r["wup"])))

        def go():
            want = use_glove.get()
            if want and not emb.available():
                self.status.set("NumPy is not installed - showing WordNet columns only")
                want = False
            btn.config(state="disabled")

            def done(_, err):
                btn.config(state="normal")
                if err:
                    fill(False)
                    self.status.set(f"GloVe failed ({err}); showing WordNet columns only")
                else:
                    fill(want)          # runs on the main thread, GloVe is already loaded
                    self.status.set("Sample table ready")

            self.background(lambda: self._table_work(want), done, "Building table...")

        btn.config(command=go)
        fill(False)

    @staticmethod
    def _table_work(glove_on):
        # Tk widgets must be updated on the main thread, so only load GloVe here.
        if glove_on:
            emb.load()
        return None


def main():
    wt.ensure_wordnet()
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
