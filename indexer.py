"""
Build a TF-only forward and inverted index for a TREC style corpus.

The output mirrors the Project 1/2 contract:

    term_dictionary.txt   stem<TAB>termID         (alphabetical)
    doc_dictionary.txt    DOCNO<TAB>docID         (sorted by docID)
    forward_index.txt     docID: termID: tf; termID: tf; ...
    inverted_index.txt    termID: docID: tf; docID: tf; ...

Only the contents of the ``<TEXT>`` tag are indexed (matching the
behaviour of the C++ indexer used in the previous projects).  Term
frequencies are stored without any normalisation; IDF and cosine
weights are computed at query time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from tokenize_text import load_stopwords, tokenize

DOC_RE = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
DOCNO_RE = re.compile(r"<DOCNO>(.*?)</DOCNO>", re.DOTALL)
TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)


def _doc_numeric_id(docno: str) -> int:
    if "-" in docno:
        try:
            return int(docno.split("-", 1)[1])
        except ValueError:
            return 0
    return 0


def _iter_corpus_files(corpus_dir: str | Path) -> List[Path]:
    p = Path(corpus_dir)
    if p.is_file():
        return [p]
    return sorted(x for x in p.iterdir() if x.is_file())


def build_index(corpus_dir: str | Path,
                stopwords_path: str | Path,
                output_dir: str | Path) -> Dict[str, str]:
    """Build the index and return a mapping of artefact name to file path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stopwords = load_stopwords(stopwords_path)

    docno_by_id: Dict[int, str] = {}
    doc_term_freq: Dict[int, Dict[str, int]] = {}
    all_stems: set[str] = set()

    for corpus_file in _iter_corpus_files(corpus_dir):
        with open(corpus_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        for block in DOC_RE.findall(content):
            docno_match = DOCNO_RE.search(block)
            if not docno_match:
                continue
            docno = re.sub(r"\s+", "", docno_match.group(1))
            doc_id = _doc_numeric_id(docno)
            if doc_id == 0:
                continue
            text_match = TEXT_RE.search(block)
            if not text_match:
                continue
            tokens = tokenize(text_match.group(1), stopwords)
            tf: Dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
                all_stems.add(tok)
            docno_by_id[doc_id] = docno
            doc_term_freq[doc_id] = tf

    sorted_stems = sorted(all_stems)
    term_to_id: Dict[str, int] = {s: i + 1 for i, s in enumerate(sorted_stems)}

    forward: Dict[int, List[Tuple[int, int]]] = {}
    inverted: Dict[int, List[Tuple[int, int]]] = {tid: [] for tid in term_to_id.values()}
    for doc_id in sorted(doc_term_freq.keys()):
        postings: List[Tuple[int, int]] = []
        for stem_str, freq in doc_term_freq[doc_id].items():
            term_id = term_to_id[stem_str]
            postings.append((term_id, freq))
        postings.sort()
        forward[doc_id] = postings
        for term_id, freq in postings:
            inverted[term_id].append((doc_id, freq))

    paths = {
        "term_dictionary": str(out / "term_dictionary.txt"),
        "doc_dictionary": str(out / "doc_dictionary.txt"),
        "forward_index": str(out / "forward_index.txt"),
        "inverted_index": str(out / "inverted_index.txt"),
    }

    with open(paths["term_dictionary"], "w", encoding="utf-8") as f:
        for s in sorted_stems:
            f.write(f"{s}\t{term_to_id[s]}\n")

    with open(paths["doc_dictionary"], "w", encoding="utf-8") as f:
        for doc_id in sorted(docno_by_id.keys()):
            f.write(f"{docno_by_id[doc_id]}\t{doc_id}\n")

    with open(paths["forward_index"], "w", encoding="utf-8") as f:
        for doc_id in sorted(forward.keys()):
            postings_str = "; ".join(f"{tid}: {tf}" for tid, tf in forward[doc_id])
            f.write(f"{doc_id}: {postings_str}\n")

    with open(paths["inverted_index"], "w", encoding="utf-8") as f:
        for term_id in sorted(inverted.keys()):
            postings_str = "; ".join(f"{did}: {tf}" for did, tf in inverted[term_id])
            f.write(f"{term_id}: {postings_str}\n")

    print(f"Indexed {len(docno_by_id)} documents and {len(sorted_stems)} unique terms.")
    return paths


def load_index(index_dir: str | Path):
    """Load an index that was previously written by ``build_index``."""
    index_dir = Path(index_dir)
    term_to_id: Dict[str, int] = {}
    id_to_term: Dict[int, str] = {}
    with open(index_dir / "term_dictionary.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            term, tid = parts[0], int(parts[1])
            term_to_id[term] = tid
            id_to_term[tid] = term

    docno_to_id: Dict[str, int] = {}
    id_to_docno: Dict[int, str] = {}
    with open(index_dir / "doc_dictionary.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            docno, did = parts[0], int(parts[1])
            docno_to_id[docno] = did
            id_to_docno[did] = docno

    forward: Dict[int, Dict[int, int]] = {}
    with open(index_dir / "forward_index.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").strip()
            if not line:
                continue
            head, _, body = line.partition(":")
            doc_id = int(head.strip())
            postings: Dict[int, int] = {}
            for chunk in body.split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                a, _, b = chunk.partition(":")
                postings[int(a.strip())] = int(b.strip())
            forward[doc_id] = postings

    inverted: Dict[int, Dict[int, int]] = {}
    with open(index_dir / "inverted_index.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").strip()
            if not line:
                continue
            head, _, body = line.partition(":")
            term_id = int(head.strip())
            postings: Dict[int, int] = {}
            for chunk in body.split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                a, _, b = chunk.partition(":")
                postings[int(a.strip())] = int(b.strip())
            inverted[term_id] = postings

    return {
        "term_to_id": term_to_id,
        "id_to_term": id_to_term,
        "docno_to_id": docno_to_id,
        "id_to_docno": id_to_docno,
        "forward": forward,
        "inverted": inverted,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build TF-only inverted index for a TREC corpus.")
    parser.add_argument("--corpus", default="data/ft911", help="Path to TREC corpus directory or file.")
    parser.add_argument("--stopwords", default="stopwords.txt", help="Path to stopword list.")
    parser.add_argument("--output", default="index", help="Output directory for index artefacts.")
    args = parser.parse_args()

    build_index(args.corpus, args.stopwords, args.output)
