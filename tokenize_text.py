"""
Tokenisation and normalisation utilities.

The same tokenisation rules are applied to both documents and queries
so that a query term such as "Antarctica" is mapped onto the same
stem the indexer wrote for documents containing "Antarctic".

Rules (matching the Project 1/2 indexer):
    1. Lower-case the input.
    2. Split on every non alphanumeric character.
    3. Drop tokens that contain any digit.
    4. Drop tokens that appear in the stopword list.
    5. Apply the Porter stemmer to the surviving tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Set

from porter import stem


def load_stopwords(path: str | Path) -> Set[str]:
    words: Set[str] = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                words.add(w)
    return words


def _contains_digit(token: str) -> bool:
    return any(ch.isdigit() for ch in token)


def tokenize(text: str, stopwords: Set[str]) -> List[str]:
    """Return the ordered list of stems extracted from ``text``."""
    tokens: List[str] = []
    current: List[str] = []
    for ch in text:
        if ch.isalnum():
            current.append(ch.lower())
        elif current:
            tok = "".join(current)
            current.clear()
            if _contains_digit(tok) or tok in stopwords:
                continue
            stemmed = stem(tok)
            if stemmed:
                tokens.append(stemmed)
    if current:
        tok = "".join(current)
        if not _contains_digit(tok) and tok not in stopwords:
            stemmed = stem(tok)
            if stemmed:
                tokens.append(stemmed)
    return tokens


def tokens_to_tf(tokens: Iterable[str]) -> dict:
    tf: dict = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf
