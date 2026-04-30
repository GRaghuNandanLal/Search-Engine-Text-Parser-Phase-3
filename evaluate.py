"""
Evaluate ranked retrieval output against ``main.qrels``.

For each topic we compute:

    * total relevant documents in the qrels file
    * documents retrieved at the top-K cut-off
    * relevant documents retrieved (true positives)
    * precision  = TP / retrieved
    * recall     = TP / relevant

A document not present in the qrels file for a topic is treated as
non-relevant (per the readme).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_qrels(path: str | Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            topic, _, docno, rel = parts
            qrels[topic][docno] = int(rel)
    return qrels


def relevant_docs(qrels_for_topic: Dict[str, int]) -> set[str]:
    return {d for d, r in qrels_for_topic.items() if r > 0}


def precision_recall(retrieved: List[str], relevant: set[str]) -> Tuple[int, int, int, float, float]:
    tp = sum(1 for d in retrieved if d in relevant)
    n_ret = len(retrieved)
    n_rel = len(relevant)
    precision = tp / n_ret if n_ret else 0.0
    recall = tp / n_rel if n_rel else 0.0
    return tp, n_ret, n_rel, precision, recall


def evaluate_run(run: Dict[str, List[Tuple[str, float]]],
                 qrels: Dict[str, Dict[str, int]],
                 cutoff: int | None = None,
                 corpus_docnos: set[str] | None = None) -> Dict[str, Dict[str, float]]:
    """Compute precision/recall per topic.

    If ``corpus_docnos`` is supplied, the relevant set is intersected
    with the documents actually present in the indexed corpus.  This
    yields the "achievable" recall when only part of the original
    TREC collection is available (see project report for context).
    """
    out: Dict[str, Dict[str, float]] = {}
    for topic, ranked in run.items():
        retrieved = [d for d, _ in ranked]
        if cutoff is not None:
            retrieved = retrieved[:cutoff]
        relevant = relevant_docs(qrels.get(topic, {}))
        if corpus_docnos is not None:
            relevant = {d for d in relevant if d in corpus_docnos}
        tp, n_ret, n_rel, p, r = precision_recall(retrieved, relevant)
        out[topic] = {
            "retrieved": n_ret,
            "relevant": n_rel,
            "tp": tp,
            "precision": p,
            "recall": r,
        }
    return out
