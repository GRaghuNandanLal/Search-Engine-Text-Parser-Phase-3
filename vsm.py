"""
Vector Space Model retrieval using TF-IDF and cosine similarity.

Implementation follows the *cosineScore* pseudo-code in
Manning, Raghavan & Schuetze, Figure 6.14:

    1. For every query term ``t`` look up its postings in the
       inverted index and accumulate the contribution
       ``w_{t,q} * w_{t,d}`` into a score for every document ``d``.
    2. Divide each accumulated score by ``|V(d)|``, the Euclidean
       length of the document's TF-IDF vector.
    3. Return the top-K documents sorted by the resulting score.

Term weighting (lnc.ltc, the SMART scheme recommended by the book):

    document side  -> w_{t,d} = 1 + log10(tf_{t,d})            (logarithmic, no idf)
    query side     -> w_{t,q} = (1 + log10(tf_{t,q})) * log10(N / df_t)

After accumulation the document score is divided by the document
length ``|V(d)| = sqrt(sum_t (1 + log10(tf_{t,d}))^2)`` which gives
true cosine similarity (the query length is a constant per query so
it does not affect the ranking; we still divide by it so the scores
are bounded in [0, 1] and comparable across queries).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


def _doc_length(forward: Dict[int, Dict[int, int]]) -> Dict[int, float]:
    """Pre-compute the lnc length |V(d)| for every document."""
    lengths: Dict[int, float] = {}
    for doc_id, postings in forward.items():
        s = 0.0
        for tf in postings.values():
            w = 1.0 + math.log10(tf)
            s += w * w
        lengths[doc_id] = math.sqrt(s) if s > 0 else 1.0
    return lengths


class VSMRetriever:
    """Pre-computes IDF + document lengths, then ranks queries."""

    def __init__(self, index: Dict, stopwords: set[str]):
        self.term_to_id: Dict[str, int] = index["term_to_id"]
        self.id_to_docno: Dict[int, str] = index["id_to_docno"]
        self.forward: Dict[int, Dict[int, int]] = index["forward"]
        self.inverted: Dict[int, Dict[int, int]] = index["inverted"]
        self.stopwords = stopwords

        self.N = len(self.id_to_docno)
        self.idf: Dict[int, float] = {
            tid: math.log10(self.N / len(postings))
            for tid, postings in self.inverted.items()
            if len(postings) > 0
        }
        self.doc_length = _doc_length(self.forward)

    def _query_weights(self, query_tokens: List[str]) -> Dict[int, float]:
        tf: Dict[int, int] = {}
        for tok in query_tokens:
            tid = self.term_to_id.get(tok)
            if tid is None:
                continue
            tf[tid] = tf.get(tid, 0) + 1
        weights: Dict[int, float] = {}
        for tid, count in tf.items():
            idf = self.idf.get(tid, 0.0)
            if idf <= 0:
                continue
            weights[tid] = (1.0 + math.log10(count)) * idf
        return weights

    def score(self, query_tokens: List[str]) -> List[Tuple[str, float]]:
        q_weights = self._query_weights(query_tokens)
        if not q_weights:
            return []

        scores: Dict[int, float] = {}
        for tid, w_tq in q_weights.items():
            postings = self.inverted.get(tid, {})
            for doc_id, tf_td in postings.items():
                w_td = 1.0 + math.log10(tf_td)
                scores[doc_id] = scores.get(doc_id, 0.0) + w_tq * w_td

        q_length = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0
        ranked: List[Tuple[str, float]] = []
        for doc_id, raw in scores.items():
            cosine = raw / (self.doc_length[doc_id] * q_length)
            ranked.append((self.id_to_docno[doc_id], cosine))
        ranked.sort(key=lambda x: (-x[1], x[0]))
        return ranked
