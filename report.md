# Project 3 Report — Query Retrieval and Performance Analysis

**Course:** CSCE 5200 — Information Retrieval and Web Search  
**Project:** 3 of 3 — Query Processor on top of the Project 2 index  

---

## 1. Goal

Extend the Project 2 indexer with a **query processor** that supports
the **Vector Space Model** with **TF-IDF** weighting and uses the
**cosine similarity** as the relevance measure.  The processor must
handle the four queries supplied in `topics.txt` in batch and write a
ranked list of documents per query.  The system must also be evaluated
against the supplied relevance judgments (`main.qrels`) in three
different query settings:

1. **title** — only the `<title>` field is used as the query
2. **title + desc** — the `<title>` and `<desc>` fields are concatenated
3. **title + narr** — the `<title>` and `<narr>` fields are concatenated

## 2. System design

### 2.1 Pipeline

```
        ┌────────────────────┐
        │   topics.txt       │
        └────────┬───────────┘
                 │ parse_topics()
                 ▼
       ┌───────────────────────┐
       │ Topic(title,desc,narr)│
       └────────┬──────────────┘
                │ tokenize_text.tokenize()
                ▼                                  ┌────────────────────┐
        ┌───────────────────┐  ┌────────────┐      │  data/ft911/       │
        │ stems for query   │  │  Porter    │      └────────┬───────────┘
        └────────┬──────────┘  │  stemmer   │               │
                 │             └────────────┘               │ build_index()
                 ▼                                           ▼
        ┌─────────────────────────────────────────────────────────┐
        │   index/  term_dictionary, doc_dictionary,              │
        │           forward_index, inverted_index   (TF-only)     │
        └────────┬────────────────────────────────────────────────┘
                 │
                 │ vsm.VSMRetriever(score)  (Fig 6.14 cosineScore)
                 ▼
        ┌─────────────────────────┐  evaluate.evaluate_run()  ┌────────────────┐
        │ ranked (docno, cosine)  │ ────────────────────────► │ evaluation.txt │
        └────────┬────────────────┘                           └────────────────┘
                 │ write_run()
                 ▼
        vsm_output.txt / vsm_output_title_desc.txt / vsm_output_title_narr.txt
```

### 2.2 Tokenisation and normalisation

Both the indexer and the query processor share `tokenize_text.tokenize`,
ensuring queries are normalised exactly like the documents that were
indexed.  The pipeline performs:

1. **Case folding** to lowercase.
2. **Splitting on non-alphanumeric** characters (the same rule as the
   Project 1 C++ indexer).
3. **Digit filter** — tokens that contain at least one digit are
   discarded (consistent with the previous projects so the term
   dictionary remains identical).
4. **Stopword removal** using the SMART list of 504 entries
   (`stopwords.txt`).
5. **Porter stemming** (`porter.py`, a pure-Python implementation of
   the 1980 algorithm).

The token "exploration" therefore becomes the same stem `explor` in
both the index and a query, which is critical for cosine matching.

### 2.3 Index data structures

The index from Project 2 is reused **unchanged**:

| File                  | Format                                            |
| --------------------- | ------------------------------------------------- |
| `term_dictionary.txt` | `stem<TAB>termID`, alphabetical, IDs from 1       |
| `doc_dictionary.txt`  | `DOCNO<TAB>docID`, sorted by docID                |
| `forward_index.txt`   | `docID: termID: tf; termID: tf; ...`              |
| `inverted_index.txt`  | `termID: docID: tf; docID: tf; ...`               |

Only **term frequencies** are stored — IDF and cosine normalisation are
computed at query time, exactly as the assignment requires.  At
load-time the retriever materialises a few light-weight dictionaries
(`term -> id`, `id -> docno`, `inverted` and `forward` postings) and
*pre-computes*:

* the IDF of every indexed term, `idf_t = log10(N / df_t)`, and
* the document length `|V(d)| = sqrt( Σ_t (1 + log10 tf_{t,d})^2 )`.

These two structures are kept in memory, so scoring a query is
a sparse dot-product over the postings of its terms followed by a
length normalisation.

### 2.4 Term weighting

We use the SMART **lnc.ltc** scheme recommended by Manning et al.
(Section 6.4.3).  This is the most common pairing for ad-hoc retrieval
because the document side avoids IDF (so the document weights can be
pre-computed once) and the cosine length normalisation cancels out
document-length bias:

| component                            | document term weight | query term weight                   |
| ------------------------------------ | -------------------- | ----------------------------------- |
| **logarithmic TF**                   | `1 + log10(tf_{t,d})`| `1 + log10(tf_{t,q})`               |
| **IDF**                              | none                 | `log10(N / df_t)`                   |
| **cosine length normalisation**      | yes                  | yes                                 |

Hence

```
score(q,d) = Σ_t∈q  (1+log10 tf_{t,q})·log10(N/df_t) · (1+log10 tf_{t,d})
             ───────────────────────────────────────────────────────────
                                |V(q)| · |V(d)|
```

This is exactly the *cosineScore* algorithm in Manning et al.
**Figure 6.14** with `wf-idf` weighting.

### 2.5 Cosine retrieval algorithm (Figure 6.14)

```
function cosineScore(q):
    Scores ← {}                                            # docID -> partial score
    for each term t in q:
        compute w_{t,q}
        fetch postings list of t from the inverted index
        for each (d, tf_{t,d}) in postings:
            w_{t,d} ← 1 + log10(tf_{t,d})
            Scores[d] += w_{t,q} · w_{t,d}
    for each d in Scores:
        Scores[d] /= ( |V(d)| · |V(q)| )
    return top-K of Scores in decreasing order
```

The implementation lives in `vsm.VSMRetriever.score`.

### 2.6 Output format

`vsm_output*.txt` follows the prescribed `TOPIC<TAB>DOCUMENT<TAB>UNIQUE#<TAB>COSINE`
layout with cosine values printed to six decimals.  The unique counter
is restarted per topic.

## 3. Corpus availability

The relevance judgments cover documents from the FT911, FT921, FT922
and FT923 partitions of the Financial Times collection.  Only the
**FT911** subset (15 raw files, **5,368 documents**, **32,645** unique
stems) was available — the same corpus used in Project 1 and Project 2.
The retrieval is therefore performed over FT911 only.

To make the recall numbers comparable across queries, the evaluator
restricts the relevant set to the documents that actually exist in the
indexed corpus (`Rel (corpus)` in `evaluation.txt`).  The full qrels
counts are also reported as `Rel (qrels)` for transparency.

## 4. Performance results

The tables below are reproduced from `evaluation.txt` (cut-off **K =
50**, also reporting **P@10**).

### 4.1 Per-topic results

| Topic | Title                       | Setting               | Rel (qrels) | Rel (corpus) | TP@10 | P@10  | TP@K | P@K    | R@K    |
|------:|-----------------------------|-----------------------|------------:|-------------:|------:|-------|-----:|-------:|-------:|
| 352   | British Chunnel impact      | Title only            | 52          | 2            | 0     | 0.000 | 0    | 0.0000 | 0.0000 |
|       |                             | Title + Description   | 52          | 2            | 0     | 0.000 | 0    | 0.0000 | 0.0000 |
|       |                             | Title + Narrative     | 52          | 2            | 0     | 0.000 | 0    | 0.0000 | 0.0000 |
| 353   | Antarctica exploration      | Title only            | 22          | 10           | 6     | 0.600 | 7    | 0.1400 | 0.7000 |
|       |                             | Title + Description   | 22          | 10           | 6     | 0.600 | 7    | 0.1400 | 0.7000 |
|       |                             | **Title + Narrative** | 22          | 10           | **7** | **0.700** | **10** | **0.2000** | **1.0000** |
| 354   | Journalist risks            | **Title only**        | 25          | 9            | **5** | **0.500** | **6** | **0.1200** | **0.6667** |
|       |                             | Title + Description   | 25          | 9            | 4     | 0.400 | 5    | 0.1000 | 0.5556 |
|       |                             | Title + Narrative     | 25          | 9            | 5     | 0.500 | 5    | 0.1000 | 0.5556 |
| 359   | Mutual fund predictors      | **Title only**        | 6           | 1            | 0     | 0.000 | 1    | 0.0200 | **1.0000** |
|       |                             | Title + Description   | 6           | 1            | 0     | 0.000 | 0    | 0.0000 | 0.0000 |
|       |                             | Title + Narrative     | 6           | 1            | 0     | 0.000 | 0    | 0.0000 | 0.0000 |

### 4.2 Macro-averages over the four topics

| Setting               | Avg P@10 | Avg P@K | Avg R@K |
|-----------------------|---------:|--------:|--------:|
| Title only            |   0.2750 |  0.0700 |  0.5917 |
| Title + Description   |   0.2500 |  0.0600 |  0.3139 |
| **Title + Narrative** | **0.3000** |  **0.0750** |  0.3889 |

(P@K and R@K use a cut-off of K = 50 ranked documents per query.  P@K
is mechanically pulled down by sparse relevance judgements — only ~0–10
documents per topic are judged relevant *within FT911* — but the
ranking itself is good, as P@10 makes clear.)

## 5. Discussion

**Title + narrative is the best setting on average.** It wins or ties
on P@10 for every topic where there is at least one relevant document
to find and is the only setting that achieves perfect recall on
Topic 353 (Antarctica exploration).  The narrative explicitly mentions
*"seismology, ionospheric physics, possible economic development"*
and *"banning of mineral mining"*, all of which rank highly because
those terms have very high IDF (they are rare in financial news) and
match documents about Antarctic research that the bare title would
otherwise miss.

**Title only is surprisingly strong on Topic 354 (journalist risks)
and Topic 359 (mutual fund predictors).**  Both titles are themselves
good keyword bags (`journalist`, `risk`, `mutual`, `fund`,
`predictors` after stemming), and the description / narrative
introduce common verbs and connectives that, after IDF weighting, dilute
the signal of the rare key terms — even with TF/IDF weighting the longer
queries lower cosine of the relevant docs relative to longer
non-relevant docs.

**Topic 352 (British Chunnel impact) yields zero retrieval in every
setting**, which is *not* a bug.  Inspection of the only two
documents judged relevant in FT911 (`FT911-808` and `FT911-1868`)
shows the corpus calls the same artefact the **Channel tunnel**, never
the **Chunnel**; neither the title, the description nor the narrative
ever uses the words *channel* or *tunnel*.  In other words, the queries
and the relevant documents share **no content terms at all**.  This is
a textbook example of the **vocabulary-mismatch problem** that motivates
relevance feedback / pseudo-relevance feedback / query expansion using
a thesaurus, none of which is part of the Project 3 scope.  A simple
alias expansion (`Chunnel → Channel tunnel`) would immediately raise
recall on this topic.

**Effect of using description.**  Adding the description rarely helps
and on average *hurts* recall (0.59 → 0.31).  The descriptions in this
topic file are short, lexically conservative reformulations of the
title and contribute mostly low-IDF function words after stopping.

**Effect of using narrative.**  The narrative tends to enumerate the
*aspects* that make a document relevant ("banning of mineral mining",
"taken hostage", "rankings, risks, yields, or costs", ...).  Those
domain terms are individually rare and therefore drive the score of
on-topic documents up.  This is consistent with the expectation that
longer, more content-rich queries help.

**Why P@K looks low.**  At K = 50 every topic has at most ten judged
relevant FT911 documents, so the ceiling is P@K ≤ 0.20.  P@10 is the
fairer absolute precision measure for this benchmark.

## 6. Reproducing the run

```bash
python3 query_processor.py --rebuild-index --top-k 50
```

This produces:

```
index/{term_dictionary,doc_dictionary,forward_index,inverted_index}.txt
vsm_output.txt
vsm_output_title_desc.txt
vsm_output_title_narr.txt
evaluation.txt
```

Total runtime on a laptop is ~7 seconds (most of which is index
construction).

## 7. Files included in the submission

* `porter.py`, `tokenize_text.py`, `indexer.py`, `topics.py`, `vsm.py`,
  `evaluate.py`, `query_processor.py`
* `stopwords.txt`
* `vsm_output.txt`, `vsm_output_title_desc.txt`, `vsm_output_title_narr.txt`
* `evaluation.txt`
* `README.md`, `report.md`
* `index/` (generated artefacts), `data/ft911/` (corpus)
