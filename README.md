# Project 3 — Query Retrieval and Performance Analysis

CSCE 5200 Information Retrieval and Web Search.

This package implements the query-processing portion of a search engine
on top of the Project 1/2 inverted index.  It supports the **Vector
Space Model** with **TF-IDF** weighting and **cosine similarity** as
defined in Manning, Raghavan & Schütze, Chapter 6 (Figure 6.14).

## Layout

```
.
├── data/ft911/                 # FT911 TREC corpus (15 source files, ~5,368 docs)
├── stopwords.txt               # SMART stopword list (504 entries)
├── topics.txt                  # 4 topics provided with the assignment
├── main.qrels                  # Relevance judgments (provided)
├── porter.py                   # Pure-Python Porter (1980) stemmer
├── tokenize_text.py            # Tokeniser (case-folding, digit drop,
│                               #            stopword drop, Porter stem)
├── indexer.py                  # Builds term/doc dictionaries +
│                               # forward and inverted indexes
├── topics.py                   # Parses topics.txt into structured records
├── vsm.py                      # VSM cosine retriever (lnc.ltc)
├── evaluate.py                 # Precision / Recall vs main.qrels
├── query_processor.py          # End-to-end driver
├── index/                      # Generated index artefacts
│   ├── term_dictionary.txt
│   ├── doc_dictionary.txt
│   ├── forward_index.txt
│   └── inverted_index.txt
├── vsm_output.txt              # Ranked output - title-only setting
├── vsm_output_title_desc.txt   # Ranked output - title + description
├── vsm_output_title_narr.txt   # Ranked output - title + narrative
├── evaluation.txt              # Per-topic precision/recall summary
├── report.md                   # Project report
└── README.md
```

## Running

```bash
python3 query_processor.py            # builds index if missing, runs all 3 settings
python3 query_processor.py --rebuild-index --top-k 100
```

The script has no third-party dependencies; everything is implemented
with the Python standard library so it can be graded directly with
`python3` (3.8+).

### What it does

1. Reads or builds the inverted index (`indexer.py`) for the FT911
   corpus that lives under `data/ft911/`.
2. Parses each topic in `topics.txt` into title / description /
   narrative fields.
3. For each of three query settings runs the cosine score algorithm of
   Figure 6.14 over the inverted index and writes the ranked top-K
   documents in the format

   ```
   TOPIC<TAB>DOCUMENT<TAB>RANK<TAB>COSINE
   ```

   to `vsm_output*.txt`.
4. Computes per-topic **Precision @ 10**, **Precision @ K** and
   **Recall @ K** against `main.qrels` and writes a comparison table
   to `evaluation.txt`.

### Output format

`vsm_output.txt` (title-only) follows the format prescribed in
`readme.txt` exactly:

```
352  FT911-122      1   0.230794
352  FT911-558      2   0.212843
...
```

(separator is a single TAB).

## Notes on the corpus

The relevance judgments file references documents from FT911, FT921,
FT922 and FT923.  Only the **FT911** subset (5,368 documents) is
available on this machine, identical to the corpus used in the previous
projects.  Recall in `evaluation.txt` is reported against the
*in-corpus* relevant set (i.e. the relevant judgments restricted to
FT911) — see `report.md` for details.
