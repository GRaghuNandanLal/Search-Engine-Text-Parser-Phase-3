"""
Query Processor for Project 3 (CSCE 5200).

Reads the inverted/forward index produced by ``indexer.py``, parses
``topics.txt``, runs each query under three settings (title only,
title + description, title + narrative), writes the ranked output
files in the format described in ``readme.txt`` and computes
precision/recall against ``main.qrels``.

Run with no arguments for the standard pipeline:

    python3 query_processor.py

Outputs:
    vsm_output.txt              - ranked output for the title setting
    vsm_output_title_desc.txt   - ranked output for title + description
    vsm_output_title_narr.txt   - ranked output for title + narrative
    evaluation.txt              - per-topic precision/recall comparison
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

from indexer import build_index, load_index
from tokenize_text import load_stopwords, tokenize
from topics import Topic, parse_topics
from vsm import VSMRetriever
from evaluate import evaluate_run, load_qrels


SETTINGS: List[Tuple[str, str, str]] = [
    ("title",       "vsm_output.txt",            "Title only"),
    ("title+desc",  "vsm_output_title_desc.txt", "Title + Description"),
    ("title+narr",  "vsm_output_title_narr.txt", "Title + Narrative"),
]


def write_run(path: str, run: Dict[str, List[Tuple[str, float]]], top_k: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for topic in sorted(run.keys(), key=lambda t: int(t) if t.isdigit() else t):
            ranked = run[topic][:top_k]
            for rank, (docno, score) in enumerate(ranked, start=1):
                f.write(f"{topic}\t{docno}\t{rank}\t{score:.6f}\n")


def run_setting(retriever: VSMRetriever,
                topics: List[Topic],
                stopwords: set[str],
                mode: str,
                top_k: int,
                min_score: float) -> Dict[str, List[Tuple[str, float]]]:
    run: Dict[str, List[Tuple[str, float]]] = {}
    for topic in topics:
        query_text = topic.text(mode)
        tokens = tokenize(query_text, stopwords)
        ranked = retriever.score(tokens)
        ranked = [(d, s) for d, s in ranked if s >= min_score]
        run[topic.number] = ranked[:top_k]
    return run


def write_evaluation(path: str,
                     topics: List[Topic],
                     runs: Dict[str, Dict[str, List[Tuple[str, float]]]],
                     qrels,
                     cutoff: int | None,
                     top_k: int,
                     corpus_docnos: set[str]) -> None:
    lines: List[str] = []
    cutoff_label = f"top {cutoff}" if cutoff else f"top {top_k}"
    lines.append("Project 3 - VSM Cosine Retrieval Evaluation")
    lines.append("===========================================")
    lines.append("")
    lines.append(f"Corpus     : Financial Times 1991 (FT911) - {len(corpus_docnos):,} documents")
    lines.append(f"Cut-off    : {cutoff_label} ranked documents per query")
    lines.append("Weighting  : lnc.ltc (Manning et al. Fig 6.14)")
    lines.append("            documents -> (1 + log10 tf)")
    lines.append("            queries   -> (1 + log10 tf) * log10(N / df)")
    lines.append("Stopwords  : SMART stopword list (504 entries)")
    lines.append("Stemmer    : Porter (1980)")
    lines.append("")
    lines.append("'Relevant (corpus)' is the number of documents judged relevant in the qrels")
    lines.append("file that are actually present in the indexed corpus (FT911 only).  Recall")
    lines.append("is computed against this in-corpus relevant set so it reflects what is")
    lines.append("achievable given the available document collection.")
    lines.append("")

    header = (f"{'Topic':<6} {'Setting':<22} "
              f"{'Rel (qrels)':>11} {'Rel (corpus)':>13} "
              f"{'TP@10':>6} {'P@10':>6} {'TP@K':>5} {'P@K':>7} {'R@K':>7}")
    lines.append(header)
    lines.append("-" * len(header))

    for topic in topics:
        full_rel = len(_relevant_for(qrels, topic.number, None))
        for mode, _out, label in SETTINGS:
            m_full = evaluate_run({topic.number: runs[mode][topic.number]}, qrels,
                                  cutoff=cutoff, corpus_docnos=corpus_docnos)[topic.number]
            m_10 = evaluate_run({topic.number: runs[mode][topic.number]}, qrels,
                                cutoff=10, corpus_docnos=corpus_docnos)[topic.number]
            lines.append(
                f"{topic.number:<6} {label:<22} "
                f"{full_rel:>11} {m_full['relevant']:>13} "
                f"{m_10['tp']:>6} {m_10['precision']:>6.3f} "
                f"{m_full['tp']:>5} {m_full['precision']:>7.4f} {m_full['recall']:>7.4f}"
            )
        lines.append("")

    lines.append("Macro-averages over the four topics")
    lines.append("-----------------------------------")
    macro_header = f"{'Setting':<22} {'Avg P@10':>9} {'Avg P@K':>9} {'Avg R@K':>9}"
    lines.append(macro_header)
    lines.append("-" * len(macro_header))
    for mode, _out, label in SETTINGS:
        per_full = evaluate_run(runs[mode], qrels, cutoff=cutoff, corpus_docnos=corpus_docnos)
        per_10 = evaluate_run(runs[mode], qrels, cutoff=10, corpus_docnos=corpus_docnos)
        n = len(topics)
        avg_p10 = sum(per_10[t.number]["precision"] for t in topics) / n
        avg_p = sum(per_full[t.number]["precision"] for t in topics) / n
        avg_r = sum(per_full[t.number]["recall"] for t in topics) / n
        lines.append(f"{label:<22} {avg_p10:>9.4f} {avg_p:>9.4f} {avg_r:>9.4f}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _relevant_for(qrels, topic: str, corpus_docnos: set[str] | None) -> set[str]:
    rel = {d for d, r in qrels.get(topic, {}).items() if r > 0}
    if corpus_docnos is not None:
        rel &= corpus_docnos
    return rel


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 3 query processor.")
    parser.add_argument("--corpus", default="data/ft911", help="TREC corpus directory.")
    parser.add_argument("--stopwords", default="stopwords.txt")
    parser.add_argument("--index-dir", default="index")
    parser.add_argument("--topics", default="topics.txt")
    parser.add_argument("--qrels", default="main.qrels")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Number of documents to keep per query in vsm_output*.txt files.")
    parser.add_argument("--eval-cutoff", type=int, default=50,
                        help="Cut-off used when computing precision/recall.")
    parser.add_argument("--min-score", type=float, default=1e-6,
                        help="Discard documents with cosine below this threshold.")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Force rebuilding the index even if files already exist.")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    needed = [index_dir / f for f in (
        "term_dictionary.txt", "doc_dictionary.txt",
        "forward_index.txt", "inverted_index.txt",
    )]
    if args.rebuild_index or not all(p.exists() for p in needed):
        print(f"Building index from {args.corpus} -> {index_dir} ...")
        build_index(args.corpus, args.stopwords, index_dir)
    else:
        print(f"Using existing index in {index_dir}.")

    print("Loading index ...")
    index = load_index(index_dir)
    stopwords = load_stopwords(args.stopwords)
    retriever = VSMRetriever(index, stopwords)

    topics = parse_topics(args.topics)
    print(f"Loaded {len(topics)} topics from {args.topics}.")

    runs: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    for mode, out_file, label in SETTINGS:
        print(f"  Running setting: {label} ({mode}) -> {out_file}")
        run = run_setting(retriever, topics, stopwords, mode,
                          top_k=args.top_k, min_score=args.min_score)
        runs[mode] = run
        write_run(out_file, run, top_k=args.top_k)

    qrels = load_qrels(args.qrels)
    corpus_docnos = set(index["docno_to_id"].keys())
    write_evaluation("evaluation.txt", topics, runs, qrels,
                     cutoff=args.eval_cutoff, top_k=args.top_k,
                     corpus_docnos=corpus_docnos)
    print("Wrote evaluation.txt with per-topic precision/recall.")


if __name__ == "__main__":
    main()
