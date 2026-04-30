"""
Parse the TREC ``topics.txt`` file into a list of structured queries.

Each topic is delimited by ``<top> ... </top>`` and contains four
fields: ``<num>``, ``<title>``, ``<desc>`` and ``<narr>``.  We strip
the leading "Number:", "Description:" and "Narrative:" labels that
the corpus uses and return the remaining text untouched -- the
caller is expected to apply tokenisation/stemming.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Topic:
    number: str
    title: str
    description: str
    narrative: str

    def text(self, mode: str) -> str:
        if mode == "title":
            return self.title
        if mode == "title+desc":
            return f"{self.title} {self.description}"
        if mode == "title+narr":
            return f"{self.title} {self.narrative}"
        raise ValueError(f"Unknown mode: {mode}")


_TOP_RE = re.compile(r"<top>(.*?)</top>", re.DOTALL | re.IGNORECASE)


def _extract(block: str, tag: str, next_tags: List[str]) -> str:
    """Extract text after ``<tag>`` until the next of ``next_tags`` (or end)."""
    lookahead_parts = [rf"<{t}>" for t in next_tags] + [r"\Z"]
    pattern = rf"<{tag}>(.*?)(?=" + "|".join(lookahead_parts) + r")"
    match = re.search(pattern, block, re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def _strip_label(text: str, label: str) -> str:
    pattern = rf"^\s*{label}\s*:?\s*"
    return re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()


def parse_topics(path: str | Path) -> List[Topic]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    topics: List[Topic] = []
    for block in _TOP_RE.findall(content):
        num_raw = _extract(block, "num", ["title", "desc", "narr"])
        title_raw = _extract(block, "title", ["desc", "narr"])
        desc_raw = _extract(block, "desc", ["narr"])
        narr_raw = _extract(block, "narr", [])

        number = _strip_label(num_raw, "Number")
        number = re.sub(r"\s+", "", number)
        title = _strip_label(title_raw, "Topic")
        description = _strip_label(desc_raw, "Description")
        narrative = _strip_label(narr_raw, "Narrative")

        topics.append(
            Topic(
                number=number,
                title=" ".join(title.split()),
                description=" ".join(description.split()),
                narrative=" ".join(narrative.split()),
            )
        )
    return topics


if __name__ == "__main__":
    for t in parse_topics("topics.txt"):
        print(f"{t.number}: title={t.title!r}")
        print(f"     desc={t.description!r}")
        print(f"     narr={t.narrative!r}")
        print()
