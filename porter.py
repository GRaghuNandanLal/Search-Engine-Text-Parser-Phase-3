"""
Porter Stemming Algorithm - Python implementation.

Reference: M.F. Porter, "An algorithm for suffix stripping",
Program 14(3), 1980, pp. 130-137.

This module provides a single ``stem(word)`` function used by the
indexer and the query processor so that queries are normalised in
exactly the same way as document text was during indexing.

The implementation here is a transliteration of the classic Porter
algorithm and is deliberately self contained (no external
dependencies) so the project can be graded without installing nltk.
"""

from __future__ import annotations


class _PorterStemmer:
    def __init__(self) -> None:
        self.b = ""
        self.k = 0
        self.j = 0

    def _is_consonant(self, i: int) -> bool:
        ch = self.b[i]
        if ch in ("a", "e", "i", "o", "u"):
            return False
        if ch == "y":
            return True if i == 0 else (not self._is_consonant(i - 1))
        return True

    def _measure(self) -> int:
        n = 0
        i = 0
        while True:
            if i > self.j:
                return n
            if not self._is_consonant(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._is_consonant(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._is_consonant(i):
                    break
                i += 1
            i += 1

    def _vowel_in_stem(self) -> bool:
        return any(not self._is_consonant(i) for i in range(self.j + 1))

    def _double_consonant(self, j: int) -> bool:
        if j < 1:
            return False
        if self.b[j] != self.b[j - 1]:
            return False
        return self._is_consonant(j)

    def _cvc(self, i: int) -> bool:
        if i < 2 or not self._is_consonant(i) or self._is_consonant(i - 1) or not self._is_consonant(i - 2):
            return False
        ch = self.b[i]
        if ch in ("w", "x", "y"):
            return False
        return True

    def _ends(self, suffix: str) -> bool:
        length = len(suffix)
        if length > self.k + 1:
            return False
        if self.b[self.k - length + 1 : self.k + 1] != suffix:
            return False
        self.j = self.k - length
        return True

    def _set_to(self, s: str) -> None:
        length = len(s)
        self.b = self.b[: self.j + 1] + s + self.b[self.j + length + 1 :]
        self.k = self.j + length

    def _replace(self, s: str) -> None:
        if self._measure() > 0:
            self._set_to(s)

    def _step1ab(self) -> None:
        if self.b[self.k] == "s":
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._set_to("i")
            elif self.b[self.k - 1] != "s":
                self.k -= 1
        if self._ends("eed"):
            if self._measure() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowel_in_stem():
            self.k = self.j
            if self._ends("at"):
                self._set_to("ate")
            elif self._ends("bl"):
                self._set_to("ble")
            elif self._ends("iz"):
                self._set_to("ize")
            elif self._double_consonant(self.k):
                self.k -= 1
                if self.b[self.k] in ("l", "s", "z"):
                    self.k += 1
            elif self._measure() == 1 and self._cvc(self.k):
                self._set_to("e")

    def _step1c(self) -> None:
        if self._ends("y") and self._vowel_in_stem():
            self.b = self.b[: self.k] + "i" + self.b[self.k + 1 :]

    def _step2(self) -> None:
        if self.k <= 0:
            return
        ch = self.b[self.k - 1]
        mapping = {
            "a": [("ational", "ate"), ("tional", "tion")],
            "c": [("enci", "ence"), ("anci", "ance")],
            "e": [("izer", "ize")],
            "l": [("bli", "ble"), ("alli", "al"), ("entli", "ent"),
                  ("eli", "e"), ("ousli", "ous")],
            "o": [("ization", "ize"), ("ation", "ate"), ("ator", "ate")],
            "s": [("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
                  ("ousness", "ous")],
            "t": [("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")],
            "g": [("logi", "log")],
        }
        for suffix, replacement in mapping.get(ch, []):
            if self._ends(suffix):
                self._replace(replacement)
                return

    def _step3(self) -> None:
        if self.k < 0:
            return
        ch = self.b[self.k]
        mapping = {
            "e": [("icate", "ic"), ("ative", ""), ("alize", "al")],
            "i": [("iciti", "ic")],
            "l": [("ical", "ic"), ("ful", "")],
            "s": [("ness", "")],
        }
        for suffix, replacement in mapping.get(ch, []):
            if self._ends(suffix):
                self._replace(replacement)
                return

    def _step4(self) -> None:
        if self.k <= 0:
            return
        ch = self.b[self.k - 1]
        suffixes_by_letter = {
            "a": ["al"],
            "c": ["ance", "ence"],
            "e": ["er"],
            "i": ["ic"],
            "l": ["able", "ible"],
            "n": ["ant", "ement", "ment", "ent"],
            "o": ["ion", "ou"],
            "s": ["ism"],
            "t": ["ate", "iti"],
            "u": ["ous"],
            "v": ["ive"],
            "z": ["ize"],
        }
        for suffix in suffixes_by_letter.get(ch, []):
            if self._ends(suffix):
                if suffix == "ion":
                    if self.j >= 0 and self.b[self.j] in ("s", "t") and self._measure() > 1:
                        self.k = self.j
                elif self._measure() > 1:
                    self.k = self.j
                return

    def _step5(self) -> None:
        self.j = self.k
        if self.b[self.k] == "e":
            a = self._measure()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == "l" and self._double_consonant(self.k) and self._measure() > 1:
            self.k -= 1

    def stem(self, word: str) -> str:
        if len(word) <= 2:
            return word
        self.b = word
        self.k = len(word) - 1
        self.j = self.k
        self._step1ab()
        if self.k > 0:
            self._step1c()
            self._step2()
            self._step3()
            self._step4()
            self._step5()
        return self.b[: self.k + 1]


_stemmer = _PorterStemmer()


def stem(word: str) -> str:
    """Return the Porter stem of a single lowercase word."""
    if not word:
        return word
    return _stemmer.stem(word)
