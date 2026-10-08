"""Shared text tokenisation and matching primitives.

search.py, memory.py, read_docs.py and todo.py each carried their own copy of
the CJK-aware word scanner; this module is the single source of truth so a
query is tokenised exactly like the haystack it is ranked against, and the
dedup key used by one store is the same key the other stores use.

Two consumers, one scanner:

* :func:`tokenize` -- for *ranking* (BM25, memory recall).  Stopwords are
  dropped and a run of CJK characters expands to its 2-grams, so "北京"
  matches "在北京工作" through the shared "北京" bigram.
* :func:`raw_keywords` -- for *literal* line matching (read_docs).  No
  stopword filtering: a user may type a word that is a stopword for ranking
  (e.g. "how") and still expect it to hit.
"""

from __future__ import annotations

import re

# ASCII word runs plus runs of CJK characters (so Chinese bigrams can be derived).
WORD = re.compile(r"[a-z0-9]+|[一-鿿]+")

_EN_STOP = {
    "the", "a", "an", "is", "are", "you", "i", "me", "my", "to", "of", "in",
    "on", "at", "for", "and", "or", "with", "it", "this", "that", "do", "be",
    "how", "what", "can", "as", "by", "from", "does", "one",
}
_CJK_STOP = {
    "我", "你", "的", "了", "是", "吧", "吗", "呢", "个",
    "在", "和", "与", "请", "帮", "一下", "怎么", "什么", "如何", "一个",
}
STOP = frozenset(_EN_STOP | _CJK_STOP)


def is_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def cjk_bigrams(tok: str) -> list[str]:
    if len(tok) == 1:
        return [tok]
    return [tok[i : i + 2] for i in range(len(tok) - 1)]


def _scan(text: str) -> list[str]:
    """Lower-cased ASCII word runs and CJK runs, in order."""
    return WORD.findall(text.lower())


def tokenize(text: str) -> list[str]:
    """Token list used for ranking.

    Case-folded; stopwords dropped; each CJK run expands to its 2-grams
    (so "北京" matches "在北京工作" through the shared "北京" bigram); single
    ASCII letters contribute nothing.
    """
    out: list[str] = []
    for tok in _scan(text):
        if tok in STOP:
            continue
        if len(tok) == 1 and tok.isascii():
            continue
        out.extend(cjk_bigrams(tok) if is_cjk(tok) else [tok])
    return out


def raw_keywords(text: str) -> set[str]:
    """Lower-cased tokens for literal matching, no stopword filtering.

    CJK runs contribute both their 2-grams and their single characters, so a
    query like "记忆" hits a line containing "记忆" and a query like "京" still
    hits "北京".  Single ASCII letters are dropped as noise.
    """
    out: set[str] = set()
    for tok in _scan(text):
        if len(tok) == 1 and tok.isascii():
            continue
        if is_cjk(tok):
            out.update(cjk_bigrams(tok))
            out.update(tok)
        else:
            out.add(tok)
    return out


def normalise(text: str) -> str:
    """Dedup key: lower-cased with all non-word characters removed, so "北京"
    and "北 京" and "北京。" compare equal."""
    return re.sub(r"[\s\W_]+", "", text.lower())