"""Tests for the shared text scanner: tokenize (ranking) and raw_keywords
(literal matching)."""

from __future__ import annotations

from pathlib import Path

from min_agent import textutil
from min_agent.tools.read_docs import _rank, _scan


def test_tokenize_expands_cjk_runs_to_bigrams():
    """A CJK run must yield its 2-grams, not single characters -- otherwise
    "北京" cannot match "在北京工作" through the shared bigram."""
    tokens = textutil.tokenize("在北京工作")
    assert "北京" in tokens
    assert "北" not in tokens
    assert "京" not in tokens


def test_tokenize_keeps_non_stopword_cjk():
    tokens = textutil.tokenize("在北京工作")
    assert "北京" in tokens
    assert "工作" in tokens


def test_tokenize_drops_stopwords():
    tokens = textutil.tokenize("我的是什么")
    assert "我" not in tokens
    assert "的" not in tokens


def test_tokenize_keeps_ascii_words_and_drops_single_letters():
    tokens = textutil.tokenize("the agent a b")
    assert "agent" in tokens
    assert "the" not in tokens
    assert "a" not in tokens
    assert "b" not in tokens


def test_raw_keywords_keeps_cjk():
    """raw_keywords must not drop CJK: read_docs matches lines literally, and
    a Chinese query used to produce an empty set (zero hits)."""
    assert textutil.raw_keywords("北京") != set()
    assert textutil.raw_keywords("记忆") != set()


def test_raw_keywords_matches_bigram_and_single_char():
    kws = textutil.raw_keywords("记忆")
    assert "记忆" in kws
    assert "记" in kws
    assert "忆" in kws


def test_raw_keywords_does_not_filter_stopwords():
    """A word that is a stopword for ranking must still hit in read_docs."""
    assert "how" in textutil.raw_keywords("how")


def test_normalise_collapses_whitespace_and_punctuation():
    assert textutil.normalise("北京。") == textutil.normalise("北 京")
    assert textutil.normalise("北京") == "北京"


def test_read_docs_search_hits_chinese_query(tmp_path):
    """End-to-end: a Chinese query must find the docs that contain it."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# 标题\n\n记忆分三层。\n", encoding="utf-8")
    (docs / "b.md").write_text("# 标题\n\n天气查询。\n", encoding="utf-8")
    files = _scan(docs)
    hits = _rank(files, "记忆")
    assert len(hits) == 1
    assert hits[0][1].name == "a.md"