"""Tests for the read_docs tool: list, search, read, section extraction."""

from __future__ import annotations

import pytest

from min_agent.errors import ToolError
from min_agent.tools.read_docs import _resolve, build_read_docs_tool


@pytest.fixture()
def docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "alpha.md").write_text(
        "# Alpha\n\nFirst line about memory.\nSecond line.\n", encoding="utf-8"
    )
    (d / "beta.md").write_text("# Beta\n\nWeather query here.\n", encoding="utf-8")
    return d


def _tool(docs):
    return build_read_docs_tool(docs)


def test_list_returns_files_with_sizes(docs):
    res = _tool(docs).fn(action="list")
    assert "alpha.md" in res
    assert "beta.md" in res
    assert "bytes" in res


def test_search_hits_matching_lines(docs):
    res = _tool(docs).fn(action="search", query="memory")
    assert "alpha.md" in res
    assert "beta.md" not in res


def test_search_chinese_query(docs):
    (docs / "gamma.md").write_text("# Gamma\n\n记忆分三层。\n", encoding="utf-8")
    res = _tool(docs).fn(action="search", query="记忆")
    assert "gamma.md" in res


def test_search_no_match(docs):
    res = _tool(docs).fn(action="search", query="zzzznonexistent")
    assert "No lines" in res


def test_read_full_file(docs):
    res = _tool(docs).fn(action="read", name="alpha.md")
    assert "First line about memory." in res


def test_read_section(docs):
    (docs / "sections.md").write_text(
        "# Top\n\n## Details\n\nDetail body.\n\n## Other\n\nOther body.\n", encoding="utf-8"
    )
    res = _tool(docs).fn(action="read", name="sections.md", section="Details")
    assert "Detail body." in res
    assert "Other body." not in res


def test_read_section_not_found(docs):
    with pytest.raises(ToolError, match="No section"):
        _tool(docs).fn(action="read", name="alpha.md", section="Missing")


def test_read_truncates_long_files(docs):
    (docs / "long.md").write_text("x" * 7000, encoding="utf-8")
    res = _tool(docs).fn(action="read", name="long.md")
    assert "truncated" in res
    assert len(res) < 7000


def test_resolve_exact_name(docs):
    assert _resolve(docs, "alpha.md").name == "alpha.md"


def test_resolve_substring(docs):
    assert _resolve(docs, "alph").name == "alpha.md"


def test_resolve_not_found(docs):
    with pytest.raises(ToolError, match="No document"):
        _resolve(docs, "nonexistent")