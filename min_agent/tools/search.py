"""``search`` -- a mock web search.

Mocked on purpose (the brief allows it) but *honestly* mocked: it is a real
BM25-lite ranker over a small fixed corpus, so the model gets back something
shaped like search results -- ranked, snippet-truncated, with a source URL and
a date -- rather than a fixed string.  That is what makes the loop behaviour
(did the snippet help? does it need a follow-up search?) testable.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from ..errors import ToolError
from .base import tool

# A deliberately small, stable corpus.  Swap this module out for a real HTTP
# search client and nothing else in the agent changes.
_CORPUS: list[dict[str, Any]] = [
    {
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "url": "https://arxiv.org/abs/2210.03629",
        "date": "2022-10-06",
        "text": (
            "Reasoning traces let a language model take actions and incorporate observations into "
            "further reasoning. The interleaving of reasoning and action allows the model to "
            "dynamically construct and maintain a plan, adapt to exceptions, and improve over "
            "iterations. ReAct works in question answering, fact verification and text-to-SQL."
        ),
    },
    {
        "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
        "url": "https://arxiv.org/abs/2302.04761",
        "date": "2023-02-09",
        "text": (
            "Toolformer learns when and how to call external APIs by filtering self-generated "
            "samples that reduce perplexity on a held-out set. The method works for a single API, "
            "multiple APIs and even for a model to choose between its own future predictions."
        ),
    },
    {
        "title": "Lost in the Middle: How Language Models Use Long Contexts",
        "url": "https://arxiv.org/abs/2307.03172",
        "date": "2023-07-06",
        "text": (
            "Performance degrades when relevant information appears in the middle of a long input "
            "sequence, even with full attention. Effective context usage is U-shaped: information at "
            "the beginning and end is used best. Retrieval and reranking help recover accuracy."
        ),
    },
    {
        "title": "Compacting Language Model Contexts with a Focus on Summarization",
        "url": "https://arxiv.org/abs/2109.01982",
        "date": "2021-09-04",
        "text": (
            "ChatGPT recycles the conversation buffer with summarization to keep information from "
            "early turns while dropping exact phrasing. The summary is used as a stand-in for the "
            "messages it replaces, and works well as long as the detail is not needed verbatim."
        ),
    },
    {
        "title": "A Survey on Large Language Model based Autonomous Agents",
        "url": "https://arxiv.org/abs/2308.11432",
        "date": "2023-08-22",
        "text": (
            "Agent frameworks are organised around a planning module, a memory module and a tool "
            "module. Memory is usually split into short-term working memory inside the context "
            "window and long-term memory stored outside it and retrieved on demand."
        ),
    },
    {
        "title": "The Reflexion paper: verbal reinforcement learning",
        "url": "https://arxiv.org/abs/2303.11366",
        "date": "2023-03-20",
        "text": (
            "Reflexion lets an agent critique its own output in natural language and store the "
            "reflection in episodic memory, then use that memory to improve the next attempt. "
            "No weight updates are needed; the improvement comes entirely from the text."
        ),
    },
    {
        "title": "Anthropic Claude documentation: tool use",
        "url": "https://docs.anthropic.com/en/docs/tool-use",
        "date": "2025-01-01",
        "text": (
            "A tool definition has a name, a description and an input_schema. The model returns a "
            "tool_use block with an id and the input object; the caller must reply with a "
            "tool_result block carrying the same id. Every tool_use must be answered or the next "
            "request will be rejected."
        ),
    },
    {
        "title": "Shanghai climate: monsoon and humidity by month",
        "url": "https://example.org/climate/shanghai",
        "date": "2024-03-02",
        "text": (
            "Shanghai has a subtropical monsoon climate. July and August are the hottest and "
            "humidest months, with average highs near 32 C and heavy plum rain. January is the "
            "coolest and driest month, averaging about 5 C with northerly winds."
        ),
    },
    {
        "title": "Weekly report writing guide",
        "url": "https://example.org/guides/weekly-report",
        "date": "2024-06-11",
        "text": (
            "A weekly report usually has three parts: what was done against last week's plan, what "
            "is in progress with its blocker, and what is planned for next week. Keep numbers "
            "concrete and list risks explicitly rather than hiding them in prose."
        ),
    },
]

_TOKEN = re.compile(r"[a-z0-9]+|[一-鿿]")
_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are", "how", "what",
    "with", "that", "this", "it", "as", "by", "at", "from", "be", "do", "does", "can", "you",
    "我", "的", "了", "是", "在", "和", "与", "吗", "呢", "吧", "个", "请", "帮", "一下", "怎么",
    "什么", "如何", "一个",
}


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP:
            continue
        out.extend(_cjk_bigrams(tok) if _is_cjk(tok) else [tok])
    return out


def _is_cjk(tok: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in tok)


def _cjk_bigrams(tok: str) -> list[str]:
    if len(tok) == 1:
        return [tok]
    return [tok[i : i + 2] for i in range(len(tok) - 1)]


def search_corpus(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Rank the corpus against ``query``. Exposed for tests."""
    q = _tokens(query)
    if not q:
        return []
    qset = set(q)
    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in _CORPUS:
        haystack = _tokens(f"{doc['title']} {doc['title']} {doc['text']}")  # title weighted x2
        if not haystack:
            continue
        overlap = sum(1 for t in haystack if t in qset)
        if overlap == 0:
            continue
        norm = overlap / (len(haystack) ** 0.5 + 1)
        scored.append((norm, doc))
    scored.sort(key=lambda pair: -pair[0])
    return [
        {
            "title": doc["title"],
            "url": doc["url"],
            "date": doc["date"],
            "score": round(score, 4),
            "snippet": _snippet(doc["text"], qset),
        }
        for score, doc in scored[:limit]
    ]


def _snippet(text: str, qset: set[str], width: int = 220) -> str:
    toks = _tokens(text)
    if not toks:
        return text[:width]
    hit = next((i for i, t in enumerate(toks) if t in qset), None)
    if hit is None:
        return text[:width] + ("..." if len(text) > width else "")
    # 1 token == 1 char for our corpus, so a token window is a char window.
    start = max(0, hit - width // 3)
    end = min(len(text), start + width)
    return ("..." if start > 0 else "") + text[start:end].strip() + ("..." if end < len(text) else "")


@tool(tags=("knowledge",))
def search(
    query: Annotated[str, "the search query, in the user's language; 2-20 words works best"],
    limit: Annotated[int, "how many results to return (1-5, default 3)"] = 3,
) -> str:
    """Search the web for a query and return ranked results with title, url, date and snippet. Use this before answering anything you are not certain about."""
    if not query.strip():
        raise ToolError("Empty query")
    limit = max(1, min(int(limit), 5))
    hits = search_corpus(query, limit=limit)
    if not hits:
        return f"No results for {query!r}. (This is a mocked search over a small offline corpus; try different keywords.)"
    lines = [f"{i}. {h['title']}\n   {h['url']} ({h['date']})\n   {h['snippet']}" for i, h in enumerate(hits, 1)]
    return "\n".join(lines)
