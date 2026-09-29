"""Three-tier memory.

* *working*   -- the messages in the current turn (in ``session.messages``).
* *episodic*  -- the session transcript + its rolling summary, kept per window
  (``.sessions/<user>/<id>/transcript.jsonl`` and ``summary.md``).
* *semantic*  -- long-term facts shared across all of a user's windows, in
  ``.memory/<user>/facts.json``.  This module ends there.

Two timings matter, and both are documented in the README as the deliberate
answers to "when do you recall, and where do you put it?":

* **store**  -- when the model explicitly calls ``remember`` (LLM decides), and
  again at the end of a window session via one extraction pass.
* **recall** -- at the start of *every* user turn, scored against the current
  input; the top hits are injected into the *system* prompt as background, not
  into the user turn as instructions.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .llm import LLMRequest
from .paths import atomic_write_text

_WORDS = re.compile(r"[a-z0-9]+|[一-鿿]")
_EN_STOP = {
    "the", "a", "an", "is", "are", "you", "i", "me", "my", "to", "of", "in",
    "on", "at", "for", "and", "or", "with", "it", "this", "that", "do", "be",
    "我", "你", "的", "了", "是", "吧", "吗", "呢", "个",
}

# Statements that are almost never durable facts.
_TEMPORAL = ("今天", "昨天", "刚才", "明天", "稍后", "现在", "今早", "今晚", "上周", "下周", "刚刚")
_ACTIONS = ("帮我", "请", "记一下", "查一下", "算一下", "搜一下", "写一下")


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for tok in _WORDS.findall(text.lower()):
        if tok in _EN_STOP:
            continue
        if len(tok) == 1 and tok.isascii():
            continue
        out.extend(_bigrams(tok) if _is_cjk(tok) and len(tok) > 1 else [tok])
    return out


def _bigrams(tok: str) -> list[str]:
    return [tok[i : i + 2] for i in range(len(tok) - 1)]


def _is_cjk(tok: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in tok)


def _normalise(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


@dataclass
class MemoryItem:
    id: str
    text: str
    tags: list[str]
    source_session: str
    created_at: float
    hits: int = 0


def _read_rows(path: Path) -> list[MemoryItem]:
    """Load ``facts.json``, tolerating a file that is absent or corrupt.

    An unreadable store degrades to "no memories" rather than taking down the
    window: memory is an optimisation, and a person asking a question does not
    care whether the long-term file parsed.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return []
    if not isinstance(data, list):
        return []
    rows: list[MemoryItem] = []
    for row in data:
        if not isinstance(row, dict) or "id" not in row or "text" not in row:
            continue
        try:
            rows.append(MemoryItem(**row))
        except TypeError:
            continue  # a row from a future/other schema -- skip, don't crash
    return rows


class MemoryStore:
    """Long-term facts for one user, persisted as JSON."""

    def __init__(self, path: Path):
        self.path = path
        self._items: list[MemoryItem] = []
        self._dirty = False
        self._load()

    # -- persistence -------------------------------------------------------
    # Two rules, both learned the hard way:
    #
    # 1. Nothing on the *read* path writes.  recall() runs at the start of every
    #    user turn, and it used to call save() just to bump a usage counter --
    #    an O(n) rewrite of the whole file on the critical path of every turn.
    #    The counter is diagnostic, not state anything reads back, so it is now
    #    held in memory and flushed at an explicit boundary (:meth:`flush`).
    #    A crash loses hit counts; nothing else.
    # 2. A write is read-merge-write, not a rewrite of a stale list.  Each
    #    window builds its own MemoryStore at startup, so two windows of the
    #    same user hold two divergent in-memory lists; whoever saved second used
    #    to erase the other's facts outright.  Merging on id keeps both.

    def _load(self) -> None:
        self._items = _read_rows(self.path)
        self._dirty = False

    def save(self) -> None:
        """Persist the store, keeping rows another window wrote since we loaded."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged: dict[str, MemoryItem] = {i.id: i for i in _read_rows(self.path)}
        for item in self._items:
            merged[item.id] = item  # ours wins on conflict
        self._items = list(merged.values())
        atomic_write_text(
            self.path,
            json.dumps([asdict(i) for i in self._items], ensure_ascii=False, indent=2),
        )
        self._dirty = False

    def flush(self) -> None:
        """Write pending hit counters.  A no-op when nothing changed."""
        if self._dirty:
            self.save()

    # -- store timing: explicit tool + end-of-session extraction ----------
    def remember(self, text: str, source_session: str, *, tags: Iterable[str] = ()) -> str:
        """Store ``text`` if it passes the gating checks.

        Returns a short status string; '' means rejected.  The gating rules
        are deliberately cheap and rule-driven:
          1. too short, or a question/action, or time-sensitive;
          2. already stored (dedup on the normalised text);
          3. still TTL-relevant (a *replacement* re-light is fine).
        """
        text = text.strip()
        reason = self._gate(text)
        if reason:
            return ""
        existing = self._find_duplicate(text)
        if existing is not None:
            existing.tags = sorted(set(existing.tags) | set(tags))
            existing.hits += 1
            self.save()  # explicit store -> durable now, not deferred
            return f"already remembered: {text}"
        item = MemoryItem(
            id=uuid.uuid4().hex[:12],
            text=text,
            tags=sorted(set(tags)),
            source_session=source_session,
            created_at=time.time(),
        )
        self._items.append(item)
        self.save()  # explicit store -> durable now, not deferred
        return f"remembered: {text}"

    def _gate(self, text: str) -> str:
        """Return a rejection reason, or '' to allow storage."""
        if len(text) < 6:
            return "too short"
        if not any(ch.isalpha() or "\u4e00" <= ch <= "\u9fff" for ch in text):
            return "no meaningful content"
        if re.search(r"\d+[.,]\d+", text):
            return "looks like numeric tool output"
        if any(w in text for w in _TEMPORAL):
            return "time-sensitive"
        if any(w in text for w in ("? ", "？", "吗", "嘛", "？")):
            return "looks like a question"
        if any(text.strip().startswith(w) for w in _ACTIONS):
            return "looks like an instruction"
        return ""

    def _find_duplicate(self, text: str) -> MemoryItem | None:
        needle = _normalise(text)
        for item in self._items:
            if _normalise(item.text) == needle:
                return item
        return None

    # -- recall timing: scores against the user's current turn -------------
    def recall(
        self, query: str, *, top_k: int = 5, ttl_days: int | None = None
    ) -> list[MemoryItem]:
        """Rank memories by term overlap with ``query``, newest-tiebreak.

        No embeddings, no LLM call -- a keyword scorer is the platform this
        minimal agent can afford every turn without a second request.  It is
        also why the memory prompt asks for *distinct nouns and names*.
        """
        now = time.time()
        if ttl_days is not None:
            cutoff = now - ttl_days * 86400
        else:
            cutoff = 0.0
        q = _tokens(query)
        if not q:
            return []
        qset = set(q)
        scored: list[tuple[float, MemoryItem]] = []
        for item in self._items:
            if item.created_at < cutoff:
                continue
            words = _tokens(item.text)
            hits = sum(1 for w in words if w in qset)
            if hits == 0:
                continue
            norm = hits / (len(words) + len(q) + 1)
            scored.append((norm, item))
        scored.sort(key=lambda pair: (-pair[0], -pair[1].created_at))
        top = scored[:top_k]
        for _, item in top:
            item.hits += 1
        # Read path: mark dirty, do not write.  This runs at the start of every
        # user turn, and a full rewrite here was an O(n) disk write per turn for
        # a counter nothing reads back.  flush() persists it at the window edge.
        if top:
            self._dirty = True
        return [item for _, item in top]

    # -- digest for the system prompt ---------------------------------------
    def notes(self, items: Iterable[MemoryItem]) -> str:
        if not items:
            return ""
        return "\n".join(f"- {it.text}" for it in items)

    # -- introspection for tests -------------------------------------------
    def all(self) -> list[MemoryItem]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


# --------------------------------------------------------------------------- #
# End-of-session extraction (store timing: window close)
# --------------------------------------------------------------------------- #
_EXTRACT_PROMPT = """You are a memory curator. You are given the tail of a chat
session. Pick out the statements that would still be true next week:

- preferences and constraints ("喜欢BA风格", "不要用英文缩写", "讨厌推销电话"),
- stable facts ("我住在北京", "我是后端实习生", "周一上午开会"),
- long-running commitments ("在准备面试", "项目下周五必须上线").

Maintain no more than {max} facts and ONLY from what the user actually said.
Output a JSON array of strings, e.g. ["我住在北京"]. If nothing qualifies,
output []. Output only the JSON."""


def extract_facts(messages: list[dict], llm: Any, *, max_facts: int = 8) -> list[str]:
    """One LLM call to pull candidate long-term facts out of a session tail."""
    tail_text = _tail_as_text(messages, limit=60)
    if not tail_text:
        return []
    request = LLMRequest(
        messages=[{"role": "user", "content": _EXTRACT_PROMPT.format(max=max_facts)}],
        max_tokens=512,
        temperature=0,
    )
    try:
        response = llm.complete(request)
    except Exception:  # noqa: BLE001 - extraction is best-effort
        return []
    return _parse_str_list(response.text)


def _tail_as_text(messages: list[dict], *, limit: int) -> str:
    out: list[str] = []
    for m in messages[-limit:]:
        content = m.get("content")
        if isinstance(content, str):
            out.append(f"[{m['role']}] {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    out.append(f"[{m['role']}] {block.get('text', '')}")
    joined = "\n".join(out)
    return joined[-8000:]


def _parse_str_list(text: str) -> list[str]:
    import json

    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(x).strip() for x in data if isinstance(x, str) and x.strip()]