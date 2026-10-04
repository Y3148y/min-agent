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
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .llm import LLMRequest
from . import textutil
from .paths import atomic_write_text

# Every durable-fact example that appears in the prompts, the README and the
# tool descriptions.  The gate must accept all of them -- the test that pins
# this lives in test_memory.py, so a tightening rule cannot silently break the
# exact examples we hand the model.
_DOCUMENTED_FACTS = (
    "我住在北京",
    "喜欢BA风格",
    "不要用英文缩写",
    "讨厌推销电话",
    "我是后端实习生",
    "周一上午开会",
    "在准备面试",
    "项目下周五必须上线",
    "讨厌被叫英文名",
)

# Statements that are almost never durable facts because they are anchored to
# *now*.  Recurring schedule words (上周/下周/本周/周X) are deliberately NOT
# here: "项目下周五必须上线" is exactly the standing commitment memory should
# keep, and it used to be rejected for containing 下周.
_NOW_TEMPORAL = ("今天", "今日", "昨天", "明天", "后天", "大后天", "刚才", "刚刚", "现在", "此刻", "今晚", "今早", "目前")
_ACTION_PREFIXES = ("帮我", "请", "记一下", "查一下", "算一下", "搜一下", "写一下", "记得", "记住", "提醒", "标记")
_QUESTION_WORDS = ("吗", "嘛", "呢", "吧", "么", "啊", "呀")
_QUESTION_LEADS = ("什么", "怎么", "怎样", "如何", "为什么", "为啥", "哪", "几")


def _meaningful_units(text: str) -> int:
    """Token budget for the length gate: CJK characters plus separate ASCII
    words.  Pure punctuation, spaces and stopwords contribute nothing, so
    ``CAFE`` and ``好的`` stay below the keep bar and ``在准备面试`` clears it."""
    cjk = sum(1 for ch in text if textutil.is_cjk(ch))
    words = sum(
        1
        for w in textutil.WORD.findall(text.lower())
        if w.isascii() and w not in textutil.STOP and len(w) > 1
    )
    return cjk + words


def _looks_like_question(text: str) -> bool:
    striped = text.strip()
    if "?" in striped or "？" in striped:
        return True
    tail = striped.rstrip("。．.!！?？，, ")
    if tail.endswith(_QUESTION_WORDS):
        return True
    return any(striped.startswith(w) for w in _QUESTION_LEADS)


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
        """Return a rejection reason, or '' to allow storage.

        Cheap and rule-driven on purpose; the real protection is *who* feeds it
        -- the ``remember`` tool plus an extraction job that now reads user
        turns only, so numeric or relational numbers in a fact are no longer
        mistaken for tool output and no numeric rule is needed.  This gate only
        keeps obviously transient utterances (filler, questions, instructions,
        now-relative timestamps) out of long-term memory.
        """
        if not any(ch.isalpha() or textutil.is_cjk(ch) for ch in text):
            return "no meaningful content"
        if _meaningful_units(text) < 3:
            return "too short"
        if _looks_like_question(text):
            return "looks like a question"
        if any(text.strip().startswith(w) for w in _ACTION_PREFIXES):
            return "looks like an instruction"
        if any(w in text for w in _NOW_TEMPORAL):
            return "time-sensitive"
        return ""

    def _find_duplicate(self, text: str) -> MemoryItem | None:
        needle = textutil.normalise(text)
        for item in self._items:
            if textutil.normalise(item.text) == needle:
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
        q = textutil.tokenize(query)
        if not q:
            return []
        qset = set(q)
        scored: list[tuple[float, MemoryItem]] = []
        for item in self._items:
            if item.created_at < cutoff:
                continue
            words = textutil.tokenize(item.text)
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

Examples: {examples}

Maintain no more than {max} facts and ONLY from what the user actually said.
Output a JSON array of strings, e.g. ["我住在北京"]. If nothing qualifies,
output []. Output only the JSON."""


def extract_facts(messages: list[dict], llm: Any, *, max_facts: int = 8) -> list[str]:
    """One LLM call to pull candidate long-term facts out of a session tail."""
    tail_text = _tail_as_text(messages, limit=60)
    if not tail_text:
        return []
    prompt = _EXTRACT_PROMPT.format(
        examples="、".join(_DOCUMENTED_FACTS), max=max_facts)
    prompt += "\n\nSession tail (user's turns only):\n" + tail_text
    request = LLMRequest(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    try:
        response = llm.complete(request)
    except Exception:  # noqa: BLE001 - extraction is best-effort
        return []
    return _parse_str_list(response.text)


def _tail_as_text(messages: list[dict], *, limit: int) -> str:
    """The curator reads the *user's* turns only.

    Facts must come from what the user actually said; assistant text (including
    an answer that says "你住在北京" only because the bot guessed it back) must
    never be transcribed into long-term memory.
    """
    out: list[str] = []
    count = 0
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            out.append(f"[user] {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    out.append(f"[user] {block.get('text', '')}")
        count += 1
        if count >= limit:
            break
    joined = "\n".join(reversed(out))
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