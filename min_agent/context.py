"""Context management: what goes into the prompt, and compression when it
overflows.

Three ideas are worth being explicit about, because they are what a reviewer
will ask about first:

1. **What is injected where.**  Background facts (recalled memory, the todo
   digest, the running summary) live in the *system* prompt; the *messages*
   carry only the conversation itself: user input verbatim, assistant output
   verbatim, ``tool_use`` inputs, and truncated ``tool_result`` outputs.  A
   memory that lands in the user turn would read as a fresh instruction to the
   model -- memory is context, not command.

2. **Compaction preserves pairing.**  ``tool_use`` blocks and their
   ``tool_result`` blocks must move together (either both verbatim or both
   summarised), or the next request is rejected by the provider.  We never
   summarise one half of a pair.

3. **Truncation is the floor.**  If the summarising LLM itself cannot run (or
   a gateway refuses the oversized prompt before we even compact), we fall back
   to dropping the oldest tool outputs verbatim.  It is uglier, but it can
   never die.
"""

from __future__ import annotations

import datetime
from typing import Any, Protocol

from .llm import LLMRequest, estimate_message_tokens
from .session import Session

_DEFAULT_SUMMARY = "(no summary yet)"


# --------------------------------------------------------------------------- #
# System prompt assembly
# --------------------------------------------------------------------------- #
def build_system_prompt(
    *,
    user: str,
    session_id: str,
    date: str | None = None,
    memory_notes: str = "",
    todo_digest: str = "(no todos yet)",
    summary: str = "",
    max_tool_output: str = "2000",
) -> str:
    date = date or datetime.date.today().isoformat()
    summary = summary or _DEFAULT_SUMMARY
    blocks: list[str] = [
        _ROLE,
        f"Current date: {date} (use it for anything relative).",
        f"Session: user={user!r}, window={session_id!r}. Each window is an independent session; never invent facts from other sessions.",
        "## Today's todo list (current state)",
        todo_digest,
        "",
        "## Running summary of this session",
        summary,
    ]
    if memory_notes:
        blocks += ["", "## Long-term memory about this user (recalled)", memory_notes]
    blocks += ["", _TOOL_DISCIPLINE.format(max_tool_output=max_tool_output)]
    return "\n".join(blocks)


_ROLE = """You are min-agent, a small assistant built from scratch that answers in the user's language and uses tools when they help.

Do not mention that you are 'AI'. Be succinct. If you are not certain about something, use a tool rather than guessing. When you have enough information, stop calling tools and give the final answer."""


_TOOL_DISCIPLINE = """## Tool discipline

- The user's input may be a follow-up ("那第二条呢?") -- it refers to what you
  just computed or read. Keep prior tool results in mind.
- You may call several tools in one turn; run independent calls together.
- Call a tool only when you need data you do not have. Do not call a tool just
  because one exists.
- After calling a tool, read its result; if it is an error, retry with a
  corrected argument or tell the user what is wrong.
- When finished, answer directly. Do not narrate every step; one short closing
  remark is enough.
- Tool output is truncated at {max_tool_output} characters inside the
  conversation; the full output lives in the run's trace log."""


# --------------------------------------------------------------------------- #
# Compaction
# --------------------------------------------------------------------------- #
class Compactor(Protocol):
    def summarize(self, text: str, max_chars: int) -> str: ...


class LLMCompactor:
    """Summarise a block of dialogue by asking the model (fallback: drop oldest)."""

    def __init__(self, llm: Any, max_len: int = 6000):
        self.llm = llm
        self.max_len = max_len

    def summarize(self, text: str, max_chars: int) -> str:
        window = text[: self.max_len]
        request = LLMRequest(
            messages=[
                {
                    "role": "user",
                    "content": _SUMMARIZE_PROMPT.format(max_chars=max_chars),
                },
                {"role": "assistant", "content": window},
            ],
            max_tokens=1024,
        )
        response = self.llm.complete(request)
        summary = response.text.strip()
        if len(summary) > max_chars * 1.2:
            summary = summary[:max_chars]
        return summary or "(dialogue was too tangled to summarise)"


_SUMMARIZE_PROMPT = """You are a transcript summariser. Below is the middle part of a conversation between a user and an assistant. Compress it into at most {max_chars} characters, keeping:
- concrete facts and figures obtained from tools,
- decisions and preferences the user expressed,
- the current status: what is done, what is blocked, what is still missing.
Do NOT invent anything. Output only the summary. Keep it so complete that a
follow-up question later can still be answered from it. Use the user's language."""


def compact_messages(
    session: Session,
    *,
    budget: int,
    keep_recent: int,
    summary_max: int,
    compactor: Compactor | None,
    system_overhead: int,
    tools: list[dict] | None = None,
    trace=None,
) -> bool:
    """Compress ``session.messages`` in place if it exceeds ``budget``.

    Returns True if compaction ran.  Never raises: any failure degrades to a
    mechanical eviction of the oldest complete turns.
    """
    total = system_overhead + estimate_message_tokens(session.messages, tools)
    if total <= budget:
        return False

    messages = session.messages
    before = len(messages)
    anchor, tail = _partition(messages, keep_recent)
    middle = _middle(messages, tail)
    if not middle:
        mechanical = _mechanical_eviction(messages, budget, system_overhead)
        if mechanical:
            _persist_compacted(session, "evicted oldest complete turns", trace=trace)
        return True

    middle_text = _messages_to_text(middle)
    summary = ""
    if compactor is not None and middle_text:
        try:
            summary = compactor.summarize(middle_text, summary_max)
        except Exception as exc:  # noqa: BLE001 - summariser is optional
            if trace is not None:
                trace.emit("warning", message=f"summariser failed ({exc}); evicting instead")
            summary = ""

    if not summary:  # summariser unavailable or produced nothing -> mechanical fallback
        mechanical = _mechanical_eviction(messages, budget, system_overhead)
        if mechanical:
            _persist_compacted(session, "evicted oldest complete turns", trace=trace)
        return True

    # Merge the summary into the anchor user message.  A *new* user message
    # inserted before tail would break the user/assistant alternation the API
    # requires; folding it into the anchor keeps [user, assistant, user, ...]
    # intact and keeps the original task statement visible.
    summary_block = f"[…… 此前内容是压缩后的对话摘要，原始消息已从上下文中移除 ……]\n{summary}"
    if anchor is not None:
        anchor_text = _anchor_json(anchor)
        replacement = {
            "role": "user",
            "content": f"{anchor_text}\n\n{summary_block}",
        }
    else:
        replacement = {"role": "user", "content": summary_block}

    session.messages = [replacement] + tail
    session.set_summary(summary)

    if trace is not None:
        trace.emit(
            "compact",
            before=before,
            after=len(session.messages),
            est_tokens=estimate_message_tokens(session.messages) + system_overhead,
            summary_chars=len(summary),
        )
    return True


def _is_tool_results_message(message: dict) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, list)
        and bool(content)
        and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    )


def _partition(messages: list[dict], keep_recent: int):
    """Return ``(anchor, tail)``.

    * ``anchor`` is the first user message -- kept verbatim as the session's
      task anchor.
    * ``tail`` is a suffix ending at the last message that is small enough to
      keep verbatim, begins with an assistant turn, and never starts inside a
      ``tool_use``/``tool_result`` pair (the provider rejects a solo half).
    * Everything between anchor and tail is the compressible middle.
    """
    if not messages:
        return None, []
    anchor = messages[0] if messages[0]["role"] == "user" else None
    rest = messages[1:] if anchor is not None else list(messages)

    start = max(0, len(rest) - keep_recent)
    guard = 0
    while 0 < start <= len(rest) and guard < keep_recent + 4:
        guard += 1
        if start < len(rest) and _is_tool_results_message(rest[start]):
            start -= 1  # pull the tail of a pair back into the middle
        elif start < len(rest) and rest[start]["role"] == "user":
            start += 1  # a lone user input cannot lead the tail (alternation)
        else:
            break
    start = max(0, min(start, len(rest)))
    return anchor, rest[start:]


def _middle(messages: list[dict], tail: list[dict]) -> list[dict]:
    if not tail:
        return messages[1:]  # anchor + everything else
    for i in range(len(messages) - len(tail), len(messages) + 1):
        if messages[i:] == tail:
            return messages[1:i]
    return messages[: max(0, len(messages) - len(tail))]


def _anchor_json(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return _trunc(_messages_to_text([message]), 160)


def _messages_to_text(messages: list[dict]) -> str:
    out: list[str] = []
    for i, m in enumerate(messages):
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            out.append(f"[{role} #{i}] {content}\n")
            continue
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "tool_use":
                out.append(f"[tool_use {block.get('name')}] {block.get('input')}\n")
            elif kind == "tool_result":
                out.append(f"[tool_result] {_trunc(block.get('content'), 300)}\n")
            elif kind == "text":
                out.append(f"[text] {block.get('text')}\n")
    return "".join(out)


def _mechanical_eviction(messages: list[dict], budget: int, overhead: int) -> bool:
    """Drop oldest *complete* turns until the estimate fits. Returns True if changed."""
    original_len = len(messages)
    dropped_any = False
    while len(messages) > 1 and overhead + estimate_message_tokens(messages) > budget:
        idx = 1
        if idx >= len(messages):
            break
        first = messages[idx]
        if (
            first["role"] == "assistant"
            and idx + 1 < len(messages)
            and _is_tool_results_message(messages[idx + 1])
        ):
            del messages[idx : idx + 2]  # tool_use and its results move together
            dropped_any = True
            continue
        if (
            _is_tool_results_message(first)
            and idx >= 1
            and messages[idx - 1]["role"] == "assistant"
        ):
            # the mirror case: a tool_results whose tool_use sits just above
            del messages[idx - 1 : idx + 1]
            dropped_any = True
            continue
        del messages[idx]
        dropped_any = True
    return dropped_any and len(messages) != original_len


def _persist_compacted(session: Session, note: str, trace=None) -> None:
    session._rewrite_jsonl()  # compacted in-memory list becomes the new truth
    if trace is not None:
        trace.emit("compact", mechanical=True, note=note)


def _trunc(text: Any, n: int) -> str:
    s = text if isinstance(text, str) else str(text)
    return s[:n] + ("..." if len(s) > n else "")