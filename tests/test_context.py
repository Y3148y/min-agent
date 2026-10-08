"""Context management: memory placement in the system prompt, compaction."""

from __future__ import annotations

import pytest

from min_agent.context import (
    LLMCompactor,
    build_system_prompt,
    compact_messages,
    _partition,
    _mechanical_eviction,
)
from min_agent.llm import estimate_message_tokens
from min_agent.session import Session, SessionMeta

from fake_llm import ScriptedLLM, text_block


# --------------------------------------------------------------------------- #
# system prompt: placement is part of the design
# --------------------------------------------------------------------------- #


def test_recalled_memory_goes_into_system_not_messages():
    prompt = build_system_prompt(
        user="alice",
        session_id="w1",
        memory_notes="- 用户住在北京\n- 喜欢简洁的答案",
        todo_digest="- 买牛奶",
        summary="聊了天气",
    )
    assert "Long-term memory about this user (recalled)" in prompt
    assert "住在北京" in prompt
    assert "买牛奶" in prompt and "聊了天气" in prompt
    assert "Today's todo list" in prompt


def test_no_memory_no_section():
    prompt = build_system_prompt(user="bob", session_id="w2")
    assert "Long-term memory" not in prompt
    assert "todo list" in prompt


# --------------------------------------------------------------------------- #
# compaction helpers
# --------------------------------------------------------------------------- #


def _mk_session(messages, tmp_path):
    s = Session.create("c", "u", tmp_path / "sess")
    s.messages = list(messages)
    return s


def _tool_turn(tool_id, expr, result="4"):
    return (
        ("assistant", [{"type": "tool_use", "id": tool_id, "name": "calculator", "input": {"expression": expr}}]),
        ("user", [{"type": "tool_result", "tool_use_id": tool_id, "content": result}]),
    )


def _with_tools(n=3):
    msgs = [{"role": "user", "content": "anchor question"}]
    for i in range(n):
        tool_turn = _tool_turn(f"t{i}", f"{i}+1")
        msgs.append({"role": "assistant", "content": tool_turn[0][1]})
        msgs.append({"role": "user", "content": tool_turn[1][1]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": f"answer {i}"}]})
    return msgs


def _alternating(messages) -> bool:
    roles = [m["role"] for m in messages]
    return all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))


def _pairs_intact(messages) -> bool:
    """Every tool_result in the transcript has its tool_use in the same list."""
    uses = {b.get("id") for m in messages for b in m["content"] if isinstance(m["content"], list) and b.get("type") == "tool_use"}
    results = {b.get("tool_use_id") for m in messages for b in m["content"] if isinstance(m["content"], list) and b.get("type") == "tool_result"}
    return results <= uses


def test_partition_never_starts_tail_inside_pair():
    messages = _with_tools(3)
    anchor, tail = _partition(messages, keep_recent=3)
    assert anchor["content"] == "anchor question"
    assert tail[0]["role"] == "assistant"  # never opens inside a pair
    assert _alternating(tail) and _pairs_intact(tail)


def test_estimate_no_overflow_within_budget():
    messages = _with_tools(2)
    assert estimate_message_tokens(messages) > 0
    assert estimate_message_tokens(messages) + 100 < 10_000_000


# --------------------------------------------------------------------------- #
# compaction: summariser path (the happy route)
# --------------------------------------------------------------------------- #


def test_compact_with_summary_keeps_anchor_and_tail(tmp_path):
    messages = _with_tools(3)
    session = _mk_session(messages, tmp_path)
    compactor = LLMCompactor(
        ScriptedLLM([{"content": [text_block("一致同意的摘要")], "stop_reason": "end_turn"}])
    )
    budget = 1  # forces compaction of everything except the protected tail
    ok = compact_messages(
        session,
        budget=budget,
        keep_recent=2,
        summary_max=200,
        compactor=compactor,
        system_overhead=100,
    )
    assert ok is True
    result = session.messages
    # anchor user message is preserved and now carries the summary
    assert result[0]["role"] == "user"
    assert result[0]["content"].startswith("anchor question")
    assert "一致同意的摘要" in result[0]["content"]
    # tail is verbatim: tool pair kept together, alternation intact
    assert _alternating(result)
    assert _pairs_intact(result)
    # running summary was persisted on the session
    assert "一致同意的摘要" in session.meta.summary


def test_compact_noop_when_under_budget(tmp_path):
    session = _mk_session(_with_tools(1), tmp_path)
    assert (
        compact_messages(
            session,
            budget=10_000_000,
            keep_recent=4,
            summary_max=200,
            compactor=None,
            system_overhead=100,
        )
        is False
    )


# --------------------------------------------------------------------------- #
# compaction: mechanical fallback (summariser unavailable / overflow)
# --------------------------------------------------------------------------- #


def test_mechanical_eviction_never_splits_a_pair_and_keeps_current_turn(tmp_path):
    messages = _with_tools(3)
    session = _mk_session(messages, tmp_path)
    ok = compact_messages(
        session,
        budget=1,
        keep_recent=2,
        summary_max=200,
        compactor=None,  # summariser unavailable -> mechanical fallback
        system_overhead=100,
    )
    assert ok is True
    result = session.messages
    # under maximum pressure eviction drops all but the anchor user message --
    # but it never leaves a half-message or splits a tool_use/tool_result pair
    assert len(result) < len(messages)
    assert result[0]["role"] == "user"
    assert _alternating(result)
    assert _pairs_intact(result)


def test_summariser_exception_falls_back_to_eviction(tmp_path):
    messages = _with_tools(3)
    session = _mk_session(messages, tmp_path)

    class Boom:
        def summarize(self, text, max_chars):
            raise RuntimeError("summariser died")

    ok = compact_messages(
        session,
        budget=1,
        keep_recent=2,
        summary_max=200,
        compactor=Boom(),
        system_overhead=100,
    )
    assert ok is True
    assert _alternating(session.messages)
    assert _pairs_intact(session.messages)


def test_direct_eviction_keeps_tool_pairs_together():
    messages = _with_tools(3)
    saved = list(messages)
    changed = _mechanical_eviction(messages, budget=1, overhead=100)
    assert changed is True
    assert _alternating(messages)
    assert _pairs_intact(messages)
    assert len(messages) < len(saved)


def test_mechanical_eviction_drops_the_mirror_pair():
    """A tool_results whose tool_use sits just above it must be dropped together
    -- otherwise the eviction leaves an orphaned tool_use that the provider
    rejects."""
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "calculator", "input": {"expression": "1+1"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "2"}],
        },
    ]
    changed = _mechanical_eviction(messages, budget=1, overhead=100)
    assert changed is True
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_mechanical_eviction_note_reaches_the_trace(tmp_path):
    """Regression: _persist_compacted took a ``note`` it never used, so a
    mechanical eviction was invisible in the trace.  The note must surface as a
    ``compact``/``mechanical`` event."""
    from min_agent.trace import Tracer

    messages = _with_tools(3)
    session = _mk_session(messages, tmp_path)
    with Tracer("w-ctx", traces_root=tmp_path / "traces", console=False) as trace:
        ok = compact_messages(
            session,
            budget=1,
            keep_recent=2,
            summary_max=200,
            compactor=None,
            system_overhead=100,
            trace=trace,
        )
    assert ok is True
    compact_events = [ev for ev in trace.events if ev.kind == "compact"]
    assert any(ev.data.get("mechanical") is True for ev in compact_events)
    assert any(ev.data.get("note") == "evicted oldest complete turns" for ev in compact_events)