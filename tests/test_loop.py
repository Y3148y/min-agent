"""The four steps of the loop, driven by a scripted model (no network)."""

from __future__ import annotations

import pytest

from min_agent.errors import LLMContextOverflow
from min_agent.llm import LLMRequest
from min_agent.parser import parse_response
from fake_llm import ScriptedLLM, text_block, think_block, tool_block


# --------------------------------------------------------------------------- #
# Step 2/4: replvs act
# --------------------------------------------------------------------------- #


def test_pure_chat_single_turn(agent, llm):
    llm.script = [
        {"content": [think_block("直接回答即可"), text_block("你好呀")], "stop_reason": "end_turn"}
    ]
    result = agent.run_turn("你好")
    assert result.text == "你好呀"
    assert result.tool_calls == 0
    # thinking is traced but not persisted into the transcript
    assert agent.trace.last("reasoning") is not None
    assistant = agent.session.messages[-1]
    assert all(b.get("type") != "thinking" for b in assistant["content"])


def test_multi_step_tool_then_answer(agent, llm):
    llm.script = [
        {
            "content": [
                think_block("用户要算乘法"),
                tool_block("t1", "calculator", expression="37*4"),
            ],
            "stop_reason": "tool_use",
        },
        {
            "content": [text_block("37 乘 4 等于 148。")],
            "stop_reason": "end_turn",
        },
    ]
    result = agent.run_turn("37*4 是多少？")
    assert result.tool_calls == 1
    assert "148" in result.text
    # the transcript holds assistant(tool_use) -> user(tool_result) -> assistant(text)
    roles = [m["role"] for m in agent.session.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    results = agent.session.messages[2]
    assert results["content"][0]["type"] == "tool_result"
    assert "148" in results["content"][0]["content"]


def test_parallel_tool_calls_all_executed(agent, llm):
    llm.script = [
        {
            "content": [
                tool_block("a", "calculator", expression="2+2"),
                tool_block("b", "calculator", expression="7*6"),
            ],
            "stop_reason": "tool_use",
        },
        {"content": [text_block("2+2=4，7*6=42。")], "stop_reason": "end_turn"},
    ]
    result = agent.run_turn("算两个式子")
    assert result.tool_calls == 2
    assert len(agent.session.messages[2]["content"]) == 2


# --------------------------------------------------------------------------- #
# Follow-up questions
# --------------------------------------------------------------------------- #


def test_follow_up_sees_previous_tool_result(agent, llm):
    """纯对话追问: the second user input must reach the model with the full
    first turn (tool_use + tool_result + final answer) already in context."""
    llm.script = [
        {
            "content": [
                tool_block("t1", "calculator", expression="128*2"),
                think_block("算出来是256"),
            ],
            "stop_reason": "tool_use",
        },
        {"content": [text_block("128*2=256。")], "stop_reason": "end_turn"},
        # second user turn '那再加 100 呢' -- model reads context, answers
        {"content": [text_block("256+100=356。")], "stop_reason": "end_turn"},
    ]
    first = agent.run_turn("帮我算 128*2")
    assert first.tool_calls == 1

    second = agent.run_turn("那再加 100 呢？")
    assert "356" in second.text
    assert second.tool_calls == 0

    # the last request carried the earlier turn verbatim
    last_req = llm.last_request()
    flat = "\n".join(str(m.get("content")) for m in last_req.messages)
    assert "tool_use" in flat and "tool_result" in flat


def test_follow_up_triggers_another_tool(agent, llm):
    """带着工具的追问: a follow-up may legitimately call the tool again."""
    llm.script = [
        {
            "content": [tool_block("t1", "calculator", expression="10*3")],
            "stop_reason": "tool_use",
        },
        {"content": [text_block("10*3=30。")], "stop_reason": "end_turn"},
        {
            "content": [tool_block("t2", "calculator", expression="30*2")],
            "stop_reason": "tool_use",
        },
        {"content": [text_block("30*2=60。")], "stop_reason": "end_turn"},
    ]
    agent.run_turn("10*3 是多少")
    result = agent.run_turn("再乘以 2 呢？")
    assert result.tool_calls == 1
    assert "60" in result.text


# --------------------------------------------------------------------------- #
# guard rails
# --------------------------------------------------------------------------- #


def test_max_turns_forces_wrap_up(agent, llm, cfg):
    # The model keeps wanting to call a tool; MAX_TURNS is small (cfg=6).
    llm.script = [
        {"content": [tool_block(f"t{i}", "calculator", expression=f"{i}+1")], "stop_reason": "tool_use"}
        for i in range(1, 7)
    ] + [
        {"content": [text_block("我用到上限前就停下来总结。")], "stop_reason": "end_turn"},
    ]
    result = agent.run_turn("一直算下去")
    assert result.turns == cfg.max_turns
    assert result.turn_budget_exhausted is True
    assert "总结" in result.text


def test_repeat_call_guard_aborts(agent, llm):
    # 3 identical signatures max; the 4th attempt must abort into wrap-up.
    llm.script = [
        {"content": [tool_block("t1", "calculator", expression="1+1")], "stop_reason": "tool_use"},
        {"content": [tool_block("t2", "calculator", expression="1+1")], "stop_reason": "tool_use"},
        {"content": [tool_block("t3", "calculator", expression="1+1")], "stop_reason": "tool_use"},
        {"content": [text_block("同一个工具调用已在反复循环，我停止。")], "stop_reason": "end_turn"},
    ]
    result = agent.run_turn("循环测试")
    assert result.guard_aborted is True
    assert "循环" in result.text


def test_max_tokens_truncation_triggers_continuation(agent, llm):
    llm.script = [
        {"content": [think_block("先想"), text_block("最后的")], "stop_reason": "max_tokens"},
        {"content": [text_block("答案是完整的这个。")], "stop_reason": "end_turn"},
    ]
    result = agent.run_turn("上下文截断测试")
    assert result.text == "答案是完整的这个。"


# --------------------------------------------------------------------------- #
# tool failure is data, not a crash
# --------------------------------------------------------------------------- #


def test_tool_error_feeds_back_to_model(agent, llm):
    llm.script = [
        {"content": [tool_block("t1", "calculator", expression="1/0")], "stop_reason": "tool_use"},
        {"content": [text_block("除数为零，换个说法。")], "stop_reason": "end_turn"},
    ]
    agent.registry.unregister("todo")
    result = agent.run_turn("算一下 1/0")
    assert "除数为零" in result.text
    results_msg = agent.session.messages[2]
    assert results_msg["content"][0]["is_error"] is True


def test_context_overflow_compacts_and_retries(agent, llm, cfg):
    """LLMContextOverflow mid-turn -> force compaction -> retry succeeds."""
    cfg.context_budget = 100  # tiny budget forces compaction of anything
    agent.compactor = None    # mechanical compaction only, so it can't eat scenes

    llm.script = [
        LLMContextOverflow("prompt is too long"),
        {"content": [text_block("压缩后正常回答。")], "stop_reason": "end_turn"},
    ]
    result = agent.run_turn("会触发上下文溢出的请求")
    assert result.text == "压缩后正常回答。"
    assert len(llm.requests) == 2  # first call overflowed, retry succeeded


# --------------------------------------------------------------------------- #
# session keeps state across user turns
# --------------------------------------------------------------------------- #


def test_session_message_growth_across_turns(agent, llm):
    llm.script = [
        {"content": [text_block("第一问答复。")], "stop_reason": "end_turn"},
        {"content": [text_block("第二问答复。")], "stop_reason": "end_turn"},
    ]
    agent.run_turn("第一问")
    n1 = len(agent.session.messages)
    agent.run_turn("第二问")
    n2 = len(agent.session.messages)
    assert n2 > n1


def test_llm_request_shapes_are_api_compatible(agent, llm):
    """The messages we send must satisfy Anthropic's alternation rule."""
    llm.script = [
        {"content": [tool_block("a", "search", query="x")], "stop_reason": "tool_use"},
        {"content": [text_block("done")], "stop_reason": "end_turn"},
        {"content": [text_block("again done")], "stop_reason": "end_turn"},
    ]
    agent.run_turn("查 x")
    agent.run_turn("再查 y")

    from min_agent.session import _drop_hanging_tool_refs

    saved = _drop_hanging_tool_refs(agent.session.messages)
    assert saved == agent.session.messages  # nothing to repair
    roles = [m["role"] for m in saved]
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))


# --------------------------------------------------------------------------- #
# memory TTL is actually wired to the store
# --------------------------------------------------------------------------- #


def test_expired_memories_are_not_recalled(agent, llm, cfg):
    """Regression: the loop called recall() without ttl_days.

    `MemoryStore.recall` treats `ttl_days=None` as "no cutoff", so the
    configured MEMORY_TTL_DAYS (and the README's "TTL 90 天") never applied --
    nothing could ever expire.  The store's own TTL is covered in
    test_memory.py; what was untested is that anybody passed the argument.
    """
    import time

    agent.memory.remember("用户常驻厦门", "w-test")
    agent.memory.remember("用户喜欢跑步", "w-test")

    now = time.time()
    for item in agent.memory.all():
        # one is fresh, one is older than the 1-day budget below
        if "厦门" in item.text:
            item.created_at = now - 5 * 86400
    agent.memory.save()

    cfg.memory_ttl_days = 1
    llm.script = [{"content": [text_block("知道了。")], "stop_reason": "end_turn"}]
    agent.run_turn("用户住哪里，喜欢什么")

    hits = [h for ev in agent.trace.events if ev.kind == "memory_recall" for h in ev.data["hits"]]
    assert hits, "expected the fresh memory to still be recalled"
    assert not any("厦门" in h for h in hits), "expired memory was recalled"

    system = llm.requests[0].system
    assert "跑步" in system
    assert "厦门" not in system
