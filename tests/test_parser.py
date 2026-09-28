"""Tests for output parsing: thinking/text/tool_use split, max_tokens, fallback."""

from __future__ import annotations

from min_agent.parser import (
    parse_response,
    parse_single_tool_json,
)
from min_agent.tools.base import ToolCall


def _blocks(*items):
    return list(items)


def test_thinking_text_and_tool_use_are_split():
    out = parse_response(
        [
            {"type": "thinking", "thinking": "我需要先算出 2*3。"},
            {"type": "tool_use", "id": "t1", "name": "calculator", "input": {"expression": "2*3"}},
        ]
    )
    assert out.reasoning == ["我需要先算出 2*3。"]
    assert out.tool_calls == [ToolCall("t1", "calculator", {"expression": "2*3"})]
    assert out.should_act is True


def test_final_answer_from_text_block():
    out = parse_response([{"type": "text", "text": "答案是 6。"}])
    assert out.should_act is False
    assert out.final_answer == "答案是 6。"
    assert out.has_substantive_answer()


def test_empty_output_is_not_an_answer():
    out = parse_response([])
    assert out.should_act is False
    assert out.final_answer == ""
    assert not out.has_substantive_answer()


def test_max_tokens_truncation_is_flagged():
    out = parse_response(
        [{"type": "text", "text": "答案是"}],
        stop_reason="max_tokens",
    )
    assert out.truncated is True


def test_sdk_object_shapes_are_tolerated():
    class _B:
        def __init__(self, d):
            self.__d = d

        def __getattr__(self, item):
            return self.__d.get(item)

    out = parse_response(
        [
            _B({"type": "thinking", "thinking": "hesitate"}),
            _B({"type": "tool_use", "id": "x", "name": "search", "input": {"query": "ai"}}),
            _B({"type": "text", "text": "查到了"}),
        ]
    )
    assert out.reasoning == ["hesitate"]
    assert len(out.tool_calls) == 1
    assert out.texts == ["查到了"]


def test_unknown_block_types_are_ignored():
    out = parse_response(
        [{"type": "redacted_thinking", "text": "secret"}, {"type": "text", "text": "hi"}]
    )
    assert out.ignored == ["redacted_thinking"]
    assert out.final_answer == "hi"


def test_parallel_tool_calls_parsed_in_order():
    out = parse_response(
        [
            {"type": "tool_use", "id": "a", "name": "calculator", "input": {"expression": "2+2"}},
            {"type": "tool_use", "id": "b", "name": "calculator", "input": {"expression": "7*6"}},
        ]
    )
    assert [c.name for c in out.tool_calls] == ["calculator", "calculator"]
    assert [c.args["expression"] for c in out.tool_calls] == ["2+2", "7*6"]


# --------------------------------------------------------------------------- #
# text fallback (provider returns text instead of native tool_use)
# --------------------------------------------------------------------------- #


def test_text_fallback_extracts_tool_call_envelope():
    out = parse_response(
        [{"type": "text", "text": "先算一下。\n<tool_call>{\"name\": \"calculator\", \"input\": {\"expression\": \"9*9\"}}</tool_call>"}]
    )
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "calculator"
    assert out.tool_calls[0].args == {"expression": "9*9"}


def test_text_fallback_json_fence_and_string_args():
    out = parse_response(
        [{"type": "text", "text": '```json\n{"tool": "weather", "arguments": "{\\"city\\": \\"Beijing\\"}"}\n```'}]
    )
    assert out.tool_calls[0].name == "weather"
    assert out.tool_calls[0].args == {"city": "Beijing"}


def test_text_without_call_stays_final_answer():
    out = parse_response([{"type": "text", "text": "无需工具，直接回答：是的。"}])
    assert out.tool_calls == []
    assert out.final_answer == "无需工具，直接回答：是的。"


def test_parse_single_tool_json_direct():
    call = parse_single_tool_json('{"name": "search", "input": {"query": "记忆系统"}}')
    assert call is not None
    assert call.name == "search"


def test_dedupe_recovered_tool_calls():
    out = parse_response(
        [
            {"type": "tool_use", "id": "n", "name": "calculator", "input": {"expression": "1+1"}},
            {"type": "text", "text": '<tool_call>{"name": "calculator", "input": {"expression": "1+1"}}</tool_call>'},
        ]
    )
    assert len(out.tool_calls) == 1  # the recovered one is a duplicate