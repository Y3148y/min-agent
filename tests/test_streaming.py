"""Streaming: deltas reach the tracer live, the loop still assembles ``LLMResponse``.

The scripted model has no ``stream`` surface, so these tests wrap it: the
wrapper's ``stream()`` replays the same scripted response in small ``text``
deltas, which is exactly what decides whether the loop *and* the console
renderer are genuinely incremental.
"""

from __future__ import annotations

import io

import pytest

from min_agent.llm import LLMRequest, LLMResponse, StreamEvent
from min_agent.trace import Tracer
from fake_llm import ScriptedLLM, text_block, think_block


def _chunks(text: str, size: int = 3):
    for i in range(0, len(text), size):
        yield text[i : i + size]


class StreamScriptedLLM(ScriptedLLM):
    """ScriptedLLM plus a ``stream`` that replays the response in deltas."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.stream_calls = 0

    def stream(self, request: LLMRequest):
        self.stream_calls += 1
        response = self.complete(request)
        for b in response.content:
            kind = b.get("type")
            if kind == "text":
                for piece in _chunks(b.get("text", "")):
                    yield StreamEvent("text", delta=piece)
            elif kind == "thinking":
                yield StreamEvent("thinking", delta=b.get("thinking", ""))
        yield StreamEvent(
            "done",
            response=LLMResponse(
                content=response.content,
                stop_reason=response.stop_reason,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            ),
        )


def _tiny_agent(cfg, streamed: bool):
    from min_agent.loop import Agent

    trace = Tracer("w-stream", console=False)
    if streamed:
        llm = StreamScriptedLLM()
    else:
        llm = ScriptedLLM()
    instance = Agent(config=cfg, user="alice", session_id="w-stream", llm=llm, trace=trace)
    return instance


def test_loop_uses_stream_when_available(cfg):
    agent = _tiny_agent(cfg, streamed=True)
    scripted = agent.llm
    scripted.script = [
        {
            "content": [
                think_block("直接答即可"),
                text_block("流式输出测试，回答完毕。"),
            ],
            "stop_reason": "end_turn",
        }
    ]
    result = agent.run_turn("你好")
    assert scripted.stream_calls == 1
    assert result.text == "流式输出测试，回答完毕。"
    # the durable record still holds the whole final message, not the deltas
    assert agent.trace.last("final").data["text"] == "流式输出测试，回答完毕。"
    agent.close()


def test_loop_keeps_blocking_path_without_stream_surface(cfg):
    agent = _tiny_agent(cfg, streamed=False)
    agent.llm.script = [{"content": [text_block("阻塞路径")], "stop_reason": "end_turn"}]
    result = agent.run_turn("你好")
    assert result.text == "阻塞路径"
    agent.close()


def test_streamed_tool_turn_still_round_trips(cfg):
    agent = _tiny_agent(cfg, streamed=True)
    scripted = agent.llm
    scripted.script = [
        {
            "content": [
                think_block("要用工具"),
                {"type": "tool_use", "id": "t1", "name": "calculator", "input": {"expression": "6*7"}},
            ],
            "stop_reason": "tool_use",
        },
        {"content": [text_block("6 乘 7 等于 42。")], "stop_reason": "end_turn"},
    ]
    result = agent.run_turn("6*7?")
    assert result.text == "6 乘 7 等于 42。"
    roles = [m["role"] for m in agent.session.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    agent.close()


def test_console_streams_one_growing_line_and_closes_once():
    buf = io.StringIO()
    tracer = Tracer("w-live", console=True, stream=buf)
    tracer.stream("text", "你")
    tracer.stream("text", "好")
    tracer.stream("text", "世界\n第二行")
    tracer.emit(
        "final",
        text="你好世界\n第二行",
        turn=1,
    )
    out = buf.getvalue()
    # the live line opens once with the `agent` label and grows via \r...
    assert "agent 你" in out
    assert "agent 你好" in out
    assert "agent 你好世界" in out
    # ...complete lines close with a newline and rewrite from column 0
    assert "agent 你好世界\n" in out
    assert "第二行\n" in out
    # the final event does NOT reprint the whole answer (close only adds a
    # trailing newline to the pending partial + a blank separator)
    assert out.count("你好世界") == 1
    assert out.count("第二行") == 1
    # next turn opens a fresh line
    tracer.stream("text", "A")
    assert "agent A" in buf.getvalue()


def test_stream_ignored_when_console_off(cfg):
    agent = _tiny_agent(cfg, streamed=True)
    agent.llm.script = [{"content": [text_block("不显示")], "stop_reason": "end_turn"}]
    result = agent.run_turn("x")
    assert result.text == "不显示"  # no crash with console=False
    agent.close()


@pytest.mark.live
def test_live_stream_smoke():
    from pathlib import Path
    from tempfile import mkdtemp

    from min_agent.config import load_config
    from min_agent.loop import Agent
    from min_agent.trace import Tracer

    cfg = load_config(workspace=Path(mkdtemp()), max_turns=4)
    agent = Agent(
        config=cfg,
        user="live",
        session_id="stream-smoke",
        session=__import__("min_agent.store", fromlist=["SessionStore"]).SessionStore(
            cfg.sessions_root
        ).open("live", "stream-smoke", create=True),
        trace=Tracer("stream-smoke", console=False),
    )
    assert hasattr(agent.llm, "stream"), "AnthropicLLM must expose stream()"
    result = agent.run_turn("你好，用一句话介绍你自己")
    assert result.text
    agent.patrol()
    agent.close()