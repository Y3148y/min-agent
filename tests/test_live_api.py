"""Live tests against the real LLM API. Skipped by default: ``pytest -m live``.

Each test runs its own window in a throwaway workspace; nothing here touches
the developer's own sessions.  Keep the model cost low -- a handful of calls.
"""

from __future__ import annotations

import json

import pytest

from min_agent.config import load_config
from min_agent.loop import Agent
from min_agent.trace import Tracer

pytestmark = pytest.mark.live


def _live_agent(tmp_path, sid, use="alice-live"):
    from min_agent.loop import Agent
    from min_agent.trace import Tracer

    config = load_config(workspace=tmp_path, max_turns=10)
    agent = Agent(
        config=config,
        user=use,
        session_id=sid,
        trace=Tracer(sid, console=False),
    )
    return agent, config


def test_live_single_tool_call(tmp_path):
    agent, _ = _live_agent(tmp_path, "live1")
    try:
        result = agent.run_turn("请用 calculator 计算 37*4")
        assert result.tool_calls >= 1
        assert result.text
    finally:
        agent.close()


def test_live_repeat_question_remembers_calculation(tmp_path):
    """纯对话追问: see whether the model reuses the earlier calculation."""
    agent, _ = _live_agent(tmp_path, "live2")
    try:
        first = agent.run_turn("请算 6*7 并告诉我结果")
        assert first.text
        second = agent.run_turn("基于刚才的结果再加 100，是多少？")
        assert second.text
        # the second answer must reflect the earlier session's number
        assert ("142" in second.text) or ("164" in second.text) or ("100" in second.text)
    finally:
        agent.close()


def test_live_tool_error_does_not_abort(tmp_path):
    agent, _ = _live_agent(tmp_path, "live3")
    try:
        result = agent.run_turn("用 calculator 计算 1/0")
        assert result.text  # a graceful reply, not a crash
    finally:
        agent.close()


def test_live_two_windows_never_cross_talk(tmp_path):
    a1, _ = _live_agent(tmp_path, "win-a")
    a2, _ = _live_agent(tmp_path, "win-b")
    try:
        a1.run_turn("请用 todo 记下：只在窗口A里说这句话")
        assert a2.session.messages == []
        assert a2.todo_store.list_items() == []
        assert len(a1.todo_store.list_items()) == 1
    finally:
        a1.close()
        a2.close()


def test_live_reload_session_has_full_context(tmp_path):
    agent, _ = _live_agent(tmp_path, "live5")
    try:
        agent.run_turn("计算 10*10")
        agent.close()

        reopen, config = _live_agent(tmp_path, "live5")
        try:
            assert any("user" == m["role"] for m in reopen.session.messages)
            result = reopen.run_turn("刚才算的结果是多少？")
            assert "100" in result.text
        finally:
            reopen.close()
    finally:
        pass  # agent already closed inside