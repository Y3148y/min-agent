"""Shared fixtures: an isolated workspace + a scripted model per test."""

from __future__ import annotations

import pytest

from min_agent.config import Config
from min_agent.store import SessionStore
from min_agent.tools import TodoStore
from min_agent.trace import Tracer

from fake_llm import ScriptedLLM


def pytest_collection_modifyitems(config, items):
    """Offline runs: drop ``@pytest.mark.live`` tests unless ``-m live`` is set.

    Keeps plain ``pytest`` from spending money/compute on the real API.
    """
    if config.getoption("-m"):
        return
    items[:] = [item for item in items if "live" not in item.keywords]


@pytest.fixture()
def cfg(tmp_path):
    """A Config pointed at a throwaway workspace."""
    return Config(
        api_key="test-key-not-used",
        model="scripted",
        workspace=tmp_path,
        max_turns=6,
        max_repeat_call=3,
        context_budget=10**9,  # effectively never compacts in the happy-path tests
        summary_max_chars=200,
    )


@pytest.fixture()
def llm():
    return ScriptedLLM()


@pytest.fixture()
def agent(cfg, llm):
    """An Agent wired to the scripted model, one clean session 'w-test'."""
    from min_agent.loop import Agent

    trace = Tracer("w-test", console=False)
    instance = Agent(config=cfg, user="alice", session_id="w-test", llm=llm, trace=trace)
    yield instance
    instance.close()


@pytest.fixture()
def empty_session(cfg):
    return SessionStore(cfg.sessions_root).open("alice", "w-test", create=True)


@pytest.fixture()
def todo_store(cfg):
    return TodoStore(cfg.sessions_root / "alice" / "w-test" / "todo.json")