"""Sessions: isolation between windows, resume across processes, repairs."""

from __future__ import annotations

import json

import pytest

from min_agent.session import Session, SessionMeta, _drop_hanging_tool_refs
from min_agent.store import SessionStore
from min_agent.tools import TodoStore


def _session(cfg, sid, user="alice", create=True):
    store = SessionStore(cfg.sessions_root)
    return store.open(user, sid, create=create)


# --------------------------------------------------------------------------- #
# isolation
# --------------------------------------------------------------------------- #


def test_two_windows_are_independent(cfg):
    a = _session(cfg, "w1")
    b = _session(cfg, "w2")
    a.append("user", "在 w1 说的话")
    assert b.messages == []
    assert len(a.messages) == 1
    # each window persists to its own transcript
    assert not (cfg.sessions_root / "alice" / "w2" / "transcript.jsonl").exists()
    assert (cfg.sessions_root / "alice" / "w1" / "transcript.jsonl").exists()


def test_todo_isolation_between_windows(cfg):
    ta = TodoStore(cfg.sessions_root / "alice" / "w-todo" / "todo.json")
    tb = TodoStore(cfg.sessions_root / "alice" / "w-todo2" / "todo.json")
    ta.add("买牛奶")
    assert [i.text for i in ta.list_items()] == ["买牛奶"]
    assert tb.list_items() == []


def test_todo_persists_and_supports_complete(cfg):
    store = TodoStore(cfg.sessions_root / "alice" / "w-todo3" / "todo.json")
    store.add("写测试")
    store.complete(text_contains="写测试")
    assert [i.text for i in store.list_items("pending")] == []
    assert [i.text for i in store.list_items("done")] == ["写测试"]


def test_session_id_is_sanitised(cfg):
    s = _session(cfg, "a/b;c**")
    assert all(c.isalnum() or c in "-_" for c in s.meta.id)
    assert s.dir.is_dir()
    assert s.dir.name == s.meta.id


# --------------------------------------------------------------------------- #
# resume / persistence
# --------------------------------------------------------------------------- #


def test_transcript_round_trip_resume(cfg):
    s = _session(cfg, "resume")
    s.append("user", "第一句")
    s.append("assistant", "回应一")
    s.append("user", "第二句")
    meta = SessionMeta(id=s.meta.id, user=s.meta.user)
    fresh = Session(meta, s.dir)
    assert len(fresh.messages) == 3
    assert fresh.messages[0]["content"] == "第一句"
    assert fresh.messages[2]["content"] == "第二句"


def test_store_reopen_returns_same_transcript(cfg):
    """Reopening a window must see everything the previous process persisted.

    ``s1.close()`` first is not incidental: a window has exactly one writer, and
    the store hands the lock to whoever opened it.  Two live handles on one
    window is the bug this lock exists to prevent, so the test has to model the
    real sequence -- leave the window, then come back to it.
    """
    store = SessionStore(cfg.sessions_root)
    s1 = store.open("alice", "reopen", create=True)
    s1.append("user", "hello")
    s1.append("assistant", "hi")
    s1.close()
    s2 = store.open("alice", "reopen", create=False)
    assert [m["content"] for m in s2.messages] == ["hello", "hi"]


def test_turn_count_and_tool_calls_tracked(cfg):
    s = _session(cfg, "meta")
    s.append("user", "q1")
    s.append("assistant", [{"type": "tool_use", "id": "a", "name": "calc"}])
    s.append_tool_results([{"type": "tool_result", "tool_use_id": "a", "content": "4"}])
    assert s.meta.turn_count == 1
    assert s.meta.tool_calls == 1


def test_created_window_has_meta_immediately(cfg):
    """Session.create must persist meta.json before the first append.

    list_sessions (store.py) skips any window without meta.json, so without
    immediate persistence a freshly created, idle window would silently
    disappear, and a later load would forge user="default".
    """
    from min_agent.paths import safe_segment

    store = SessionStore(cfg.sessions_root)
    store.open("alice", "idle", create=True).close()
    meta_path = cfg.sessions_root / safe_segment("alice") / "idle" / "meta.json"
    assert meta_path.exists()
    rows = store.list_sessions("alice")
    assert any(r["dir"] == str(meta_path.parent) for r in rows)


def test_created_window_reloads_real_user(cfg):
    from min_agent.session import Session

    s = Session.create("fresh", "bob", cfg.sessions_root / "bob" / "fresh")
    s.close()
    reloaded = Session.load(cfg.sessions_root / "bob" / "fresh")
    assert reloaded.meta.user == "bob"


def test_corrupt_meta_degrades_to_defaults(cfg):
    """Regression: SessionMeta(**json.loads(...)) crashed the whole window on a
    torn meta.json.  MemoryStore already tolerated this; session and the
    `sessions` listing must do the same."""
    import json as _json
    from min_agent.session import Session

    node = cfg.sessions_root / "alice" / "torn"
    node.mkdir(parents=True)
    node.joinpath("meta.json").write_text("{ not json", encoding="utf-8")
    s = Session.load(node)
    assert s.meta.id == "torn"
    assert s.meta.user == "default"

    # forward-compatible: extra keys are tolerated, missing ones default
    node.joinpath("meta.json").write_text(
        _json.dumps({"id": "torn", "user": "alice", "future_field": 123}),
        encoding="utf-8",
    )
    s = Session.load(node)
    assert s.meta.user == "alice"


def test_list_sessions_skips_a_corrupt_meta(cfg):
    import json as _json
    from min_agent.paths import safe_segment

    store = SessionStore(cfg.sessions_root)
    good = store.open("alice", "good", create=True)
    good.close()
    user_dir = cfg.sessions_root / safe_segment("alice")
    (user_dir / "torn2").mkdir(parents=True)
    (user_dir / "torn2" / "meta.json").write_text("{ nope", encoding="utf-8")
    (user_dir / "badstr").mkdir(parents=True)
    (user_dir / "badstr" / "meta.json").write_text(
        _json.dumps({"id": "badstr", "user": "alice", "updated_at": "yesterday"}),
        encoding="utf-8",
    )
    rows = store.list_sessions("alice")
    ids = [r["id"] for r in rows]
    assert "good" in ids
    assert "torn2" not in ids  # unreadable json is skipped, not a crash
    assert "badstr" in ids  # readable but sloppy -> kept
    assert all(isinstance(r["updated_at"], (int, float)) for r in rows)


# --------------------------------------------------------------------------- #
# repair: transcript must stay a valid alternating sequence
# --------------------------------------------------------------------------- #


def test_hanging_tool_use_removes_pair_without_result(cfg):
    """A crash between append() and append_tool_results() must be repaired."""
    s = _session(cfg, "repair1")
    s.append("user", "q")
    s.append(
        "assistant",
        [{"type": "tool_use", "id": "t", "name": "calculator", "input": {"expression": "1+1"}}],
    )
    repaired = _drop_hanging_tool_refs(s.messages)
    assert len(repaired) == 1
    assert repaired[0]["role"] == "user"


def test_torn_transcript_line_does_not_kill_the_window(cfg):
    """Regression: _load_transcript ran json.loads unguarded, so a half-written
    last line (a crash mid-append) made the whole window unopenable -- while the
    module docstring promised the opposite."""
    import json as _json

    node = cfg.sessions_root / "alice" / "torn-transcript"
    node.mkdir(parents=True)
    (node / "meta.json").write_text(
        _json.dumps({"id": "torn-transcript", "user": "alice"}), encoding="utf-8"
    )
    (node / "transcript.jsonl").write_text(
        _json.dumps({"role": "user", "content": "first"}) + "\n"
        + _json.dumps({"role": "assistant", "content": "second"}) + "\n"
        + '{"role": "user", "content": "half',
        encoding="utf-8",
    )
    s = Session.load(node)
    assert [m["role"] for m in s.messages] == ["user", "assistant"]


def test_dangling_tool_result_is_dropped(cfg):
    s = _session(cfg, "repair2")
    s.append("user", "q")
    s.append("user", [{"type": "tool_result", "tool_use_id": "lost", "content": "x"}])
    repaired = _drop_hanging_tool_refs(s.messages)
    assert len(repaired) == 1


def test_trailing_plain_answer_is_kept(cfg):
    s = _session(cfg, "repair3")
    s.append("user", "q")
    s.append("assistant", [{"type": "text", "text": "answer"}])
    assert _drop_hanging_tool_refs(s.messages) == s.messages


def test_tool_pair_survives_repair(cfg):
    s = _session(cfg, "repair4")
    s.append("user", "q")
    s.append("assistant", [{"type": "tool_use", "id": "t", "name": "calculator", "input": {}}])
    s.append_tool_results([{"type": "tool_result", "tool_use_id": "t", "content": "2"}])
    repaired = _drop_hanging_tool_refs(s.messages)
    assert len(repaired) == 3


def test_alternation_holds_after_tool_round_trip(cfg):
    """The provider's alternation rule must hold for the persisted transcript."""
    s = _session(cfg, "alt")
    s.append("user", "q1")
    s.append("assistant", [{"type": "tool_use", "id": "t", "name": "calculator", "input": {}}])
    s.append_tool_results([{"type": "tool_result", "tool_use_id": "t", "content": "2"}])
    s.append("assistant", "2")
    roles = [m["role"] for m in s.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))