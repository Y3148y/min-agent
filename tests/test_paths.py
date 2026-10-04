"""Tests for the filesystem helpers: path containment and atomic replacement."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from min_agent import paths
from min_agent.paths import atomic_write_text, ensure_within, safe_segment
from min_agent.trace import Tracer


# --------------------------------------------------------------------------- #
# safe_segment
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("alice", "alice"),
        ("a/b", "a_b"),
        ("a\\b", "a_b"),
        ("..", "__"),
        ("", "default"),
        ("...", "___"),
        ("C:\\Windows", "C__Windows"),
    ],
)
def test_safe_segment_reduces_to_one_component(raw, expected):
    assert safe_segment(raw) == expected
    assert "/" not in safe_segment(raw) and "\\" not in safe_segment(raw)


def test_unicode_names_pass_through_untouched():
    assert safe_segment("周末规划") == "周末规划"
    assert safe_segment("张三") == "张三"


def test_windows_reserved_device_names_are_neutralised():
    """Regression: CON/PRN/AUX/NUL/COM1-9/LPT1-9 are valid Unix-ish names and
    pass ``isalnum`` untouched, but on Windows mkdir("CON") fails and the whole
    window becomes unopenable.  They must collapse to the default, like the
    empty/unsafe cases do."""
    for name in ("CON", "con", "Prn", "aux", "NUL", "COM1", "lpt9"):
        assert safe_segment(name) == "default", name
    assert safe_segment("console") == "console"  # not a device name
    assert safe_segment("commit") == "commit"
    assert safe_segment("COM10") == "COM10"  # past the reserved range


def test_delegating_callers_no_longer_inline_their_own_sanitiser():
    """loop._safe_user and Session.create used to copy the sanitise rule; both
    must now go through the single implementation in paths."""
    import inspect

    from min_agent.loop import Agent
    from min_agent.session import Session

    assert "isalnum" not in inspect.getsource(Agent._safe_user)
    assert "isalnum" not in inspect.getsource(Session.create)


# --------------------------------------------------------------------------- #
# ensure_within
# --------------------------------------------------------------------------- #


def test_ensure_within_allows_a_path_inside_and_refuses_one_outside(tmp_path):
    root = tmp_path / "traces"
    root.mkdir()
    assert ensure_within(root, root / "ok.jsonl") == root / "ok.jsonl"
    assert ensure_within(root, root / "deep" / ".." / "ok.jsonl")
    with pytest.raises(ValueError):
        ensure_within(root, root / ".." / "escaped.jsonl")


# --------------------------------------------------------------------------- #
# atomic_write_text
# --------------------------------------------------------------------------- #


def test_write_is_all_or_nothing(tmp_path):
    target = tmp_path / "state.json"
    atomic_write_text(target, json.dumps({"v": 1}))
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 1}
    assert list(tmp_path.iterdir()) == [target], "scratch file was left behind"


def test_a_failed_rename_leaves_the_original_intact(tmp_path, monkeypatch):
    """The dangerous step is the rename; nothing partial should survive it."""
    target = tmp_path / "state.json"
    target.write_text('{"v": 0}', encoding="utf-8")

    def boom(self, other):
        raise OSError("simulated failure during replace")

    monkeypatch.setattr("pathlib.Path.replace", boom)
    with pytest.raises(OSError):
        atomic_write_text(target, '{"v": 1}')

    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 0}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_scratch_names_are_private_to_each_writer(tmp_path, monkeypatch):
    """Regression: the scratch file used to be a fixed `.tmp` suffix.

    Two processes saving the same file both opened the same scratch path, so
    they interleaved and one could rename the other's half-written bytes into
    place.  Pin the invariant directly: the scratch name carries pid + a random
    token, so it is never shared.
    """
    from pathlib import Path

    target = tmp_path / "state.json"
    target.write_text("{}", encoding="utf-8")
    real_replace = Path.replace
    seen: list[str] = []

    def record_then_replace(self, other):
        seen.append(self.name)
        return real_replace(self, other)

    monkeypatch.setattr("pathlib.Path.replace", record_then_replace)
    atomic_write_text(target, '{"v":1}')
    atomic_write_text(target, '{"v":2}')

    assert len(seen) == 2
    assert seen[0] != seen[1], "two writes shared a scratch filename"
    for name in seen:
        assert name.startswith("state.json.")


def test_concurrent_writers_never_leave_the_file_unparseable(tmp_path):
    """Eight threads, one destination, a reader watching the whole time.

    The property under test is that a reader never observes a *half-written*
    file.  Note what is deliberately *not* a failure: on Windows, opening a file
    for read while another thread is mid-``replace`` on it raises
    PermissionError, because CPython does not request FILE_SHARE_DELETE.  That
    refusal is the guarantee working -- the reader saw neither version rather
    than a splice of both.  Only a JSONDecodeError would mean a real tear.
    """
    target = tmp_path / "facts.json"
    target.write_text("[]", encoding="utf-8")
    torn: list[str] = []
    refused: list[str] = []

    def worker(n: int) -> None:
        for i in range(25):
            atomic_write_text(target, json.dumps({"writer": n, "i": i, "pad": "x" * 500}))

    def watcher() -> None:
        for _ in range(400):
            try:
                json.loads(target.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                torn.append(repr(exc))  # <- the only real failure
                return
            except PermissionError as exc:
                refused.append(repr(exc))  # <- Windows refusing mid-replace
            except FileNotFoundError:
                pass  # a rename in flight; retry the observation

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    watcher_thread = threading.Thread(target=watcher)
    watcher_thread.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    watcher_thread.join()

    assert not torn, f"observed a torn file: {torn}"
    assert [p.name for p in tmp_path.iterdir()] == ["facts.json"]


def test_replace_retries_through_a_windows_sharing_violation(tmp_path, monkeypatch):
    """The destination being read must not turn into a lost write.

    Regression found while testing this file: ``tmp.replace(path)`` raised
    ``PermissionError: [WinError 5]`` on Windows as soon as a second thread
    held the destination open, because the rename is ``MoveFileEx`` and
    CPython's ``open()`` does not pass FILE_SHARE_DELETE.  The first attempt
    fails; the retry once the reader lets go succeeds.
    """
    from pathlib import Path

    target = tmp_path / "state.json"
    target.write_text("{}", encoding="utf-8")
    real_replace = Path.replace
    calls = {"n": 0}

    def flaky_replace(self, other):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(13, "simulated sharing violation")
        return real_replace(self, other)

    monkeypatch.setattr("pathlib.Path.replace", flaky_replace)
    monkeypatch.setattr("min_agent.paths.time.sleep", lambda _s: None)

    atomic_write_text(target, '{"v":1}')

    assert calls["n"] == 2
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 1}
    assert list(tmp_path.iterdir()) == [target]


# --------------------------------------------------------------------------- #
# the transcript rewrite goes through it
# --------------------------------------------------------------------------- #


def test_transcript_survives_a_failed_rewrite(tmp_path, monkeypatch):
    """_rewrite_jsonl used to open "w", truncating before it wrote.

    Compaction and turn repair both rewrite the whole transcript, so an
    interrupted rewrite took the entire session with it.
    """
    from min_agent.store import SessionStore

    session = SessionStore(tmp_path).open("alice", "w", create=True)
    session.append("user", "one")
    session.append("assistant", "two")
    transcript = session.dir / "transcript.jsonl"
    before = transcript.read_text(encoding="utf-8")
    assert before.count("\n") == 2

    def boom(*_args, **_kwargs):
        raise OSError("simulated failure mid-rewrite")

    monkeypatch.setattr("min_agent.session.atomic_write_text", boom)
    with pytest.raises(OSError):
        session._rewrite_jsonl()

    assert transcript.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------- #
# resource bounds
# --------------------------------------------------------------------------- #


def test_in_memory_event_window_is_bounded(tmp_path):
    """Regression: Tracer.events was an unbounded list.

    Every event, including the full text of every tool result, was retained for
    the life of the process.  A long REPL grew the heap monotonically even
    though the JSONL on disk is the actual record.
    """
    with Tracer("s", traces_root=tmp_path, console=False, events_keep=50) as tr:
        for i in range(500):
            tr.emit("tool_result", i=i, pad="x" * 200)

        assert len(tr.events) == 50
        # The window keeps the *newest* events, so the eviction order is right.
        assert tr.events[-1].data["i"] == 499
        assert tr.last("tool_result").data["i"] == 499

        # The file on disk is unaffected -- unbounded on purpose.
        assert len(tr.path.read_text(encoding="utf-8").strip().splitlines()) == 500


def test_add_is_idempotent_on_normalised_text(tmp_path):
    """Regression: `todo.add` minted max(id)+1 on every call.

    The model retries tool calls (a turn that ran out of output budget, a tool
    that came back is_error), and each retry appended another copy of the same
    todo with a new id.
    """
    from min_agent.tools.todo import TodoStore

    store = TodoStore(tmp_path / "todo.json")
    first = store.add("买 菜")
    again = store.add("买菜")  # differs only in whitespace
    third = store.add("买 菜")

    assert first.id == again.id == third.id
    assert len(store.items) == 1


def test_add_still_allows_distinct_items_and_keeps_ids_unique(tmp_path):
    from min_agent.tools.todo import TodoStore

    store = TodoStore(tmp_path / "todo.json")
    a = store.add("买菜")
    b = store.add("取快递")
    assert (a.id, b.id) == (1, 2)
    assert len(store.items) == 2


def test_load_skips_a_foreign_row_without_losing_the_rest(tmp_path):
    """Regression: TodoStore.load ran TodoItem(**row) unconditionally, so one
    row from another schema raised TypeError and the whole list was lost."""
    import json as _json
    from min_agent.tools.todo import TodoStore

    good = TodoStore(tmp_path / "todo.json")
    good.add("买菜")
    good.save()

    data = _json.loads((tmp_path / "todo.json").read_text(encoding="utf-8"))
    data.insert(0, {"not": "a todo row"})
    (tmp_path / "todo.json").write_text(
        _json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )

    store = TodoStore(tmp_path / "todo.json")
    assert [i.text for i in store.items] == ["买菜"]


def test_out_of_range_limits_are_reported_by_name(monkeypatch):
    """Regression: MAX_TURNS=0 made range(1, 1) empty and the agent answered
    nothing at all, with no error anywhere.  Zero and negative were accepted
    for every numeric setting."""
    from min_agent.config import Config

    cfg = Config(api_key="k", model="m", workspace=Path("."), max_turns=0)
    problems = cfg.invalid_limits()
    assert any("MAX_TURNS" in p for p in problems)

    cfg = Config(
        api_key="k",
        model="m",
        workspace=Path("."),
        max_tool_result_chars=-1,
    )
    assert any("MAX_TOOL_RESULT_CHARS" in p for p in cfg.invalid_limits())

    good = Config(api_key="k", model="m", workspace=Path("."))
    assert good.invalid_limits() == []


def test_a_sub_second_tool_timeout_is_expressible(monkeypatch):
    """Regression: TOOL_TIMEOUT went through _env_int, so 2.5 raised
    ValueError and silently became 10."""
    from min_agent import config as config_mod

    monkeypatch.setenv("TOOL_TIMEOUT", "2.5")
    assert config_mod._env_float("TOOL_TIMEOUT", 10.0) == 2.5


def test_an_unparseable_env_value_warns_instead_of_vanishing(monkeypatch):
    from min_agent import config as config_mod

    monkeypatch.setenv("MAX_TURNS", "twelve")
    with pytest.warns(RuntimeWarning, match="MAX_TURNS"):
        assert config_mod._env_int("MAX_TURNS", 12) == 12


def test_module_exports_are_the_three_helpers():
    assert paths.atomic_write_text is atomic_write_text
    assert paths.ensure_within is ensure_within


# --------------------------------------------------------------------------- #
# single-writer lock
# --------------------------------------------------------------------------- #


def test_a_second_holder_is_refused(tmp_path):
    """The bug this exists for: two processes on one window each load a copy of
    the transcript, each append, each write their copy back -- and whichever
    finishes second deletes the other's turns, with no error anywhere.  Neither
    file was ever torn, so atomic writes could not have caught it.
    """
    lock_path = tmp_path / "w" / ".lock"
    first = paths.FileLock(lock_path, label="alice/w").acquire()
    with pytest.raises(paths.LockBusy) as excinfo:
        paths.FileLock(lock_path, label="alice/w").acquire()
    assert "another process" in str(excinfo.value)
    first.release()
    # Released: the next writer gets in, which is what a closed window means.
    paths.FileLock(lock_path, label="alice/w").acquire().release()


def test_release_is_idempotent_and_the_file_survives(tmp_path):
    lock_path = tmp_path / "w.lock"
    lock = paths.FileLock(lock_path).acquire()
    assert lock.held
    lock.release()
    lock.release()  # closing an already-closed window must not raise
    assert not lock.held
    assert lock_path.exists(), "the lock file is the marker, not the lock"


def test_an_unreferenced_lock_releases_itself(tmp_path):
    """The lock *is* the open handle, so GC closes it.

    Worth pinning because it is a trap rather than an accident: writing
    ``FileLock(p).acquire()`` and dropping the result looks like taking a lock
    and silently does not, and the only symptom is a second process walking in.
    """
    import gc

    lock_path = tmp_path / "w.lock"
    paths.FileLock(lock_path).acquire()  # no reference kept -- collected right here
    gc.collect()
    with paths.FileLock(lock_path, timeout=0.2, poll=0.01) as survivor:
        assert survivor.held, "the lock should have been released by collection"


def test_store_open_keeps_the_lock_alive_for_the_window(tmp_path):
    """The regression that matters: the lock must outlive the open() call.

    If the store let the FileLock go out of scope, the window would look guarded
    from outside while nothing held it -- and the bug it was added for would
    come straight back.
    """
    from min_agent.store import SessionStore

    store = SessionStore(tmp_path)
    session = store.open("alice", "w", create=True)
    try:
        with pytest.raises(paths.LockBusy):
            paths.FileLock(store.lock_path("alice", "w"), timeout=0.1, poll=0.01).acquire()
    finally:
        session.close()
    paths.FileLock(store.lock_path("alice", "w"), timeout=0.2, poll=0.01).acquire().release()


def test_two_different_windows_can_run_side_by_side(tmp_path):
    """The lock is per window, not per user: separate conversations, and
    serialising them would defeat the point of having windows at all."""
    from min_agent.store import SessionStore

    store = SessionStore(tmp_path)
    a = store.open("alice", "w1", create=True)
    b = store.open("alice", "w2", create=True)
    try:
        a.append("user", "in w1")
        b.append("user", "in w2")
        assert [m["content"] for m in a.messages] == ["in w1"]
        assert [m["content"] for m in b.messages] == ["in w2"]
    finally:
        a.close()
        b.close()


def test_a_timeout_waits_instead_of_failing_immediately(tmp_path):
    """Patrol-on-close and a slow flush can overlap a hand-off; a brief wait is
    friendlier than refusing, and the kernel drops the lock when a process dies
    so there is no stale-lock case to wait on."""
    lock_path = tmp_path / "w.lock"
    holder = paths.FileLock(lock_path).acquire()
    result: list[str] = []

    def grab():
        try:
            with paths.FileLock(lock_path, timeout=5.0, poll=0.01):
                result.append("got it")
        except paths.LockBusy:
            result.append("busy")

    waiter = threading.Thread(target=grab)
    waiter.start()
    holder.release()  # the window closes while the other process is waiting
    waiter.join(timeout=5)
    assert result == ["got it"]


def test_the_error_names_the_process_that_won(tmp_path):
    lock_path = tmp_path / "w.lock"
    held = paths.FileLock(lock_path, label="alice/w1").acquire()
    try:
        with pytest.raises(paths.LockBusy) as excinfo:
            paths.FileLock(lock_path, label="alice/w1", timeout=0.05, poll=0.01).acquire()
        message = str(excinfo.value)
        assert f"pid={os.getpid()}" in message
        assert "alice/w1" in message
    finally:
        held.release()


def test_owner_stamp_overwrites_in_place_not_append(tmp_path):
    """Regression: the lock file used to be opened in append mode, so a write
    after a seek landed at EOF anyway, growing the file by one fixed-width
    record per takeover, while _busy_message read the record at byte 1 -- the
    *first* owner, not the current one.  The file must stay one record wide.
    """
    lock_path = tmp_path / "w.lock"
    paths.FileLock(lock_path, label="first").acquire().release()
    paths.FileLock(lock_path, label="second").acquire().release()
    raw = lock_path.read_bytes()
    assert len(raw) == 1 + 63, f"lock file grew to {len(raw)} bytes"
    assert raw[0] == 0
    assert b"second" in raw, "the newest stamp must be the one in place"


def test_busy_message_names_the_current_holder_not_the_first(tmp_path):
    """Under the append-mode bug the stamp read by _busy_message was always the
    first process ever to touch the file, so the 'held by' hint pointed at a
    window that may already be long gone."""
    lock_path = tmp_path / "w.lock"
    paths.FileLock(lock_path, label="old-and-gone").acquire().release()
    current = paths.FileLock(lock_path, label="current-window").acquire()
    try:
        with pytest.raises(paths.LockBusy) as excinfo:
            paths.FileLock(lock_path, label="loser", timeout=0.05, poll=0.01).acquire()
        message = str(excinfo.value)
        assert "current-window" in message
        assert "old-and-gone" not in message
    finally:
        current.release()


def test_the_lock_does_not_block_readers(tmp_path):
    """Only writers serialise.  `min-agent sessions` and `min-agent trace` read
    a live window's files, and atomic replacement already makes that safe."""
    from min_agent.store import SessionStore

    store = SessionStore(tmp_path)
    session = store.open("alice", "w", create=True)
    session.append("user", "hello")
    try:
        assert [r["user"] for r in store.list_sessions("alice")] == ["alice"]
        assert session.dir.joinpath("meta.json").exists()
    finally:
        session.close()


def test_the_lock_sidecar_is_not_mistaken_for_a_session(tmp_path):
    from min_agent.store import SessionStore

    store = SessionStore(tmp_path)
    session = store.open("alice", "w", create=True)
    session.append("user", "hello")  # this is what writes meta.json
    session.close()

    node = store.node_dir("alice", "w")
    assert {p.name for p in node.iterdir()} >= {".lock", "meta.json"}
    assert [r["id"] for r in store.list_sessions("alice")] == ["w"]


def test_opening_a_missing_window_without_create_does_not_invent_one(tmp_path):
    """``--session`` naming a window that does not exist yet is a legitimate
    read (trace, sessions): it must not create a directory as a side effect."""
    from min_agent.store import SessionStore

    store = SessionStore(tmp_path)
    assert store.exists("alice", "ghost") is False
    store.open("alice", "ghost", create=False)
    assert store.exists("alice", "ghost") is False

    assert paths.safe_segment is safe_segment
