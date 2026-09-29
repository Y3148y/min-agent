"""Tests for the filesystem helpers: path containment and atomic replacement."""

from __future__ import annotations

import json
import threading

import pytest

from min_agent import paths
from min_agent.paths import atomic_write_text, ensure_within, safe_segment


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


def test_module_exports_are_the_three_helpers():
    assert paths.atomic_write_text is atomic_write_text
    assert paths.ensure_within is ensure_within
    assert paths.safe_segment is safe_segment
