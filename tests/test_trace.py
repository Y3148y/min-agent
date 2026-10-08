"""Tests for the tracer: file-name construction, and the ordering guarantees
that the single ``_lock`` is there to provide.
"""

from __future__ import annotations

import io
import threading

import pytest

from min_agent.paths import ensure_within, safe_segment
from min_agent.trace import Tracer


# --------------------------------------------------------------------------- #
# path construction
# --------------------------------------------------------------------------- #


def test_prefix_and_session_keep_their_documented_shape(tmp_path):
    """The CLI passes prefix="<user>."; the on-disk name must not drift."""
    root = tmp_path / "traces"
    with Tracer("w", traces_root=root, prefix="alice.", console=False) as tr:
        assert tr.path.name == "alice.w.jsonl"

    with Tracer("w", traces_root=root, console=False) as tr:
        assert tr.path.name == "w.jsonl"


def test_unicode_window_names_survive_sanitising(tmp_path):
    """isalnum is Unicode-aware on purpose -- CJK window names stay readable."""
    with Tracer("周末规划", traces_root=tmp_path, prefix="张三.", console=False) as tr:
        assert tr.path.name == "张三.周末规划.jsonl"


def test_unknown_emit_kind_is_rejected():
    """The trace contract is enforced at the call site: a typo'd kind must be a
    loud ValueError, not a silent JSONL line the console renderer ignores."""
    tracer = Tracer("w", console=False)
    with pytest.raises(ValueError, match="unknown trace event kind"):
        tracer.emit("finale", text="boom")
    tracer.close()


@pytest.mark.parametrize(
    "user, session_id",
    [
        ("../..", "main"),
        ("..", ".."),
        ("a/b", "c\\d"),
        ("....//....//", "x"),
    ],
)
def test_traversal_in_a_window_name_cannot_escape_the_traces_root(
    tmp_path, user, session_id
):
    """Regression: the tracer interpolated both halves raw.

    `--user ../..` produced `.traces/../../.main.jsonl`; every parent component
    already existed, so the open succeeded and the agent wrote outside the
    workspace while the session store beside it was sanitising correctly.
    """
    root = tmp_path / "traces"
    with Tracer(session_id, traces_root=root, prefix=f"{user}.", console=False) as tr:
        assert tr.path.resolve().is_relative_to(root.resolve())
        assert tr.path.parent.resolve() == root.resolve()

    # And the file really is inside, not merely constructed that way.
    assert list(root.glob("*.jsonl"))


def test_ensure_within_rejects_an_escape(tmp_path):
    root = tmp_path / "traces"
    root.mkdir()
    assert ensure_within(root, root / "ok.jsonl") == root / "ok.jsonl"
    with pytest.raises(ValueError):
        ensure_within(root, root / ".." / "escaped.jsonl")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("alice", "alice"),
        ("a/b", "a_b"),
        ("a\\b", "a_b"),
        ("..", "__"),
        ("", "default"),
        ("...", "___"),
    ],
)
def test_safe_segment_reduces_to_one_component(raw, expected):
    assert safe_segment(raw) == expected
    assert "/" not in safe_segment(raw) and "\\" not in safe_segment(raw)


# --------------------------------------------------------------------------- #
# concurrency
# --------------------------------------------------------------------------- #


def test_lock_serialises_seq_numbers_under_contention(tmp_path):
    """Every emitted record needs a distinct, gap-free seq.

    This is the one place the project does real locking: the tool pool fans out
    four workers that all call emit() at once, so the increment, the append and
    the file write have to happen as a unit or the trace interleaves and the
    sequence numbers collide.
    """
    root = tmp_path / "traces"
    with Tracer("s", traces_root=root, console=False, stream=io.StringIO()) as tr:
        def worker(n: int) -> None:
            for i in range(25):
                tr.emit("tool_result", worker=n, i=i)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        path = tr.path

    assert sorted(ev.seq for ev in tr.events) == list(range(1, 101))
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 100


def test_read_trace_tolerates_a_torn_last_line(tmp_path):
    """A reader must not crash on a half-written line: the window may have been
    killed mid-append, and `min-agent trace` is how you find out what happened."""
    from min_agent.trace import read_trace

    path = tmp_path / "w.jsonl"
    path.write_text(
        '{"kind": "user", "text": "hi"}\n{"kind": "final", "text": "hel', encoding="utf-8"
    )
    events = read_trace(path)
    assert len(events) == 1
    assert events[0]["kind"] == "user"


def test_mechanical_compact_renders_the_note_not_none():
    """Regression: the compact renderer printed before/after/est_tokens for
    every compact event, so a mechanical eviction showed 'None -> None'."""
    buf = io.StringIO()
    tracer = Tracer("w", console=True, stream=buf)
    tracer.emit("compact", mechanical=True, note="evicted oldest complete turns")
    out = buf.getvalue()
    assert "evicted oldest complete turns" in out
    assert "None" not in out
