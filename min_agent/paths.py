"""Filesystem helpers for a project that turns user input into paths and keeps
its state in small JSON files.

Two concerns, both of which had a hole at some point:

* **Names.**  ``--user`` and ``--session`` come straight off the command line,
  and two components turn them into file names: :class:`~min_agent.store.SessionStore`
  and :class:`~min_agent.trace.Tracer`.  The store sanitised; the tracer
  interpolated the raw values, so a session id like ``../..`` walked the trace
  file out of ``.traces/`` while the session directory beside it was being
  filtered all along.  Keeping the rule here means a new writer cannot quietly
  reintroduce the gap.
* **Writes.**  Every state file is rewritten in full, so a plain
  ``write_text`` leaves a torn file for any reader that catches it mid-write --
  and the obvious ``with_suffix('.tmp')`` scratch file has a *fixed* name, so
  two processes saving the same file race on it.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path


def safe_segment(segment: str) -> str:
    """Reduce ``segment`` to a single harmless path component.

    Anything outside ``[alnum-_]`` becomes ``_``, so separators, ``..`` and
    Windows drive/UNC forms all collapse; an empty result becomes ``default``.

    ``str.isalnum`` is Unicode-aware, so CJK window names survive intact --
    that is deliberate, not an oversight.
    """
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in segment)
    return cleaned or "default"


def ensure_within(root: Path, path: Path) -> Path:
    """Return ``path`` unchanged, or raise if it resolves outside ``root``.

    Second line of defence.  Sanitising the segments is the real guard; this
    catches anything that still manages to climb out, and it fails loudly
    rather than silently writing somewhere else.
    """
    root = root.resolve()
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"refusing to write outside {root}: {path}")
    return path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Replace ``path`` with ``text`` in one step, or leave it as it was.

    Three things this buys over a bare ``path.write_text(...)``:

    * **No torn file.**  A reader sees either the old content or the new one,
      never half of each.  This matters most for ``transcript.jsonl``, which is
      rewritten in full after compaction and after turn repair: a truncating
      write would lose the entire session if the process died midway.
      (On Windows this needs the retry in :func:`_replace_with_retry`, because
      an atomic rename onto a file someone is reading fails outright.)
    * **No shared scratch name.**  A fixed ``.tmp`` suffix means two processes
      saving the same file scribble over the same temporary, and one of them
      can promote the other's half-written bytes.  The suffix here carries pid
      plus a random token, so the scratch file is private to its writer while
      still landing on the same filesystem, which ``replace`` needs in order to
      stay atomic.
    * **Durability.**  The payload is flushed and ``fsync``-ed before the
      rename, so a power loss cannot leave the name pointing at empty or
      half-written contents.  The *directory* entry is not fsync-ed, so on a
      power cut the rename itself can still be lost -- that needs a
      platform-specific directory handle and did not seem worth it for state
      this small and this recoverable.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding=encoding, newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        _replace_with_retry(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _replace_with_retry(tmp: Path, path: Path, attempts: int = 6) -> None:
    """``tmp.replace(path)``, retried through Windows' sharing violations.

    On POSIX ``rename()`` onto a path somebody is reading is atomic and simply
    succeeds.  On Windows it is ``MoveFileEx``, which fails with
    ``ERROR_ACCESS_DENIED`` while any other handle holds the destination open
    without ``FILE_SHARE_DELETE`` -- and CPython's ``open()`` does not request
    that flag.  So a concurrent reader (``min-agent trace`` printing a window,
    another window listing sessions) turns our rename into a PermissionError.

    That is transient by nature: the reader closes and the rename then works,
    so a short backoff is the right response.  The alternative -- going back to
    a truncating write -- trades a rare retryable error for a window in which
    the file is half-written, which is strictly worse.
    """
    for attempt in range(attempts):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.01 * (2**attempt))
