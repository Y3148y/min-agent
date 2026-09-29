"""Filesystem helpers for a project that turns user input into paths and keeps
its state in small JSON files.

Three concerns, each of which had a hole at some point:

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
* **Writers.**  Atomic writes make concurrent *readers* safe; they do nothing
  for two concurrent *writers*, which is why :class:`FileLock` exists.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import IO


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


# --------------------------------------------------------------------------- #
# single-writer lock
# --------------------------------------------------------------------------- #
# A session directory is *one writer, many readers*: the live window appends to
# the transcript and rewrites meta.json, while `min-agent sessions` and
# `min-agent trace` read from the side.  The atomic writes above already make
# the readers safe -- they see the old file or the new one.  What they cannot
# fix is two *writers*, because the real state is the in-memory message list:
# two processes each load a copy, each append, and each write its own copy back,
# so one silently deletes the other's turns.  No amount of atomicity helps,
# because neither file was ever torn.
#
# The lock is advisory and held for the lifetime of the window, not per write.
# A per-write lock would serialise the writes and still lose the turns, since
# the second writer's in-memory list is already stale by the time it gets the
# lock.
#
# It is an OS lock (``flock`` / ``LockFile``), not a marker file that has to be
# cleaned up: the kernel drops it when the process dies, so a window that was
# killed with Ctrl-C or a hard crash does not leave the next one locked out.

# Byte 0 is the byte we lock on; the owner string follows it, fixed width so a
# new owner can overwrite it in place without truncating a locked file.
_OWNER_OFFSET = 1
_OWNER_WIDTH = 64


class LockBusy(RuntimeError):
    """The lock is held by somebody else -- normally another live window."""


class FileLock:
    """An exclusive advisory lock on a sidecar file, held until released.

    Used as a context manager, or via :meth:`acquire` / :meth:`release` when the
    lifetime is the owner's (a window that stays open for hours, not a ``with``).

    **Keep a reference to the object.**  The lock is the open file handle, so an
    object nobody holds gets collected, the handle closes and the lock is gone --
    without a word.  ``FileLock(p).acquire()`` on its own is a no-op that looks
    like a lock.  Bind it (``lock = FileLock(p).acquire()``) or store it.
    """

    def __init__(
        self,
        path: Path,
        *,
        label: str = "",
        timeout: float = 0.0,
        poll: float = 0.05,
    ):
        self.path = path
        self.label = label
        self.timeout = timeout
        self.poll = poll
        self._fh: IO[bytes] | None = None

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    @property
    def held(self) -> bool:
        return self._fh is not None

    def acquire(self) -> "FileLock":
        """Take the lock and return ``self``, or raise :class:`LockBusy`.

        The returned object owns the lock -- see the class docstring.
        """
        if self._fh is not None:
            raise RuntimeError(f"lock already held: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                fh.write(b"\0")  # a byte 0 exists, so byte 0 can be locked
                fh.flush()
            deadline = time.monotonic() + self.timeout
            while True:
                try:
                    _lock_exclusive(fh)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise LockBusy(self._busy_message()) from None
                    time.sleep(self.poll)
        except BaseException:
            fh.close()
            raise
        self._fh = fh
        self._stamp_owner()
        return self

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            _unlock(fh)
        finally:
            fh.close()

    # -- internals ----------------------------------------------------------
    def _stamp_owner(self) -> None:
        """Record who holds it, so the loser's error message can say who won."""
        assert self._fh is not None
        who = f"pid={os.getpid()} {self.label}".encode("utf-8", "replace")
        self._fh.seek(_OWNER_OFFSET)
        self._fh.write(who[: _OWNER_WIDTH - 1].ljust(_OWNER_WIDTH - 1, b" "))
        self._fh.flush()

    def _busy_message(self) -> str:
        who = ""
        try:
            with open(self.path, "rb") as fh:
                fh.seek(_OWNER_OFFSET)
                who = fh.read(_OWNER_WIDTH - 1).decode("utf-8", "replace").strip()
        except OSError:
            pass
        held = f" (held by {who})" if who else ""
        return (
            f"another process already has this window open{held}: {self.path}. "
            f"Each window takes exactly one process; open a different --session."
        )


if os.name == "nt":
    import msvcrt

    def _lock_exclusive(fh: IO[bytes]) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(fh: IO[bytes]) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock_exclusive(fh: IO[bytes]) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fh: IO[bytes]) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

