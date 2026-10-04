"""The session store: every window lives under ``.sessions/<user>/<id>/``.

Keeping sessions *inside the repo directory* (not ~/.cache or a temp dir)
means two terminal windows pointed at the same checkout truly share one store:
window 2 can resume the conversation window 1 left behind, and vice versa.

Sharing is only safe one way round.  A window directory is single-writer, and
:meth:`SessionStore.open` is where that is enforced: it takes an exclusive lock
(``<session>/.lock``) and hands it to the :class:`~min_agent.session.Session`,
which releases it on ``close()``.  Two processes on one window used to each load
a copy of the transcript, each append their own turns, and each write their copy
back -- so whichever finished second deleted the other's turns, with no error
anywhere.  Atomic writes do not help there, because neither file was ever torn.

The lock is per *window*, not per user: two windows of the same user are
independent conversations and are meant to run side by side.  The shared
long-term memory file is the opposite case -- genuinely multi-writer -- and is
handled by merging on write instead, in
:meth:`~min_agent.memory.MemoryStore.save`.
"""

from __future__ import annotations

import re
from pathlib import Path

from .paths import FileLock, safe_segment
from .session import Session

_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class SessionStore:
    def __init__(self, root: Path):
        self.root = root

    # -- paths --------------------------------------------------------------
    def node_dir(self, user: str, session_id: str) -> Path:
        return self.root / safe_segment(user) / safe_segment(session_id)

    def lock_path(self, user: str, session_id: str) -> Path:
        return self.node_dir(user, session_id) / ".lock"

    # -- operations ---------------------------------------------------------
    def open(self, user: str, session_id: str, *, create: bool = True) -> Session:
        """Open ``user``'s window as *the* writer, or raise ``LockBusy``.

        Takes an exclusive lock for the lifetime of the returned Session.  Read
        the lock owner out of the exception to tell the user which process to go
        and close; a stale lock is not possible, the kernel drops it with the
        process.
        """
        node = self.node_dir(user, session_id)
        if create:
            node.mkdir(parents=True, exist_ok=True)
        if not node.is_dir():
            # Nothing to write to.  Session.load will still hand back an empty
            # session, and the first write will fail loudly on the missing dir.
            return Session.load(node)
        lock = FileLock(
            self.lock_path(user, session_id), label=f"{safe_segment(user)}/{safe_segment(session_id)}"
        ).acquire()
        try:
            meta_path = node / "meta.json"
            if create and not meta_path.exists():
                return Session.create(session_id, user, node, lock=lock)
            return Session.load(node, lock=lock)
        except BaseException:
            lock.release()
            raise

    def exists(self, user: str, session_id: str) -> bool:
        return self.node_dir(user, session_id).exists()

    def list_sessions(self, user: str) -> list[dict]:
        """All sessions for ``user``, newest first."""
        rows: list[dict] = []
        user_dir = self.root / safe_segment(user)
        if not user_dir.exists():
            return rows
        for node in sorted(user_dir.iterdir()):
            if not node.is_dir():
                continue
            meta = node / "meta.json"
            if not meta.exists():
                continue
            import json

            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue  # a torn/corrupt meta must not take down `sessions`
            if not isinstance(data, dict) or "id" not in data:
                continue
            row = dict(data)
            row["user"] = user
            row["dir"] = str(node)
            updated = row.get("updated_at", 0)
            row["updated_at"] = updated if isinstance(updated, (int, float)) else 0
            rows.append(row)
        rows.sort(key=lambda r: r.get("updated_at", 0), reverse=True)
        return rows
