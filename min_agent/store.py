"""The session store: every window lives under ``.sessions/<user>/<id>/``.

Keeping sessions *inside the repo directory* (not ~/.cache or a temp dir)
means two terminal windows pointed at the same checkout truly share one store:
window 2 can resume the conversation window 1 left behind, and vice versa.
"""

from __future__ import annotations

import re
from pathlib import Path

from .session import Session

_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class SessionStore:
    def __init__(self, root: Path):
        self.root = root

    # -- paths --------------------------------------------------------------
    def node_dir(self, user: str, session_id: str) -> Path:
        return self.root / _safe(user) / _safe(session_id)

    # -- operations ---------------------------------------------------------
    def open(self, user: str, session_id: str, *, create: bool = True) -> Session:
        node = self.node_dir(user, session_id)
        if create:
            node.mkdir(parents=True, exist_ok=True)
            meta_path = node / "meta.json"
            if not meta_path.exists():
                return Session.create(session_id, user, node)
        return Session.load(node)

    def exists(self, user: str, session_id: str) -> bool:
        return self.node_dir(user, session_id).exists()

    def list_sessions(self, user: str) -> list[dict]:
        """All sessions for ``user``, newest first."""
        rows: list[dict] = []
        user_dir = self.root / _safe(user)
        if not user_dir.exists():
            return rows
        for node in sorted(user_dir.iterdir()):
            if not node.is_dir():
                continue
            meta = node / "meta.json"
            if not meta.exists():
                continue
            import json

            row = json.loads(meta.read_text(encoding="utf-8"))
            row["user"] = user
            row["dir"] = str(node)
            rows.append(row)
        rows.sort(key=lambda r: r.get("updated_at", 0), reverse=True)
        return rows


def _safe(segment: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in segment)
    return cleaned or "default"